# RAHMAD: Retrieval-Augmented Hybrid Multi-Agent Debate
## System Architecture Documentation

The RAHMAD framework is a modular, AI-driven pipeline designed for comprehensive fake news and misinformation detection. It combines lightweight multimodal early prediction with heavyweight Retrieval-Augmented Generation (RAG), external visual search, and a constitutional LLM reasoning layer to arrive at a final verdict.

Below is the detailed architectural breakdown of each core module in the system.

Note: The LLM local model and OpenRouter model use in this system can be changed in the env file. 

---

### 1. Early Predictor Module
**Location:** `model/early_predictor.py`

The Early Predictor (EP) serves as a fast, lightweight multimodal initial estimate. It classifies the article using a fine-tuned multimodal model and provides grounding context to downstream modules. It acts strictly as an evidentiary signal and does not gate or skip downstream modules (such as Multi-Agent Debate).

**Architecture & Flow:**
1. **Input Preprocessing:** Text is cleaned (HTML tags, URLs, Emojis removed) to match the training distribution. Images are processed via `CLIPProcessor`.
2. **Feature Extraction:** Utilizes `openai/clip-vit-base-patch32` for both text and vision encoding. 
3. **LoRA Adapters:** Parameter-Efficient Fine-Tuning (PEFT) LoRA adapters are applied to the CLIP attention layers.
4. **Fusion:** The text embeddings `[512]` and image embeddings `[512]` are concatenated into a single fused tensor `[1024]`.
5. **Classification (MLP Head):** The fused tensor is passed through a single-layer Multi-Layer Perceptron (Linear `1024 -> 1` + Sigmoid activation) to output a scalar probability between `0` (Real) and `1` (Fake).
6. **Thresholding & Signal Pass-Through:**
   - **p >= 0.60:** Preliminary Fake.
   - **p <= 0.40:** Preliminary Real.
   - **0.40 < p < 0.60:** Uncertain.
   - All news items unconditionally proceed through the full Knowledge Retrieval, Reverse Image Search, and Multi-Agent Debate pipelines regardless of EP probability.

---

### 2. Knowledge Retrieval Module
**Location:** `model/knowledge_retrieval.py`, `backend/news_spider.py`, `backend/domain_whitelist.py`

The Knowledge Retrieval (KR) module is responsible for fetching, filtering, and synthesizing external, verified news to provide corroborating evidence (or debunking context) to both the Multi-Agent Debate (MAD) orchestrator and the Judge module.

#### Model & Execution Engine:
- **Semantic Verification Embedder:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense semantic embeddings).
- **Inference Engine & LLM Summarizer:** 
  - **Local Mode (`USE_LOCAL_LLM=true`):** Connects to the local Ollama OpenAI-compatible endpoint (`LOCAL_LLM_URL`, default `http://localhost:11434/v1/chat/completions`). The active model is specified by `LOCAL_LLM_MODEL` in `.env` (currently **`qwen2.5:7b`**, with code default fallback **`llama3:8b`**).
  - **Cloud Fallback (`USE_LOCAL_LLM=false`):** Queries OpenRouter API using `nvidia/nemotron-3-ultra-550b-a55b:free` with a resilient multi-model fallback chain (`nemotron-3-nano-omni-30b-a3b-reasoning`, `inclusionai/ling-3.0-flash`, `poolside/laguna-m.1`).

#### Architecture & Flow:
1. **Search & Multi-Source Retrieval:**
   - **Query Sanitization:** Automatically extracts the concise headline/claim from the input text, stripping news wire prefixes and datelines (e.g. `SINGAPORE/JAKARTA, Sept 9 (Reuters) -`) so search engines receive clean, high-precision search keywords.
   - **Primary Endpoint:** Queries the DuckDuckGo HTML endpoint (`https://html.duckduckgo.com/html/`) and `duckduckgo_search` (`DDGS.news()`).
   - **Tertiary Fallback (SerpAPI Google Search):** If DuckDuckGo hits a rate limit (`403 Ratelimit`) or returns fewer than 3 URLs, automatically falls back to querying Google Search via SerpAPI using `SERPAPI_API_KEY`, ensuring uninterrupted retrieval.
2. **Strict Domain Whitelisting:**
   - URLs are immediately checked against an authoritative list of verified, reputable international and national news publishers (`backend/domain_whitelist.py`, e.g., Reuters, AP, BBC, Straits Times, NYT).
   - Unverified blogs, opinion sites, and tabloids are filtered out prior to scraping.
3. **Concurrent Scrapy Spider & Heuristic Extraction:**
   - **Isolated Scrapy Subprocess:** Spawns an asynchronous Scrapy spider (`backend/news_spider.py`) inside an ephemeral temporary directory (`tempfile.TemporaryDirectory`) with 8 concurrent connections and a 25-second timeout.
   - **Structured Parsing:** Extracts clean body text, title, and published timestamp using `trafilatura` (with `BeautifulSoup` fallback).
   - **Direct Extraction Fallback:** If the spider yields 0 items, executes an in-process HTTP request using `requests` + `trafilatura`/`BeautifulSoup`.
   - **Heuristic Filtering:** Performs title deduplication (case-insensitive set), drops text shorter than 100 characters, and validates article recency against `max_age_years` (default <= 3 years old). Caps surviving articles to `max_articles` (default 5).
4. **Dense Semantic Verification (MiniLM Embedding & Paragraph Chunking):**
   - Splits surviving article text into paragraph chunks (>50 characters).
   - Uses `all-MiniLM-L6-v2` to compute cosine similarity between the claim embedding and each chunk embedding.
   - Retains the top 3 matching chunks per article that meet or exceed the semantic threshold (`similarity_threshold >= 0.35`), stripping irrelevant boilerplate or tangential reporting.
5. **Structured Fact-Outline LLM Synthesis:**
   - Passes verified snippets into the local Ollama `qwen2.5:7b` or `llama3:8b` at low temperature (`0.2`) to prevent hallucination.
   - Prompts the LLM under strict forensic rules to generate a structured 3-part key-fact outline:
     1. *Key Entities, Names, and Roles involved*
     2. *Monetary Amounts, Legal Rulings, or Official Outcomes*
     3. *Timeline of Events and Corroborating Evidence*
   - Negative constraints strictly prohibit inferring or extrapolating dates, locations, or unstated facts (mandating `"Date not specified"` if a year/date is absent). Gracefully falls back to raw snippets if LLM generation fails.
6. **Dedicated Forensic Logging (`output/KR_output.txt`):**
   - Automatically writes an end-to-end, timestamped forensic audit report to `output/KR_output.txt` logging:
     - **[1] Input Claim & Execution Backend** (Local LLM vs. OpenRouter).
     - **[2] Link Retrieval & Whitelist Audit:** Full list of candidate URLs tagged `[WHITELISTED]` or `[FILTERED OUT]`.
     - **[3] Extracted Article Metadata:** Article titles, verified URLs, and dates.
     - **[4] Semantic Verification Snippets:** All surviving chunks tagged with their source headlines and similarity pass status.
     - **[5] LLM Synthesis Output:** The verbatim generated key-facts outline.
     - **[6] Final Pipeline Status:** Explicit execution status (`success`, `failed - no urls`, etc.).

---

### 3. Reverse Image Search (RIS) Module
**Location:** `model/reverse_image_search.py`

The RIS module is a visual forensic agent designed to catch "False Connection" (e.g., an authentic image paired with unrelated text) or "Imposter Content" scenarios.

**Architecture & Flow:**
1. **Image Hosting Setup:** Uploads the user's image to temporary hosts (ImgBB prioritized, with aggressive fallbacks to Freeimage, tmpfiles, envs.sh, etc.) to obtain a public URL.
2. **Visual Query:** Executes a Google Lens search via SerpAPI using the public image URL.
3. **Visual Filter (CLIP):** Downloads thumbnail results and embeds them using `clip-vit-base-patch32`. Computes cosine similarity against the original image. Results below a visual threshold (e.g. 0.9) are discarded to ensure only identical/highly similar images are analyzed.
4. **Semantic Filter (MiniLM):** Uses `all-MiniLM-L6-v2` to compute semantic similarity between the user's original claim and the headlines/snippets of the visually identical image matches.
5. **Visual Agent LLM (Nuanced 4-Way Categorization):** Passes the semantic/visual findings to the Visual Agent LLM (local Ollama `qwen2.5:7b` or `llama3:8b` or OpenRouter). The agent classifies findings into 4 distinct categories:
   - `Authentic Context / Matching Event`: Genuine coverage of the event/subject matter, including syndicated news (Reuters, SCMP, KSL) and photo licensing agencies (Reuters Connect).
   - `Out-of-Context (Misattributed / Recycled)`: Imagery provably recycled from a different historical year, different disaster, or unrelated geographic location.
   - `Generic / Illustrative Imagery`: Standard stock photo or symbolic representation without deceptive intent.
   - `Inconclusive / Insufficient Evidence`: Insufficient metadata to confirm provenance.
6. **Dedicated Forensic Logging (`output/RIS_output.txt`):**
   - Automatically logs the full step-by-step forensic trail to `output/RIS_output.txt`: 
      - **[1]** Google Lens query & candidate matches
      - **[2]** candidate CLIP visual cosine similarities (with pass/fail verdicts against the `>= 0.90` threshold), MiniLM contextual semantic similarities (`>= 0.05`)
      - **[3]** the Visual Agent LLM prompts/responses
      - **[4]** and the final classification with explicit `Out-of-Context` and confidence flags.

---

### 4. Multi-Agent Debate (MAD) Module
**Location:** `model/mad_orchestrator.py`, `model/mad_agent.py`, `model/shared_memory.py`, `model/llm_client.py`

The Multi-Agent Debate (MAD) module is a structured, multi-agent parliamentary debate focusing exclusively on evaluating the factual veracity, reporting authenticity, and geopolitical context of the news claims.

#### Model & Execution Engine:
- **Inference Engine:** Powered locally via **Ollama** using the `LLMClient` (`model/llm_client.py`).
- **Models Used:** Default local LLM is **`qwen2.5:7b`** (and compatible with **`llama3:8b`** as configured in `LOCAL_LLM_MODEL`).
- **Design Rationale:** Running the debate agents locally on Ollama guarantees zero API cost and removes cloud rate-limiting across 7+ sequential LLM generation steps (1 context summary + 6 speaker rounds + 1 synthesis). It also maintains data privacy and avoids cloud content-moderation safety refusals when debating sensitive geopolitical or controversial topics.

#### How It Was Created & Core Architecture:
1. **Dual Context Input & Bias Framing:** 
   - Unlike naive summarization, MAD feeds agents both the **full, original unedited article** and the **condensed verified external knowledge** (`KR`).
   - Rather than re-summarizing the Knowledge Retrieval summary (which would cause degradation and loss of crucial factual detail), the orchestrator preserves the KR external facts verbatim. It directs the local LLM strictly to extract core factual claims from the article and analyze the **political/social climate, potential ideological biases, and narrative agendas**, pairing this analysis directly with the verified external knowledge.
2. **Dynamic Domain Profile Assignment:**
   - The orchestrator analyzes the news topic and assigns 6 specialized expert personas (3 per side) to provide multifaceted domain scrutiny (e.g., *Investigative Journalist*, *Forensic Data Analyst*, *Media Context Analyst* for Real News; *Public Records Specialist*, *Disinformation Researcher*, *Archival Historian* for Fake News).
3. **Structured 3-Stage Parliamentary Debate (3 vs 3):**
   - **Stage 1 (First Speakers):** 
     - *Real News 1st Speaker* opens, followed by *Fake News 1st Speaker*.
     - Role: Define the scope of verification, establish primary constructive arguments, and present initial evidence from their specialized domain.
   - **Stage 2 (Second Speakers):**
     - *Real News 2nd Speaker*, followed by *Fake News 2nd Speaker*.
     - Role: Vigorously attack the opposition using the **"even if" rebuttal tactic** (*"Even if we agreed to your premise/definition (which we don't), your arguments are still wrong because..."*). Summarize their own first speaker in at most 1–2 sentences, and introduce **brand new substantive arguments and evidence** from their distinct domain expertise.
   - **Stage 3 (Third Speakers):**
     - *Real News 3rd Speaker*, followed by *Fake News 3rd Speaker*.
     - Role: Act as "whip" speakers. Deliver targeted rebuttals to remaining claims and crystallize the central clash points by comparatively weighing cumulative evidence against the opposing side's gaps. Strictly prohibited from introducing new constructive arguments.
4. **Anti-Regurgitation Guardrails:**
   - To prevent agents from repeating points or echoing teammates, strict prompting rules prohibit point-cycling. Speakers are constrained to 1–2 sentence summaries and are required to explore distinct evidentiary angles (e.g. data logs vs. journalistic sourcing vs. historical precedents) based on their profile.
5. **Shared Memory Bank (`SharedMemoryBank`):**
   - Central state repository that logs all speeches, stances, and metadata round-by-round, providing the full transcript history to subsequent debaters for contextual continuity.
6. **Consolidated Discourse Synthesis:**
   - An objective academic synthesis pass summarizes the debate, mapping the key evidential weights, clash points, and veracity indicators. The complete discourse transcript is saved to `output/MAD_output.txt` and forwarded to the Judge module.

---

### 5. Judge Module (Reasoning & Judicial Synthesis Layer)
**Location:** `model/judge.py`, `backend/analyzer.py`

The Judge module serves as the final judicial authority within RAHMAD. It consumes the raw article text, the Early Predictor (EP) multimodal signal, the synthesized Knowledge Retrieval (KR) facts, the Reverse Image Search (RIS) visual provenance, and the Multi-Agent Debate (MAD) discourse. 

To support rigorous empirical evaluation and ablation studies, the Judge module implements three configurable architectural modes controlled via the `JUDGE_MODE` environment variable in `.env`:
1. **`single` (Baseline):** Standard Single-Pass Constitutional Chain-of-Thought reasoning.
2. **`self_consistency`:** Stochastic majority-vote aggregation across $N=3$ parallel reasoning paths.
3. **`panel` (Revamped Multi-Judge Architecture):** A concurrent committee of 3 domain specialists that runs in parallel (Fact-Checker, Linguist, Logician) and is synthesized by a Chief Judicial Magistrate.

#### Model & Execution Engine:
- **Inference Provider:** OpenRouter Cloud API.
- **Models Used:** Dynamically loaded via `JUDGE_MODEL` (defaulting to **`deepseek/deepseek-v4-flash-0731`**), backed by a fault-tolerant fallback chain configured in `JUDGE_MODEL_FALLBACK` (`nvidia/nemotron-3-nano-omni-30b`, `inclusionai/ling-3.0-flash`, `poolside/laguna-m.1`).
- **Prompt Frameworks:** Constitutional AI, Step-Back Domain Orientation, Persona Specialization, Chain-of-Thought (CoT), and Judicial Synthesis.

---

#### Architectural Pipelines:

##### Pipeline A: Standard Single-Pass Mode (`JUDGE_MODE=single`)
- **Execution Cost:** 1x API cost | **Latency:** Fastest (single LLM pass).
- **Architecture:** A single generalist LLM processes all multimodal and agentic evidence concurrently through a mandatory 7-step Chain-of-Thought protocol:
  1. **Step 1 — Domain Orientation:** Step-back analysis determining publication domain (Politics, Health, Satire, Finance).
  2. **Step 2 — Claim Extraction:** Isolating 3–5 core verifiable claims.
  3. **Step 3 — Internal Consistency Audit:** Auditing headline-to-body coherence, quote alignment, and numerical data.
  4. **Step 4 — Framing and Style Analysis:** Detecting emotional manipulation, sensationalism, and urgency cues.
  5. **Step 5 — Source and Attribution Analysis:** Scrutinizing source authenticity and imposter markers.
  6. **Step 6 — Multimodal & Multi-Agent Evidence Integration:** Corroborating claims against KR, checking RIS visual provenance (Authentic vs. Recycled/Out-of-Context), weighing MAD debater clash points, and calibrating against EP.
  7. **Step 7 — Self-Critique (Self-Refine):** Counterfactual reasoning challenging initial assumptions before committing to a verdict.

##### Pipeline B: Self-Consistency Majority Vote (`JUDGE_MODE=self_consistency`)
- **Execution Cost:** 3x API cost | **Latency:** Parallelized across $N=3$ threads.
- **Architecture:** Dispatches 3 parallel API queries of the 7-step protocol using a higher sampling temperature (`temperature=0.7`) to promote diverse reasoning paths.
- **Aggregation Protocol:**
  - Performs majority voting across candidate verdicts to filter out stochastic generation anomalies.
  - Averages confidence percentages across runs that agree on the majority verdict.
  - Selects the reasoning trace and justification from the highest-confidence majority run, merging unique key facts.

##### Pipeline C: Multi-Judge Specialized Panel (`JUDGE_MODE=panel`)
- **Execution Cost:** ~4x API cost | **Latency:** 2-stage pipelined execution (3 parallel specialists + 1 sequential chief synthesizer).
- **Architecture:** Rather than placing the entire cognitive load onto a single generalist prompt (which suffers from "lost in the middle" degradation), this architecture divides evaluation across three specialized, independent domain personas running concurrently via `ThreadPoolExecutor(max_workers=3)`:
  1. **The Senior Fact-Checker (`_FACT_CHECKER_PROMPT`):**
     - **Focus:** Factual veracity, external evidence corroboration, and visual/discourse provenance.
     - **Inputs Evaluated:** KR ground-truth news, RIS Google Lens findings (Out-of-Context vs. Authentic), and MAD arguments.
     - **Outputs:** Qualitative claim-by-claim corroboration summary and a **Factual Risk Score (1–10)**.
  2. **The Senior Linguist (`_LINGUIST_PROMPT`):**
     - **Focus:** Stylometrics, emotional manipulation, semantic coherence, and rhetorical framing.
     - **Inputs Evaluated:** Article text, headline-to-body alignment, sensationalism cues, loaded language, and satirical tropes.
     - **Outputs:** Stylistic markers, emotional manipulation rating (Low/Medium/High), and a **Linguistic Risk Score (1–10)**.
  3. **The Senior Logician (`_LOGICIAN_PROMPT`):**
     - **Focus:** Internal contradictions, logical fallacies, source attribution plausibility, and communicative intent.
     - **Inputs Evaluated:** Article structure, statistical consistency, imposter/attribution cues, and intent categorization (Informational, Satirical, Deceptive, Imposter, or Sensationalist).
     - **Outputs:** Logical audit trail and a **Logical Risk Score (1–10)**.

  4. **Chief Judicial Magistrate (`_CHIEF_JUDGE_PROMPT`):**
     - **Role:** Presiding judge and final synthesis arbiter.
     - **Mechanism:** Receives the original text, EP signal, and the structured assessments + risk scores from all 3 specialists.
     - **Judicial Synthesis Principles:**
       - *Factual Dominance:* Strong refutation by the Fact-Checker or recycled imagery from RIS leads to `FABRICATED CONTENT`, `FALSE CONNECTION`, or `MISLEADING CONTENT`.
       - *Satire Resolution:* If the Linguist and Logician identify satirical tropes and humor intent, the article is classified as `SATIRE OR PARODY` even if facts are literal inventions.
       - *Imposter Detection:* If the Logician detects impersonation of official institutions or journalists, classified as `IMPOSTER CONTENT`.
       - *Credibility Threshold:* Low risk scores (1–3) across all three specialists corroborate `CREDIBLE`.

---

#### Output Format & Contract:
Regardless of the active mode, the Judge module guarantees an identical JSON schema to the frontend contract:
```json
{
  "verdict": "SATIRE OR PARODY" | "FALSE CONNECTION" | "MISLEADING CONTENT" | "IMPOSTER CONTENT" | "MANIPULATED CONTENT" | "FABRICATED CONTENT" | "CREDIBLE" | "INCONCLUSIVE",
  "color_class": "satire" | "false-connection" | "misleading" | "imposter" | "manipulated" | "fabricated" | "credible" | "inconclusive",
  "confidence": "92.0%",
  "justification": "2–3 sentence executive explanation with **markdown bolded** evidence and entities.",
  "reasoning": "Full step-by-step or panel judicial deliberation scratchpad.",
  "key_facts": ["Observation 1", "Observation 2", "Observation 3"],
  "links": [{"source": "REUTERS", "title": "...", "url": "..."}],
  "ep_signal": { ... },
}
```

#### Dedicated Forensic Logging (`output/JUDGE_output.txt`):
Automatically logs the end-to-end judicial reasoning session to `output/JUDGE_output.txt`:
- **[1] Configuration & Context Audit:** Active mode (`single`, `panel`, `self_consistency`), active model IDs, EP multimodal metrics, and external evidence ingestion status.
- **[2] Original News Article:** Verbatim text analyzed.
- **[3] Mode-Specific Intermediates:**
  - `panel`: Detailed reports from the Senior Fact-Checker, Senior Linguist, and Senior Logician along with their domain risk scores (1-10) and the Chief Judge synthesis scratchpad.
  - `self_consistency`: Majority vote tallies across the $N=3$ parallel reasoning paths and selected run reasoning.
  - `single`: Full 7-step Constitutional Chain-of-Thought deliberation.
- **[4] Final Judicial Verdict:** The final classified verdict, confidence rating, color class, executive justification, key facts, and verified links.

---

### 6. Backend Orchestrator & Frontend
**Location:** `backend/analyzer.py`, `app.py`, `frontend/script.js`

- **Analyzer (`analyzer.py`):** Acts as the central nervous system. It lazily loads the heavy machine learning models, handles fallback logic (e.g., text-only processing if no image is uploaded), routes data between modules, and coordinates the Pipeline Logger.
- **Pipeline Logger (`pipeline_logger.py`):** Captures intermediate states, LLM prompts, tensor shapes, and tracks the LLM backend utilized, logging them to the terminal and saving them to `output_log.txt`.
- **Frontend App:** A sleek vanilla JS/CSS web app that renders the analysis dynamically, parses Markdown bolding tags into HTML `<strong>`, and beautifully presents the reasoning accordion, confidence score, and verified external news links.
