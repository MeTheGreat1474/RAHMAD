# RAHMAD: Retrieval-Augmented Hybrid Multi-Agent Debate
## System Architecture Documentation

The RAHMAD framework is a modular, AI-driven pipeline designed for comprehensive fake news and misinformation detection. It combines lightweight multimodal early prediction with heavyweight Retrieval-Augmented Generation (RAG), external visual search, and a constitutional LLM reasoning layer to arrive at a final verdict.

Below is the detailed architectural breakdown of each core module in the system.

---

### 1. Early Predictor Module
**Location:** `model/early_predictor.py`

The Early Predictor (EP) serves as a fast, lightweight first pass. It attempts to classify the article using a fine-tuned multimodal model. If the model is uncertain, the article is escalated to the Multi-Agent Debate pipeline.

**Architecture & Flow:**
1. **Input Preprocessing:** Text is cleaned (HTML tags, URLs, Emojis removed) to match the training distribution. Images are processed via `CLIPProcessor`.
2. **Feature Extraction:** Utilizes `openai/clip-vit-base-patch32` for both text and vision encoding. 
3. **LoRA Adapters:** Parameter-Efficient Fine-Tuning (PEFT) LoRA adapters are applied to the CLIP attention layers.
4. **Fusion:** The text embeddings `[512]` and image embeddings `[512]` are concatenated into a single fused tensor `[1024]`.
5. **Classification (MLP Head):** The fused tensor is passed through a single-layer Multi-Layer Perceptron (Linear `1024 -> 1` + Sigmoid activation) to output a scalar probability between `0` (Real) and `1` (Fake).
6. **Thresholding:**
   - **p >= 0.60:** High-confidence Fake.
   - **p <= 0.40:** High-confidence Real.
   - **0.40 < p < 0.60:** Uncertain. The `pass_to_agents` flag is set to `True`, triggering the full agent pipeline.

---

### 2. Knowledge Retrieval Module
**Location:** `model/knowledge_retrieval.py`

The Knowledge Retrieval (KR) module is responsible for fetching, filtering, and synthesizing external, verified news to provide corroborating evidence (or debunking context) to the Judge module.

**Architecture & Flow:**
1. **Search & Retrieval:** Queries DuckDuckGo HTML (with DDGS fallback) to retrieve raw news URLs related to the claim.
2. **Domain Whitelisting:** URLs are strictly filtered against a predefined list of trusted news outlets (`backend/domain_whitelist.py`) to prevent citing unreliable sources.
3. **Extraction & Heuristics:** Uses `trafilatura` (fallback to `BeautifulSoup`) to scrape article metadata. It filters out non-English content, excessively short articles, and outdated content (>3 years old).
4. **Semantic Verification:** Uses `sentence-transformers/all-MiniLM-L6-v2` to embed the claim and the article chunks. Cosine similarity is calculated to drop irrelevant snippets (similarity < threshold).
5. **LLM Synthesis:** Surviving snippets are fed into a local LLM (e.g., Ollama `llama3:8b` when `USE_LOCAL_LLM=true`) or a fast OpenRouter LLM (e.g., `nemotron-3-ultra-550b-a55b` as fallback). The LLM is instructed to construct a comprehensive outline of key facts, entities, and timelines, avoiding vague overviews.

---

### 3. Reverse Image Search (RIS) Module
**Location:** `model/reverse_image_search.py`

The RIS module is a visual forensic agent designed to catch "False Connection" (e.g., an authentic image paired with unrelated text) or "Imposter Content" scenarios.

**Architecture & Flow:**
1. **Image Hosting Setup:** Uploads the user's image to temporary hosts (ImgBB prioritized, with aggressive fallbacks to Freeimage, tmpfiles, envs.sh, etc.) to obtain a public URL.
2. **Visual Query:** Executes a Google Lens search via SerpAPI using the public image URL.
3. **Visual Filter (CLIP):** Downloads thumbnail results and embeds them using `clip-vit-base-patch32`. Computes cosine similarity against the original image. Results below a visual threshold (e.g. 0.8) are discarded to ensure only identical/highly similar images are analyzed.
4. **Semantic Filter (MiniLM):** Uses `all-MiniLM-L6-v2` to compute semantic similarity between the user's original claim and the headlines/snippets of the visually identical image matches.
5. **Visual Agent LLM (Nuanced 4-Way Categorization):** Passes the semantic/visual findings to the Visual Agent LLM (local Ollama `llama3:8b` or OpenRouter). Instead of forcing a fragile binary choice that misflags syndicated news or wire catalog titles, the agent classifies findings into 4 distinct categories:
   - `Authentic Context / Matching Event`: Genuine coverage of the event/subject matter, including syndicated news (Reuters, SCMP, KSL) and photo licensing agencies (Reuters Connect).
   - `Out-of-Context (Misattributed / Recycled)`: Imagery provably recycled from a different historical year, different disaster, or unrelated geographic location.
   - `Generic / Illustrative Imagery`: Standard stock photo or symbolic representation without deceptive intent.
   - `Inconclusive / Insufficient Evidence`: Insufficient metadata to confirm provenance.
6. **Dedicated Forensic Logging (`output/RIS_output.txt`):**
   - Automatically logs the full step-by-step forensic trail to `output/RIS_output.txt`: Google Lens query & candidate matches, candidate CLIP visual cosine similarities (with pass/fail verdicts against the `>= 0.80` threshold), MiniLM contextual semantic similarities (`>= 0.05`), the Visual Agent LLM prompts/responses, and the final classification with explicit `Out-of-Context` and confidence flags.

---

### 4. Multi-Agent Debate (MAD) Module
**Location:** `model/mad_orchestrator.py`, `model/mad_agent.py`, `model/shared_memory.py`, `model/llm_client.py`

The Multi-Agent Debate (MAD) module is a structured, multi-agent parliamentary debate focusing exclusively on evaluating the factual veracity, reporting authenticity, and geopolitical context of the news claims.

#### Model & Execution Engine:
- **Inference Engine:** Powered locally via **Ollama** using the `LLMClient` (`model/llm_client.py`).
- **Models Used:** Default local LLM is **`llama3:8b`** (and compatible with **`qwen2.5:7b`** as configured in `LOCAL_LLM_MODEL`).
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

### 5. Judge Module (Reasoning Layer)
**Location:** `model/judge.py`

The Judge module is the final arbiter. It consumes the raw article text, the Early Predictor signal, the synthesized Knowledge Retrieval facts, and the Reverse Image Search forensic analysis.

**Architecture & Flow:**
1. **Context Aggregation:** Compiles all inputs into a massive system prompt context.
2. **7-Step Reasoning Protocol:** The cloud LLM executes a mandatory Chain-of-Thought protocol:
   - **Step 1 - Domain Orientation:** Step-back analysis to determine context (Politics, Health, Satire).
   - **Step 2 - Claim Extraction:** Identifying the core falsifiable claims.
   - **Step 3 - Source & Context Verification:** Cross-referencing claims against KR and RIS data.
   - **Step 4 - Linguistic & Visual Tone Analysis:** Identifying hyperbole, emotionally manipulative language, or satirical cues.
   - **Step 5 - Deception & Logical Fallacy Check:** Looking for misleading phrasing.
   - **Step 6 - Classification Alignment:** Mapping the evidence to one of the 8 predefined taxonomy categories.
   - **Step 7 - Self-Critique:** A self-refine step challenging the current assumption (e.g., "Am I misinterpreting satire as fabricated news?").
3. **JSON Output Formulation:** The LLM produces a strict JSON structure containing the `verdict`, `color_class`, `confidence`, a markdown-bolded `justification`, `key_facts`, and the raw `reasoning` trace.

---

### 6. Backend Orchestrator & Frontend
**Location:** `backend/analyzer.py`, `app.py`, `frontend/script.js`

- **Analyzer (`analyzer.py`):** Acts as the central nervous system. It lazily loads the heavy machine learning models, handles fallback logic (e.g., text-only processing if no image is uploaded), routes data between modules, and coordinates the Pipeline Logger.
- **Pipeline Logger (`pipeline_logger.py`):** Captures intermediate states, LLM prompts, tensor shapes, and tracks the LLM backend utilized, logging them to the terminal and saving them to `output_log.txt`.
- **Frontend App:** A sleek vanilla JS/CSS web app that renders the analysis dynamically, parses Markdown bolding tags into HTML `<strong>`, and beautifully presents the reasoning accordion, confidence score, and verified external news links.
