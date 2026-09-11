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
5. **LLM Synthesis:** Surviving snippets are fed into a local LLM (e.g., Ollama `qwen2.5:7b` when `USE_LOCAL_LLM=true`) or a fast OpenRouter LLM (e.g., `nemotron-3-ultra-550b-a55b` as fallback). The LLM is instructed to construct a comprehensive outline of key facts, entities, and timelines, avoiding vague overviews.

---

### 3. Reverse Image Search (RIS) Module
**Location:** `model/reverse_image_search.py`

The RIS module is a visual forensic agent designed to catch "False Connection" (e.g., an authentic image paired with unrelated text) or "Imposter Content" scenarios.

**Architecture & Flow:**
1. **Image Hosting Setup:** Uploads the user's image to temporary hosts (ImgBB prioritized, with aggressive fallbacks to Freeimage, tmpfiles, envs.sh, etc.) to obtain a public URL.
2. **Visual Query:** Executes a Google Lens search via SerpAPI using the public image URL.
3. **Visual Filter (CLIP):** Downloads thumbnail results and embeds them using `clip-vit-base-patch32`. Computes cosine similarity against the original image. Results below a visual threshold (e.g. 0.8) are discarded to ensure only identical/highly similar images are analyzed.
4. **Semantic Filter (MiniLM):** Uses `all-MiniLM-L6-v2` to compute semantic similarity between the user's original claim and the headlines/snippets of the visually identical image matches.
5. **Visual Agent LLM:** Passes the semantic/visual dissonance findings to a visual agent LLM (local Ollama or OpenRouter, depending on `USE_LOCAL_LLM`). The LLM determines if the image is being used Out-Of-Context (OOC), e.g., if the original image originates from a completely different year or event.

---

### 4. Multi-Agent Debate (MAD) Module
**Location:** `model/mad_orchestrator.py`, `model/mad_agent.py`, `model/shared_memory.py`

The MAD module is activated when the Early Predictor is uncertain. It instantiates a structured, parliamentary-style debate focusing strictly on the factual veracity and authenticity of the news claim.

**Architecture & Flow:**
1. **Context Summarization:** The orchestrator retrieves the raw article and synthesizes external knowledge into a focused factual summary.
2. **Profile Generation:** Based on the context, the LLM generates specific expert profiles (e.g., Investigative Journalist, Forensic Data Analyst) to ensure diverse perspectives.
3. **Structured Debate:** A 3-vs-3 debate takes place (Real News vs. Fake News). 
   - **First Speakers:** Introduce the case and present core arguments and evidence.
   - **Second Speakers:** Rebute the previous speaker's points and expand on their own team's case.
   - **Third Speakers:** Provide final rebuttals and synthesize the team's overarching arguments without introducing new claims.
4. **Shared Memory:** A `SharedMemoryBank` tracks the arguments round-by-round, ensuring agents have access to the complete history to form cohesive rebuttals.
5. **Discourse Synthesis:** The orchestrator analyzes the completed debate and generates a consolidated discourse synthesis highlighting the strongest arguments regarding the article's veracity.

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
