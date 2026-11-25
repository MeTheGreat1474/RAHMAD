# RAHMAD
Retrieval-Augmented Hybrid Multi-Agent Debate

## Project Structure
```
RAHMAD/
├── app.py                      # Flask entry point — serves the dashboard, exposes POST /api/analyze (NDJSON stream)
├── requirements.txt            # Python dependencies
├── .env / .env.example         # Secrets & runtime configuration (API keys, model selection)
│
├── frontend/                   # Single-page dashboard (served by Flask as static folder)
│   ├── index.html              # Input panel, live process stream, verdict card, evidence links
│   ├── styles.css              # Dashboard styling incl. verdict colour classes
│   └── script.js               # NDJSON stream reader (Fetch API), status & result rendering
│
├── backend/                    # Web-facing orchestration & utilities
│   ├── analyzer.py             # Pipeline orchestrator — analyze_text() yields NDJSON status/result chunks
│   ├── input_processing.py     # Two-branch pre-processing: clean LLM text + CLIP tensors
│   ├── input_retrieval.py      # Loads the default test article from res/ (module self-tests only)
│   ├── domain_whitelist.py     # VERIFIED_DOMAINS set (76 accredited news outlets) + is_domain_whitelisted()
│   ├── news_spider.py          # Scrapy spider extracting title/text/date via trafilatura
│   ├── pipeline_logger.py      # PipelineLogger → output/output_log.txt (structured per-stage audit log)
│   └── analyzer_new.py         # Stale near-duplicate of analyzer.py — imported nowhere, safe to ignore
│
├── model/                      # The six AI modules
│   ├── early_predictor.py      # CLIP ViT-B/32 + LoRA + MLP head inference wrapper
│   ├── knowledge_retrieval.py  # 4-engine news search → whitelist filter → scrape → semantic filter → LLM synthesis
│   ├── reverse_image_search.py # Image hosting bridge → SerpApi Google Lens → CLIP visual filter → Visual Agent LLM
│   ├── mad_orchestrator.py     # Debate orchestrator: context summary, profile assignment, consolidation
│   ├── mad_agent.py            # DebaterAgent: role/side/speaker-structured argument generation
│   ├── shared_memory.py        # SharedMemoryBank: immutable per-stage argument history
│   ├── llm_client.py           # LLMClient: local Ollama client with model fallback chain (used by MAD)
│   └── judge.py                # Judge: OpenRouter reasoning layer, constitution, P8 guardrail, 3 modes
│
├── data_processing/
│   └── early_predictor_output/ # Trained Early Predictor artifacts (required at runtime)
│       ├── lora_adapter/       # PEFT LoRA adapter (r=16, α=32, dropout 0.1)
│       ├── mlp_head.pt         # Linear(1024→1) + Sigmoid classification head
│       └── processor/          # CLIPProcessor (tokenizer + image processor)
│
├── res/                        # Bundled sample articles & images (news1–news3), served at /res/<filename>
├── output/                     # Runtime forensic logs (overwritten each run)
│   ├── output_log.txt          # Full unified pipeline log
│   └── KR/RIS/MAD/JUDGE_output.txt   # Per-module forensic logs
│
├── testing/                    # 3-phase test suite, evaluation runners, datasets & result artefacts
├── docs/                       # Canonical engineering reference — docs/RAHMAD_ARCHITECTURE.md
└── archive/                    # Historical material (EP training scripts, dataset prep, notebooks)
```

## Installation & Setup

**Prerequisites**
- Python 3.10+ with a virtual environment
- [Ollama](https://ollama.com) installed (the Multi-Agent Debate — and with `USE_LOCAL_LLM=true`, the Knowledge Retrieval synthesiser and RIS Visual Agent — run on a local Ollama model)
- API keys: [OpenRouter](https://openrouter.ai/keys) (Judge + cloud fallbacks), [SerpApi](https://serpapi.com/manage-api-key) (Google Lens reverse image search) and [ImgBB](https://api.imgbb.com/) (image hosting for the reverse image search bridge)

**Step 1 — Create a virtual environment and install dependencies:**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Step 2 — Pull the local debate model:**
```powershell
ollama pull qwen2.5:7b
```

**Step 3 — Configure the environment:**
Copy `.env.example` to `.env` and fill in your keys:
```ini
OPENROUTER_API_KEY=<your-openrouter-key>     # Judge Module (cloud reasoning LLM)
SERPAPI_API_KEY=<your-serpapi-key>           # Google Lens reverse image search

USE_LOCAL_LLM=true                           # true  → KR/RIS summarisation on local Ollama
                                             # false → KR/RIS summarisation via OpenRouter
OLLAMA_BASE_URL=http://localhost:11434
LOCAL_LLM_MODEL=qwen2.5:7b
JUDGE_MODEL=deepseek/deepseek-v4.1-flash
JUDGE_MODE=single                            # single | panel | self_consistency
IMGBB_API_KEY=<your-imgbb-key>               # hosts uploaded images for the reverse image search
```
See `.env.example` for the full list of optional variables (judge fallback chain, judge mode, thresholds, timeouts).

> The trained Early Predictor artifacts in `data_processing/early_predictor_output/` are required at runtime. They are included in the repository; if they are missing, the analyzer raises a clear error pointing to the training script in `archive/`.

## Running the App (each time)

**Step 1 — Start Ollama** (required for the Multi-Agent Debate):
Launch the Ollama desktop app, or run:
```powershell
ollama serve
```
Verify it is up: `curl http://localhost:11434/api/tags` should return your model list.

**Step 2 — Start the Flask server:**
```powershell
.\venv\Scripts\activate
python app.py
```

**Step 3 — Open your browser:**
```
http://127.0.0.1:5000
```

On every run, the pipeline first performs a **preflight health check** of the Ollama backend — you will see `CHECKING LLM BACKEND... OK` (or a red `FAIL` warning) at the top of the live process stream. If it fails, start Ollama and run again; the pipeline will still complete, but KR/RIS/MAD will be degraded.

**Step 4 — Run a detection:**
Paste an article's text and (optionally) attach its image, then press **RUN NEWS DETECTION**. The live process stream shows each pipeline stage; the verdict card displays the classification, confidence, justification and supporting links.
