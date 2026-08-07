# Local LLM Integration Guide for RAHMAD Knowledge Retrieval & RIS

This guide outlines how to set up and integrate a local LLM (using Ollama) for the **Knowledge Retrieval** and **Reverse Image Search (RIS)** summarizer modules in RAHMAD, replacing cloud API calls (e.g. OpenRouter).

---

## 1. Suitable Local LLM Models

In the RAHMAD pipeline:
- **Knowledge Retrieval** synthesizes web search snippets into factual outlines.
- **Reverse Image Search (RIS)** evaluates context mismatch using visual match metadata.

Because visual feature extraction and image matching are handled upstream by CLIP (`openai/clip-vit-base-patch32`), you only need a fast, open-weights **text LLM** (3B – 8B parameters) with high instruction-following and summarization accuracy.

### Recommended Models

| Model | Ollama Tag | Best For | VRAM Requirement |
| :--- | :--- | :--- | :--- |
| **Qwen 2.5 (7B)** *(Recommended)* | `qwen2.5:7b` | Exceptional structured extraction, fast summarization | ~6 GB VRAM |
| **Qwen 2.5 (3B)** | `qwen2.5:3b` | Lightweight setups, CPU-only or low GPU VRAM | ~3 GB VRAM |
| **Llama 3.1 (8B)** | `llama3.1:8b` | High accuracy, standard instruction following | ~6 GB VRAM |
| **Llama 3.2 (3B)** | `llama3.2:3b` | Fast inference on modest hardware | ~3 GB VRAM |

---

## 2. Setting Up Ollama

Ollama serves local LLMs with an OpenAI-compatible REST API at `http://localhost:11434/v1/chat/completions`.

### Step 1: Install Ollama
Download and run the installer from [ollama.com](https://ollama.com).

### Step 2: Download Your Model
Open PowerShell or Terminal and pull your preferred model:

```powershell
ollama pull qwen2.5:7b
```

### Step 3: Verify Local Endpoint
Ollama automatically runs in the background. Verify the service is active by sending a test request:

```powershell
curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen2.5:7b",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

---

## 3. RAHMAD Code Integration

### Step 1: Update `.env` Configuration
Add the local LLM settings to your `.env` file:

```env
# Local LLM Configuration
USE_LOCAL_LLM=true
LOCAL_LLM_URL=http://localhost:11434/v1/chat/completions
LOCAL_LLM_MODEL=qwen2.5:7b
```

### Step 2: Code Modification Pattern
Update `model/knowledge_retrieval.py` and `model/reverse_image_search.py` to check for `USE_LOCAL_LLM`:

```python
import os

# Check environment for local vs cloud endpoint
USE_LOCAL_LLM = os.environ.get("USE_LOCAL_LLM", "false").lower() == "true"

if USE_LOCAL_LLM:
    _API_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1/chat/completions")
    _MODEL_ID = os.environ.get("LOCAL_LLM_MODEL", "qwen2.5:7b")
    _HEADERS  = {"Content-Type": "application/json"}
else:
    _API_URL = "https://openrouter.ai/api/v1/chat/completions"
    _MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b:free"
    _HEADERS  = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}",
        "Content-Type": "application/json"
    }
```

Since Ollama's `/v1/chat/completions` endpoint uses the standard OpenAI payload format (`{"model": ..., "messages": [...]}`), the existing `requests.post()` calls will work seamlessly without changing prompt structures.
