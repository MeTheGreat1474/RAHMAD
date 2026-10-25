import os
import json
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with local LLMs via Ollama.
    Supports primary model configuration and automatic fallback.
    """
    def __init__(self, model_name=None, base_url=None):
        fallback_raw = os.environ.get("LOCAL_LLM_MODEL_FALLBACK", "llama3:8b")
        self.fallback_models = [m.strip() for m in fallback_raw.split(",") if m.strip()]
        default_fallback = self.fallback_models[0] if self.fallback_models else "llama3:8b"
        
        self.model_name = model_name or os.environ.get("LOCAL_LLM_MODEL", default_fallback)
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    def generate(self, prompt, system_prompt=None):
        url = f"{self.base_url}/api/generate"
        
        models_to_try = [self.model_name]
        for m in self.fallback_models:
            if m not in models_to_try:
                models_to_try.append(m)

        last_error = None
        for model in models_to_try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            if system_prompt:
                payload["system"] = system_prompt

            try:
                logger.debug(f"Calling Ollama with model {model}")
                response = requests.post(url, json=payload, timeout=180)  # Local models can take time
                response.raise_for_status()
                data = response.json()
                return data.get("response", "").strip()
            except requests.exceptions.HTTPError as e:
                error_msg = f"Error calling Ollama API ({model}): {e} - Response: {e.response.text}"
                logger.warning(error_msg)
                last_error = error_msg
            except requests.exceptions.RequestException as e:
                error_msg = f"Error connecting to Ollama ({model}): {e}"
                logger.warning(error_msg)
                last_error = error_msg

        return f"[{last_error or 'Error connecting to Ollama'}]"
