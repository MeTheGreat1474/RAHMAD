import requests
import json
import logging

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with local LLMs via Ollama.
    """
    def __init__(self, model_name="llama3:8b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt, system_prompt=None):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            logger.debug(f"Calling Ollama with model {self.model_name}")
            response = requests.post(url, json=payload, timeout=120)  # Local models can take time
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.HTTPError as e:
            error_msg = f"Error calling Ollama API: {e} - Response: {e.response.text}"
            logger.error(error_msg)
            return f"[{error_msg}]"
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"[Error connecting to Ollama: {e}]"
