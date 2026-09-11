"""
model/reverse_image_search.py

Reverse Image Search Pipeline for RAHMAD.
Retrieves visually similar images, applies visual & semantic filters,
and routes to an LLM Visual Agent for decontextualization analysis.
"""

import os
import json
import logging
from typing import List, Dict
import requests
from io import BytesIO
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from serpapi import GoogleSearch

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

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
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rah-mad",
        "X-Title": "RAHMAD Visual Agent"
    }

_CLIP_MODEL_ID  = "openai/clip-vit-base-patch32"

class ReverseImageSearchPipeline:
    def __init__(self, visual_threshold: float = 0.80, semantic_threshold: float = 0.05):
        self.visual_threshold = visual_threshold
        self.semantic_threshold = semantic_threshold
        
        # Will be loaded on demand or initialized later
        self.clip_model = None
        self.clip_processor = None
        self.embedder = None
        self.serpapi_key = os.environ.get("SERPAPI_API_KEY", "")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        
        if not self.serpapi_key:
            log.warning("SERPAPI_API_KEY not found in environment. RIS fetching will fail.")
        if not USE_LOCAL_LLM and not self.openrouter_key:
            log.warning("OPENROUTER_API_KEY not found in environment. Visual Agent will fail.")

    def _init_models(self):
        # Load CLIP
        if self.clip_model is None:
            log.info(f"Loading CLIP model: {_CLIP_MODEL_ID}")
            self.clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_ID)
            self.clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
            
        # Load MiniLM
        if self.embedder is None:
            log.info(f"Loading SentenceTransformer: all-MiniLM-L6-v2")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def execute(self, claim_text: str, image_url: str) -> dict:
        """
        Orchestrates the RIS pipeline.
        """
        print("\n" + "=" * 50)
        print("  REVERSE IMAGE SEARCH (RIS) PIPELINE")
        print("=" * 50)
        log.info(f"Starting Reverse Image Search for claim: '{claim_text}'")
        
        result_info = {
            "status": "success",
            "visual_agent_summary": "No out-of-context imagery detected or RIS failed.",
            "category": "Unknown",
            "matches": [],
            "llm_used": "Local LLM" if USE_LOCAL_LLM else "OpenRouter"
        }
        
        self._init_models()
        
        print("  [Step 1/4] Fetching Reverse Image Search results...")
        raw_results, search_url = self.fetch_results(image_url)
        result_info["search_url"] = search_url
        if not raw_results:
            print("  [Step 1/4] [X] No results found.")
            result_info["status"] = "failed"
            result_info["visual_agent_summary"] = "No reverse image search results found."
            return result_info
            
        print(f"  [Step 2/4] Applying Visual Filter (threshold >= {self.visual_threshold})...")
        visually_filtered = self.filter_visually(image_url, raw_results)
        if not visually_filtered:
            print("  [Step 2/4] [X] No images passed visual threshold.")
            result_info["visual_agent_summary"] = "No visually similar images passed the threshold."
            return result_info
            
        print(f"  [Step 3/4] Applying Semantic Filter (threshold >= {self.semantic_threshold})...")
        semantically_filtered = self.filter_semantically(claim_text, visually_filtered)
        if not semantically_filtered:
            print("  [Step 3/4] [X] No contexts passed semantic threshold.")
            result_info["visual_agent_summary"] = "No contextual text passed the semantic similarity threshold."
            return result_info
            
        print(f"  [Step 4/4] Synthesizing context via {'Local LLM' if USE_LOCAL_LLM else 'OpenRouter'} Visual Agent...")
        summary_data = self.analyze_with_visual_agent(claim_text, semantically_filtered)
        
        result_info["category"] = summary_data.get("category", "Unknown")
        result_info["visual_agent_summary"] = summary_data.get("summary", "")
        result_info["matches"] = semantically_filtered
        
        print("=" * 50)
        print("  REVERSE IMAGE SEARCH COMPLETE")
        print(f"  Category: {result_info['category']}")
        print("=" * 50 + "\n")
        
        return result_info

    def fetch_results(self, image_url: str) -> List[Dict]:
        """
        Uses SerpApi (Google Lens) to fetch visually similar images and context.
        """
    def _upload_for_ris(self, raw_bytes: bytes, file_ext: str) -> str | None:
        """Uploads image to a temporary public host to get a URL for SerpApi."""
        # 1. Try ImgBB (Most Reliable, if user provided API key)
        imgbb_key = os.environ.get("IMGBB_API_KEY")
        if imgbb_key:
            try:
                log.info("Uploading to ImgBB...")
                import base64
                b64_str = base64.b64encode(raw_bytes).decode('utf-8')
                resp = requests.post(f"https://api.imgbb.com/1/upload?key={imgbb_key}", data={"image": b64_str}, timeout=15)
                if resp.status_code == 200:
                    return resp.json()["data"]["url"]
            except Exception as e:
                log.warning(f"ImgBB failed: {e}")
                
        # 2. Try Freeimage.host (Uses their known public API key)
        try:
            log.info("Uploading to freeimage.host...")
            files = {'source': (f'image.{file_ext}', BytesIO(raw_bytes), f'image/{file_ext}')}
            data = {'key': '6d207e02198a847aa98d0a2a901485a5', 'action': 'upload'}
            resp = requests.post('https://freeimage.host/api/1/upload', data=data, files=files, timeout=10)
            if resp.status_code == 200:
                return resp.json()['image']['url']
        except Exception as e:
            log.warning(f"freeimage.host failed: {e}")
            
        # 3. Try tmpfiles.org
        try:
            log.info("Uploading to tmpfiles.org...")
            files = {'file': (f'image.{file_ext}', BytesIO(raw_bytes), f'image/{file_ext}')}
            resp = requests.post('https://tmpfiles.org/api/v1/upload', files=files, timeout=10)
            if resp.status_code == 200:
                url = resp.json()['data']['url']
                return url.replace('tmpfiles.org/', 'tmpfiles.org/dl/')
        except Exception as e:
            log.warning(f"tmpfiles.org failed: {e}")

        # 4. Try envs.sh
        try:
            log.info("Uploading to envs.sh...")
            files = {'file': (f'image.{file_ext}', BytesIO(raw_bytes), f'image/{file_ext}')}
            resp = requests.post('https://envs.sh', files=files, timeout=10)
            if resp.status_code == 200:
                return resp.text.strip()
        except Exception as e:
            log.warning(f"envs.sh failed: {e}")
            
        # 5. Try catbox.moe
        try:
            log.info("Uploading to catbox.moe...")
            files = {'fileToUpload': (f'image.{file_ext}', BytesIO(raw_bytes), f'image/{file_ext}')}
            resp = requests.post('https://catbox.moe/user/api.php', data={'reqtype': 'fileupload'}, files=files, timeout=10)
            if resp.status_code == 200:
                return resp.text.strip()
        except Exception as e:
            log.warning(f"catbox.moe failed: {e}")

        return None

    def fetch_results(self, image_url: str) -> tuple[List[Dict], str]:
        """
        Uses SerpApi (Google Lens) to fetch visually similar images and context.
        Returns the results list and the public URL used for the search.
        """
        if not self.serpapi_key:
            return [], image_url
            
        search_url = image_url
        if image_url.startswith("data:"):
            import base64
            try:
                header, encoded = image_url.split(",", 1)
                file_ext = "jpg"
                if ";" in header and "/" in header:
                    file_ext = header.split(";")[0].split("/")[1]
                
                raw_bytes = base64.b64decode(encoded)
                
                uploaded_url = self._upload_for_ris(raw_bytes, file_ext)
                if uploaded_url:
                    search_url = uploaded_url
                    log.info(f"Image uploaded successfully: {search_url}")
                else:
                    log.error("Failed to upload image to all fallback public hosts.")
                    return [], search_url
            except Exception as e:
                log.error(f"Error uploading image for RIS: {e}")
                return [], search_url
                
        results = []
        try:
            params = {
              "engine": "google_lens",
              "url": search_url,
              "api_key": self.serpapi_key
            }
            
            search = GoogleSearch(params)
            data = search.get_dict()
            
            # Google Lens visual matches
            if "visual_matches" in data:
                for match in data["visual_matches"][:10]: # Top 10 matches
                    results.append({
                        "title": match.get("title", ""),
                        "link": match.get("link", ""),
                        "thumbnail": match.get("thumbnail", ""),
                        "source": match.get("source", "")
                    })
            log.info(f"RIS Fetcher: retrieved {len(results)} matches.")
        except Exception as e:
            log.error(f"Error fetching RIS results: {e}")
            
        return results, search_url
        
    def _download_image(self, url: str) -> Image.Image:
        try:
            if url.startswith("data:"):
                import base64
                raw_bytes = base64.b64decode(url.split(",", 1)[1])
                return Image.open(BytesIO(raw_bytes)).convert("RGB")
                
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            pass
        return None

    def _extract_tensor(self, output) -> torch.Tensor:
        """Helper to extract the tensor from Hugging Face model outputs."""
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, 'image_embeds'):
            return output.image_embeds
        if hasattr(output, 'pooler_output'):
            return output.pooler_output
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def filter_visually(self, original_image_url: str, results: List[Dict]) -> List[Dict]:
        """
        Downloads original and retrieved images, extracts features via CLIP,
        and computes cosine similarity.
        """
        filtered = []
        original_img = self._download_image(original_image_url)
        if original_img is None:
            log.error("Failed to download original image for visual filter.")
            return []
            
        # Get original image embedding
        orig_inputs = self.clip_processor(images=original_img, return_tensors="pt")
        with torch.no_grad():
            orig_out = self.clip_model.get_image_features(**orig_inputs)
            orig_features = self._extract_tensor(orig_out).detach().cpu().numpy()
        
        for res in results:
            thumb_url = res.get("thumbnail")
            if not thumb_url:
                continue
                
            match_img = self._download_image(thumb_url)
            if match_img is None:
                continue
                
            match_inputs = self.clip_processor(images=match_img, return_tensors="pt")
            with torch.no_grad():
                match_out = self.clip_model.get_image_features(**match_inputs)
                match_features = self._extract_tensor(match_out).detach().cpu().numpy()
                
            sim = cosine_similarity(orig_features, match_features)[0][0]
            log.debug(f"Visual similarity for {res.get('link')}: {sim:.4f}")
            
            if sim >= self.visual_threshold:
                res["visual_similarity"] = float(sim)
                filtered.append(res)
                
        log.info(f"Visual filtering: {len(filtered)} out of {len(results)} passed.")
        return filtered
        
    def filter_semantically(self, claim_text: str, visually_filtered: List[Dict]) -> List[Dict]:
        """
        Computes cosine similarity between original claim and retrieved contexts using MiniLM.
        """
        filtered = []
        claim_embedding = self.embedder.encode([claim_text])
        
        for res in visually_filtered:
            context = res.get("title", "")
            if not context:
                continue
                
            context_embedding = self.embedder.encode([context])
            sim = cosine_similarity(claim_embedding, context_embedding)[0][0]
            log.debug(f"Semantic similarity for '{context}': {sim:.4f}")
            
            if sim >= self.semantic_threshold:
                res["semantic_similarity"] = float(sim)
                filtered.append(res)
                
        log.info(f"Semantic filtering: {len(filtered)} out of {len(visually_filtered)} passed.")
        return filtered
        
    def analyze_with_visual_agent(self, claim_text: str, high_fidelity_results: List[Dict]) -> Dict:
        """
        Calls OpenRouter LLM to categorize the context.
        """
        if not USE_LOCAL_LLM and not self.openrouter_key:
            return {"category": "Unknown", "summary": "OpenRouter API key missing."}
            
        context_str = "\n".join([f"- {r.get('title')} (Source: {r.get('source')})" for r in high_fidelity_results])
        
        system_prompt = (
            "You are an expert fact-checker and visual analyst. Your task is to analyze the original textual claim "
            "and compare it against the verified contextual data of visually identical images found online. "
            "Determine if the image is being used in a misleading or decontextualized manner. "
            "Categorize your finding STRICTLY as either 'Identical Source' (the claim matches the original context) "
            "or 'Identical Image, Different Context' (the image is being used out-of-context). "
            "Output your response in JSON format with two keys: 'category' and 'summary'."
        )
        
        user_prompt = (
            f"Original Claim: {claim_text}\n\n"
            f"Context of identical images found online:\n{context_str}\n\n"
            "Respond ONLY in valid JSON format: {\"category\": \"...\", \"summary\": \"...\"}"
        )
        
        headers = _HEADERS
        
        if USE_LOCAL_LLM:
            models_to_try = [_MODEL_ID]
        else:
            models_to_try = [
                _MODEL_ID,
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                "inclusionai/ling-3.0-flash:free",
                "poolside/laguna-m.1:free",
            ]
        
        last_err = None
        for model in models_to_try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            try:
                log.info(f"Visual Agent: trying model {model} (via {'Local' if USE_LOCAL_LLM else 'OpenRouter'})...")
                resp = requests.post(_API_URL, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                    
                    # Try to parse JSON from response
                    try:
                        parsed = json.loads(content)
                        return parsed
                    except json.JSONDecodeError:
                        return {"category": "Unknown", "summary": content.strip()}
                else:
                    err_msg = data.get("error", {}).get("message", str(data))
                    log.warning(f"Visual Agent model {model} missing choices: {err_msg}")
                    last_err = f"API returned no choices. Error: {err_msg}"
                    
            except Exception as e:
                log.warning(f"Visual Agent model {model} failed: {e}")
                last_err = str(e)
                
        log.error(f"All Visual Agent models failed. Last error: {last_err}")
        return {"category": "Unknown", "summary": f"Visual Agent Error: {last_err}"}
