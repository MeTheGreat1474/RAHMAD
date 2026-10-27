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
from pathlib import Path
from datetime import datetime
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

_FALLBACK_MODEL_RAW = os.environ.get("LOCAL_LLM_MODEL_FALLBACK", "llama3:8b")
_LOCAL_FALLBACKS = [m.strip() for m in _FALLBACK_MODEL_RAW.split(",") if m.strip()]
_FALLBACK_MODEL = _LOCAL_FALLBACKS[0] if _LOCAL_FALLBACKS else "llama3:8b"

_FALLBACK_OPENROUTER_RAW = os.environ.get("OPENROUTER_MODEL_FALLBACK", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
_OPENROUTER_FALLBACKS = [m.strip() for m in _FALLBACK_OPENROUTER_RAW.split(",") if m.strip()]
_FALLBACK_OPENROUTER = _OPENROUTER_FALLBACKS[0] if _OPENROUTER_FALLBACKS else "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

if USE_LOCAL_LLM:
    _API_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1/chat/completions")
    _MODEL_ID = os.environ.get("LOCAL_LLM_MODEL", _FALLBACK_MODEL)
    _HEADERS  = {"Content-Type": "application/json"}
else:
    _API_URL = "https://openrouter.ai/api/v1/chat/completions"
    _MODEL_ID = os.environ.get("OPENROUTER_MODEL", _FALLBACK_OPENROUTER)
    _HEADERS  = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rah-mad",
        "X-Title": "RAHMAD Visual Agent"
    }

_CLIP_MODEL_ID  = "openai/clip-vit-base-patch32"

class ReverseImageSearchPipeline:
    def __init__(self, visual_threshold: float = 0.68, semantic_threshold: float = 0.0):
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
        Orchestrates the RIS pipeline and logs every stage to output/RIS_output.txt.
        """
        print("\n" + "=" * 50)
        print("  REVERSE IMAGE SEARCH (RIS) PIPELINE")
        print("=" * 50)
        log.info(f"Starting Reverse Image Search for claim: '{claim_text}'")

        # Describe image format for logging
        if isinstance(image_url, str) and image_url.startswith("data:"):
            header = image_url.split(",", 1)[0]
            img_desc = f"Base64 Inline Image ({header}, length: {len(image_url):,} chars)"
        elif isinstance(image_url, (str, Path)) and os.path.isfile(str(image_url)):
            img_desc = f"Local Image File ({image_url})"
        else:
            img_desc = str(image_url)

        backend_desc = f"Local LLM ({_MODEL_ID})" if USE_LOCAL_LLM else f"OpenRouter ({_MODEL_ID})"

        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
            "claim_text": claim_text,
            "image_type_desc": img_desc,
            "search_url": "Pending retrieval...",
            "backend_llm": backend_desc,
            "raw_matches": [],
            "serpapi_status": "Not started",
            "visual_filter": {"threshold": self.visual_threshold, "evaluated": [], "passed_count": 0, "status": "Pending"},
            "semantic_filter": {"threshold": self.semantic_threshold, "evaluated": [], "passed_count": 0, "status": "Pending"},
            "visual_agent": {"model": _MODEL_ID, "category": "Unknown", "summary": "", "context_sent": ""},
            "final_result": {"status": "in_progress", "category": "Unknown", "matches_count": 0}
        }
        
        result_info = {
            "status": "success",
            "visual_agent_summary": "No out-of-context imagery detected or RIS failed.",
            "category": "Unknown",
            "matches": [],
            "llm_used": "Local LLM" if USE_LOCAL_LLM else "OpenRouter"
        }
        
        try:
            self._init_models()
            
            print("  [Step 1/4] Fetching Reverse Image Search results...")
            raw_results, search_url = self.fetch_results(image_url)
            result_info["search_url"] = search_url
            log_data["search_url"] = search_url
            log_data["raw_matches"] = raw_results
            log_data["serpapi_status"] = f"Retrieved {len(raw_results)} candidate matches via Google Lens." if raw_results else "No matches returned by Google Lens."
            
            if not raw_results:
                print("  [Step 1/4] [X] No results found.")
                result_info["status"] = "failed"
                result_info["visual_agent_summary"] = "No reverse image search results found."
                log_data["final_result"] = {"status": "failed", "category": "Unknown", "matches_count": 0}
                self.save_ris_log(log_data)
                return result_info
                
            print(f"  [Step 2/4] Applying Visual Filter (threshold >= {self.visual_threshold})...")
            visually_filtered = self.filter_visually(image_url, raw_results)
            log_data["visual_filter"] = {
                "threshold": self.visual_threshold,
                "evaluated": raw_results,
                "passed_count": len(visually_filtered),
                "status": f"{len(visually_filtered)} of {len(raw_results)} candidates passed visual threshold."
            }
            
            if not visually_filtered:
                print("  [Step 2/4] [X] No images passed visual threshold.")
                result_info["visual_agent_summary"] = "No visually similar images passed the threshold."
                log_data["final_result"] = {"status": "no_visual_matches", "category": "No Visual Match", "matches_count": 0}
                self.save_ris_log(log_data)
                return result_info
                
            print(f"  [Step 3/4] Applying Semantic Filter (threshold >= {self.semantic_threshold})...")
            semantically_filtered = self.filter_semantically(claim_text, visually_filtered)
            log_data["semantic_filter"] = {
                "threshold": self.semantic_threshold,
                "evaluated": visually_filtered,
                "passed_count": len(semantically_filtered),
                "status": f"{len(semantically_filtered)} of {len(visually_filtered)} candidates passed semantic threshold."
            }
            
            # If some candidates passed semantic threshold, prioritize them;
            # otherwise, if visual matches exist, pass them directly to Visual Agent
            # so the Visual Agent can diagnose potential out-of-context misattribution.
            candidate_contexts = semantically_filtered if semantically_filtered else visually_filtered
            
            if not candidate_contexts:
                print("  [Step 3/4] [X] No contexts passed visual or semantic filters.")
                result_info["visual_agent_summary"] = "No contextual text available for visual verification."
                log_data["final_result"] = {"status": "no_contexts", "category": "No Contextual Match", "matches_count": 0}
                self.save_ris_log(log_data)
                return result_info
                
            print(f"  [Step 4/4] Synthesizing context via {'Local LLM' if USE_LOCAL_LLM else 'OpenRouter'} Visual Agent...")
            summary_data = self.analyze_with_visual_agent(claim_text, candidate_contexts)
            log_data["visual_agent"] = summary_data
            
            result_info["category"] = summary_data.get("category", "Inconclusive / Insufficient Evidence")
            result_info["is_out_of_context"] = summary_data.get("is_out_of_context", False)
            result_info["confidence"] = summary_data.get("confidence", "Medium")
            result_info["visual_agent_summary"] = summary_data.get("summary", "")
            result_info["judge_summary"] = summary_data.get("judge_summary", "")
            result_info["matches"] = semantically_filtered
            
            log_data["judge_summary"] = result_info["judge_summary"]
            log_data["final_result"] = {
                "status": "success",
                "category": result_info["category"],
                "is_out_of_context": result_info["is_out_of_context"],
                "confidence": result_info["confidence"],
                "matches_count": len(semantically_filtered)
            }
            
            print("=" * 50)
            print("  REVERSE IMAGE SEARCH COMPLETE")
            print(f"  Category: {result_info['category']}")
            print(f"  Decontextualized (OOC): {'YES' if result_info['is_out_of_context'] else 'NO'}")
            print(f"  Confidence: {result_info['confidence']}")
            print("=" * 50 + "\n")
            
            self.save_ris_log(log_data)
            return result_info

        except Exception as exc:
            log.error(f"Unhandled error in RIS execute: {exc}", exc_info=True)
            result_info["status"] = "failed"
            result_info["visual_agent_summary"] = f"RIS encountered an error: {exc}"
            log_data["final_result"] = {"status": "error", "category": "Error", "matches_count": 0}
            log_data["error_message"] = str(exc)
            try:
                self.save_ris_log(log_data)
            except Exception as e:
                log.error(f"Failed to write error log in RIS: {e}")
            return result_info
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
            
        search_url = str(image_url) if image_url is not None else ""
        if isinstance(image_url, str) and image_url.startswith("data:"):
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
        elif isinstance(image_url, (str, Path)) and os.path.isfile(str(image_url)):
            try:
                img_path = Path(image_url)
                raw_bytes = img_path.read_bytes()
                file_ext = img_path.suffix.lstrip(".").lower() or "jpg"
                uploaded_url = self._upload_for_ris(raw_bytes, file_ext)
                if uploaded_url:
                    search_url = uploaded_url
                    log.info(f"Local image uploaded successfully: {search_url}")
                else:
                    log.error("Failed to upload local image to all fallback public hosts.")
                    return [], search_url
            except Exception as e:
                log.error(f"Error uploading local image for RIS: {e}")
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
            if isinstance(url, Image.Image):
                return url.convert("RGB")
            if isinstance(url, (str, Path)) and os.path.isfile(str(url)):
                return Image.open(str(url)).convert("RGB")
            if isinstance(url, str) and url.startswith("data:"):
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
            for res in results:
                res["visual_similarity"] = None
                res["visual_passed"] = False
                res["visual_note"] = "Failed to download original target image"
            return []
            
        # Get original image embedding
        orig_inputs = self.clip_processor(images=original_img, return_tensors="pt")
        with torch.no_grad():
            orig_out = self.clip_model.get_image_features(**orig_inputs)
            orig_features = self._extract_tensor(orig_out).detach().cpu().numpy()
        
        for res in results:
            thumb_url = res.get("thumbnail")
            if not thumb_url:
                res["visual_similarity"] = None
                res["visual_passed"] = False
                res["visual_note"] = "No thumbnail URL provided"
                continue
                
            match_img = self._download_image(thumb_url)
            if match_img is None:
                res["visual_similarity"] = None
                res["visual_passed"] = False
                res["visual_note"] = "Thumbnail download failed"
                continue
                
            match_inputs = self.clip_processor(images=match_img, return_tensors="pt")
            with torch.no_grad():
                match_out = self.clip_model.get_image_features(**match_inputs)
                match_features = self._extract_tensor(match_out).detach().cpu().numpy()
                
            sim = float(cosine_similarity(orig_features, match_features)[0][0])
            res["visual_similarity"] = sim
            log.debug(f"Visual similarity for {res.get('link')}: {sim:.4f}")
            
            if sim >= self.visual_threshold:
                res["visual_passed"] = True
                filtered.append(res)
            else:
                res["visual_passed"] = False
                
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
                res["semantic_similarity"] = None
                res["semantic_passed"] = False
                res["semantic_note"] = "No title context available"
                continue
                
            context_embedding = self.embedder.encode([context])
            sim = float(cosine_similarity(claim_embedding, context_embedding)[0][0])
            res["semantic_similarity"] = sim
            log.debug(f"Semantic similarity for '{context}': {sim:.4f}")
            
            if sim >= self.semantic_threshold:
                res["semantic_passed"] = True
                filtered.append(res)
            else:
                res["semantic_passed"] = False
                
        log.info(f"Semantic filtering: {len(filtered)} out of {len(visually_filtered)} passed.")
        return filtered
        
    def analyze_with_visual_agent(self, claim_text: str, high_fidelity_results: List[Dict]) -> Dict:
        """
        Calls OpenRouter LLM or Local LLM to categorize the context with nuanced 4-category classification.
        """
        if not USE_LOCAL_LLM and not self.openrouter_key:
            return {
                "category": "Inconclusive / Insufficient Evidence",
                "is_out_of_context": False,
                "confidence": "Low",
                "summary": "OpenRouter API key missing.",
                "model": "None (Missing API Key)",
                "context_sent": "",
                "raw_response": ""
            }
            
        context_str = "\n".join([f"- {r.get('title')} (Source: {r.get('source')})" for r in high_fidelity_results])
        
        system_prompt = (
            "You are an expert fact-checker and visual forensic analyst specializing in reverse image search and media authenticity verification.\n"
            "Your task is to determine whether an image found online is being used in an AUTHENTIC context or a DECONTEXTUALIZED (Out-of-Context / Misattributed) manner.\n\n"
            "CRITICAL JOURNALISTIC & FACT-CHECKING RULES:\n"
            "1. REAL-WORLD PHOTOJOURNALISM STANDARDS:\n"
            "   - News articles frequently pair broad event headlines (e.g. 'Wildfire emissions surge in Indonesia') with authentic on-the-ground photos of that event (e.g. 'Firefighters douse flames in Sumatra', 'Peatland smoke over Borneo'). This is AUTHENTIC editorial coverage of the event, NOT out-of-context deception!\n"
            "   - Syndication & Rephrasing: Outlets (Reuters, SCMP, CNN, AP, local news) routinely rephrase headlines or license the same wire photo from agencies (Reuters Connect, Getty Images, AFP). This represents the SAME authentic event.\n"
            "   - Do NOT be pedantic about phrasing differences. If the image depicts the same incident, disaster, subject matter, or news event described in the claim, it is MATCHING.\n\n"
            "2. TRUE OUT-OF-CONTEXT (DECONTEXTUALIZED) CRITERIA:\n"
            "   - Mark as 'Out-of-Context (Misattributed / Recycled)' ONLY if the image is demonstrably recycled from a COMPLETELY DIFFERENT event, a DIFFERENT historical year/time period (e.g. using a 2015 or 2017 fire photo to represent a 2026 event), or an entirely DIFFERENT geographic location/incident (e.g. showing California fires and claiming it is Indonesia).\n\n"
            "3. WEIGHING EVIDENCE (SOURCE COUNTING):\n"
            "   - Explicitly count and evaluate the number of retrieved contexts that support the image being Authentic vs. those that suggest it is Out-Of-Context.\n"
            "   - If the majority of sources align with the claim's event, heavily weight this toward 'Authentic Context', even if a minority of sources show it used elsewhere (which could be due to other outlets misusing it or loose search filtering).\n\n"
            "4. CATEGORIES (Choose EXACTLY ONE):\n"
            "   - 'Authentic Context / Matching Event': The image depicts the actual event, disaster, location, or subject matter discussed in the claim, including syndicated reporting, photo agency catalog listings, and on-the-ground photojournalism of that event.\n"
            "   - 'Out-of-Context (Misattributed / Recycled)': The image is recycled from a different historical time, different event, or unrelated location to fabricate or distort the current claim.\n"
            "   - 'Generic / Illustrative Imagery': The image is a generic stock photo, symbolic graphic, or general illustration (e.g. generic forest, flame silhouette) with no deceptive intent about a specific historical event.\n"
            "   - 'Inconclusive / Insufficient Evidence': The retrieved image matches provide insufficient contextual details to definitively determine provenance.\n\n"
            "Output your response strictly in valid JSON format with keys:\n"
            "{\n"
            "  \"category\": \"<One of the 4 exact category names above>\",\n"
            "  \"is_out_of_context\": <true or false>,\n"
            "  \"confidence\": \"<High / Medium / Low>\",\n"
            "  \"summary\": \"<Detailed, objective forensic explanation of why the imagery matches or does not match the claim's context. You MUST explicitly state the count of sources supporting Authentic vs Out-Of-Context.>\",\n"
            "  \"judge_summary\": \"<Concise 2-4 sentence executive evidence summary formulated specifically for the Judge module, outlining: (1) what the image depicts and its verified origin, (2) whether it matches the claim's event/time, and (3) explicit judicial recommendation on visual provenance. Note: Explicitly state that an authentic image verifies media integrity but does not independently prove unverified claims or speculative character attributions in the text.>\"\n"
            "}"
        )
        
        # Extract headline and concise excerpt to prevent overwhelming the LLM with thousands of chars of body text
        lines = [line.strip() for line in claim_text.strip().splitlines() if line.strip()]
        headline = lines[0] if lines else claim_text[:150]
        body_snippet = "\n".join(lines[1:]) if len(lines) > 1 else ""
        if len(body_snippet) > 800:
            body_snippet = body_snippet[:800] + "..."
            
        user_prompt = f"Headline / Primary Claim:\n{headline}\n\n"
        if body_snippet:
            user_prompt += f"Article Body Excerpt:\n{body_snippet}\n\n"
        user_prompt += (
            f"Context of identical images found online via Google Lens:\n{context_str}\n\n"
            "Determine whether the online matches corroborate that the image depicts this authentic news event, or whether it is recycled from an unrelated event/time.\n"
            "Respond ONLY in valid JSON format matching the requested schema."
        )
        
        headers = _HEADERS
        
        if USE_LOCAL_LLM:
            models_to_try = [_MODEL_ID]
            for m in _LOCAL_FALLBACKS:
                if m not in models_to_try:
                    models_to_try.append(m)
        else:
            models_to_try = [_MODEL_ID]
            for m in _OPENROUTER_FALLBACKS:
                if m not in models_to_try:
                    models_to_try.append(m)
        
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
                resp = requests.post(_API_URL, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                    
                    # Try to parse JSON from response
                    try:
                        parsed = json.loads(content)
                        raw_cat = parsed.get("category", "")
                        cat_lower = str(raw_cat).lower()
                        
                        if "authentic" in cat_lower or "matching" in cat_lower or "identical source" in cat_lower:
                            category = "Authentic Context / Matching Event"
                            is_ooc = False
                        elif "out-of-context" in cat_lower or "recycled" in cat_lower or "misattributed" in cat_lower or "different context" in cat_lower:
                            category = "Out-of-Context (Misattributed / Recycled)"
                            is_ooc = True
                        elif "generic" in cat_lower or "illustrative" in cat_lower:
                            category = "Generic / Illustrative Imagery"
                            is_ooc = False
                        else:
                            category = "Inconclusive / Insufficient Evidence"
                            is_ooc = False
                            
                        if "is_out_of_context" in parsed and isinstance(parsed["is_out_of_context"], bool):
                            is_ooc = parsed["is_out_of_context"]
                            
                        confidence = parsed.get("confidence", "High" if "authentic" in cat_lower or is_ooc else "Medium")
                        summary = parsed.get("summary", content.strip())
                        
                        judge_summary = parsed.get("judge_summary")
                        if not judge_summary:
                            ooc_label = "Decontextualized / Misattributed" if is_ooc else "Authentic / Contextually Aligned"
                            judge_summary = (
                                f"Visual Assessment: {category} ({ooc_label}). Confidence: {confidence}.\n"
                                f"{summary}"
                            )
                        
                        return {
                            "category": category,
                            "is_out_of_context": is_ooc,
                            "confidence": confidence,
                            "summary": summary,
                            "judge_summary": judge_summary,
                            "model": model,
                            "context_sent": context_str,
                            "raw_response": content
                        }
                    except json.JSONDecodeError:
                        cat_lower = content.lower()
                        if "authentic" in cat_lower or "matching" in cat_lower or "identical source" in cat_lower:
                            category = "Authentic Context / Matching Event"
                            is_ooc = False
                        elif "out-of-context" in cat_lower or "recycled" in cat_lower or "misattributed" in cat_lower or "different context" in cat_lower:
                            category = "Out-of-Context (Misattributed / Recycled)"
                            is_ooc = True
                        elif "generic" in cat_lower or "illustrative" in cat_lower:
                            category = "Generic / Illustrative Imagery"
                            is_ooc = False
                        else:
                            category = "Inconclusive / Insufficient Evidence"
                            is_ooc = False
                            
                        ooc_label = "Decontextualized / Misattributed" if is_ooc else "Authentic / Contextually Aligned"
                        judge_summary = f"Visual Assessment: {category} ({ooc_label}). Confidence: Medium.\n{content.strip()}"
                        return {
                            "category": category,
                            "is_out_of_context": is_ooc,
                            "confidence": "Medium",
                            "summary": content.strip(),
                            "judge_summary": judge_summary,
                            "model": model,
                            "context_sent": context_str,
                            "raw_response": content
                        }
                else:
                    err_msg = data.get("error", {}).get("message", str(data))
                    log.warning(f"Visual Agent model {model} missing choices: {err_msg}")
                    last_err = f"API returned no choices. Error: {err_msg}"
                    
            except Exception as e:
                log.warning(f"Visual Agent model {model} failed: {e}")
                last_err = str(e)
                
        log.error(f"All Visual Agent models failed. Last error: {last_err}")
        return {
            "category": "Inconclusive / Insufficient Evidence",
            "is_out_of_context": False,
            "confidence": "Low",
            "summary": f"Visual Agent Error: {last_err}",
            "judge_summary": f"Reverse Image Search analysis unavailable due to an error: {last_err}. Evaluation will proceed using textual facts.",
            "model": "None (Failed)",
            "context_sent": context_str,
            "raw_response": str(last_err)
        }

    def save_ris_log(self, log_data: dict, output_path=None) -> Path:
        """
        Log all Reverse Image Search forensic stages, Google Lens matches,
        CLIP visual scores, MiniLM semantic scores, and Visual Agent analysis
        into RIS_output.txt inside the output/ folder.
        """
        if output_path is None:
            project_root = Path(__file__).resolve().parent.parent
            output_dir = project_root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "RIS_output.txt"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        now_str = log_data.get("timestamp") or datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        lines = []
        sep_major = "=" * 80
        sep_minor = "-" * 60

        lines.append(sep_major)
        lines.append(f"  REVERSE IMAGE SEARCH (RIS) FORENSIC LOG  —  {now_str}")
        lines.append(sep_major)
        lines.append("")

        lines.append("[1]  INPUT CLAIM & VISUAL TARGET")
        lines.append(sep_minor)
        lines.append(f"    Original Claim     : {log_data.get('claim_text', 'N/A')}")
        lines.append(f"    Target Image Source: {log_data.get('image_type_desc', 'N/A')}")
        lines.append(f"    Public Search URL  : {log_data.get('search_url', 'N/A')}")
        lines.append(f"    Execution Backend  : {log_data.get('backend_llm', 'N/A')}")
        if "error_message" in log_data:
            lines.append(f"    Pipeline Error     : {log_data['error_message']}")
        lines.append("")

        lines.append("[2]  GOOGLE LENS SEARCH RETRIEVAL (SERPAPI)")
        lines.append(sep_minor)
        raw_matches = log_data.get("raw_matches", [])
        lines.append(f"    Search Engine Query: Google Lens (via SerpApi)")
        lines.append(f"    Total Retrieved    : {len(raw_matches)} candidate matches")
        lines.append(f"    Search Status      : {log_data.get('serpapi_status', 'N/A')}")
        if raw_matches:
            lines.append("")
            for idx, m in enumerate(raw_matches, 1):
                lines.append(f"      Match #{idx}:")
                lines.append(f"        Title    : {m.get('title', 'N/A')}")
                lines.append(f"        Source   : {m.get('source', 'N/A')}")
                lines.append(f"        Link     : {m.get('link', 'N/A')}")
                lines.append(f"        Thumbnail: {m.get('thumbnail', 'N/A')}")
        lines.append("")

        lines.append("[3]  VISUAL SIMILARITY FILTERING (CLIP: openai/clip-vit-base-patch32)")
        lines.append(sep_minor)
        vis_data = log_data.get("visual_filter", {})
        vis_thresh = vis_data.get("threshold", self.visual_threshold)
        lines.append(f"    Filter Criterion   : Visual embedding cosine similarity >= {vis_thresh:.2f}")
        vis_eval = vis_data.get("evaluated", [])
        if vis_eval:
            lines.append(f"    Evaluated Candidates ({len(vis_eval)}):")
            for idx, cand in enumerate(vis_eval, 1):
                passed = cand.get("visual_passed", False)
                verdict = "PASSED" if passed else "FAILED"
                sim_val = cand.get("visual_similarity")
                sim_str = f"{sim_val:.4f}" if sim_val is not None else "N/A"
                lines.append(f"      {idx}. [{verdict}] Visual Similarity: {sim_str} (Threshold: >={vis_thresh:.2f})")
                lines.append(f"         Title  : {cand.get('title', 'N/A')}")
                lines.append(f"         Source : {cand.get('source', 'N/A')}")
                lines.append(f"         Link   : {cand.get('link', 'N/A')}")
                if "visual_note" in cand:
                    lines.append(f"         Note   : {cand['visual_note']}")
            lines.append("")
            lines.append(f"    Outcome: {vis_data.get('passed_count', 0)} of {len(vis_eval)} candidates passed visual filter.")
        else:
            lines.append(f"    Status : {vis_data.get('status', 'Visual filtering skipped or no candidates.')}")
        lines.append("")

        lines.append("[4]  SEMANTIC CONTEXT FILTERING (MiniLM: all-MiniLM-L6-v2)")
        lines.append(sep_minor)
        sem_data = log_data.get("semantic_filter", {})
        sem_thresh = sem_data.get("threshold", self.semantic_threshold)
        lines.append(f"    Filter Criterion   : Contextual semantic embedding similarity >= {sem_thresh:.2f}")
        sem_eval = sem_data.get("evaluated", [])
        if sem_eval:
            lines.append(f"    Evaluated Contexts ({len(sem_eval)}):")
            for idx, cand in enumerate(sem_eval, 1):
                passed = cand.get("semantic_passed", False)
                verdict = "PASSED" if passed else "FAILED"
                sim_val = cand.get("semantic_similarity")
                sim_str = f"{sim_val:.4f}" if sim_val is not None else "N/A"
                lines.append(f"      {idx}. [{verdict}] Semantic Similarity: {sim_str} (Threshold: >={sem_thresh:.2f})")
                lines.append(f"         Title  : {cand.get('title', 'N/A')}")
                lines.append(f"         Source : {cand.get('source', 'N/A')}")
                lines.append(f"         Link   : {cand.get('link', 'N/A')}")
                if "semantic_note" in cand:
                    lines.append(f"         Note   : {cand['semantic_note']}")
            lines.append("")
            lines.append(f"    Outcome: {sem_data.get('passed_count', 0)} of {len(sem_eval)} candidates passed semantic filter.")
        else:
            lines.append(f"    Status : {sem_data.get('status', 'Semantic filtering skipped or no candidates.')}")
        lines.append("")

        lines.append("[5]  VISUAL AGENT DECONTEXTUALIZATION ANALYSIS")
        lines.append(sep_minor)
        va_data = log_data.get("visual_agent", {})
        lines.append(f"    Model Employed     : {va_data.get('model', 'N/A')}")
        lines.append(f"    Assigned Category  : {va_data.get('category', 'Unknown')}")
        ooc_va = "YES (Decontextualized / Misattributed)" if va_data.get("is_out_of_context") else "NO (Authentic Context / Non-Deceptive)"
        lines.append(f"    Out-of-Context (OOC): {ooc_va}")
        lines.append(f"    Confidence Level   : {va_data.get('confidence', 'Medium')}")
        lines.append("")
        lines.append("    Context Sent to Visual Agent:")
        ctx_sent = va_data.get("context_sent", "")
        if ctx_sent:
            for c_line in str(ctx_sent).strip().splitlines():
                lines.append(f"      {c_line}")
        else:
            lines.append("      (No context passed to Visual Agent)")
        lines.append("")
        lines.append("    Visual Agent Forensic Synthesis:")
        summ = va_data.get("summary", "")
        if summ:
            for s_line in str(summ).strip().splitlines():
                lines.append(f"      {s_line}")
        else:
            lines.append("      (No summary produced)")
        lines.append("")

        lines.append("[6]  EVIDENTIARY SUMMARY FORWARDED TO JUDGE MODULE")
        lines.append(sep_minor)
        j_summ = log_data.get("judge_summary") or va_data.get("judge_summary") or va_data.get("summary", "")
        cat_str = va_data.get("category", "Unknown")
        ooc_eval = "YES (Decontextualized / Misattributed)" if va_data.get("is_out_of_context") else "NO (Authentic Context / Non-Deceptive)"
        conf_eval = va_data.get("confidence", "Medium")
        lines.append(f"    Assigned Category          : {cat_str}")
        lines.append(f"    Out-of-Context Flag (OOC)  : {ooc_eval}")
        lines.append(f"    Judicial Confidence Weight : {conf_eval}")
        lines.append("")
        lines.append("    Executive Visual Evidence Summary for Judge:")
        if j_summ:
            for j_line in str(j_summ).strip().splitlines():
                lines.append(f"      {j_line}")
        else:
            lines.append("      (No visual evidence summary generated)")
        lines.append("")

        lines.append("[7]  FINAL VERDICT & PIPELINE SUMMARY")
        lines.append(sep_minor)
        fin = log_data.get("final_result", {})
        lines.append(f"    Pipeline Status       : {fin.get('status', 'Unknown')}")
        lines.append(f"    Forensic Category     : {fin.get('category', 'Unknown')}")
        fin_ooc = "YES (Decontextualized / Misattributed)" if fin.get("is_out_of_context") else "NO (Authentic Context / Non-Deceptive)"
        lines.append(f"    Out-of-Context (OOC)   : {fin_ooc}")
        lines.append(f"    Confidence Level      : {fin.get('confidence', 'Medium')}")
        lines.append(f"    High-Fidelity Matches : {fin.get('matches_count', 0)}")
        lines.append(f"    Forwarded to Judge    : {'YES' if fin.get('status') == 'success' else 'NO'}")
        lines.append("")

        lines.append(sep_major)
        lines.append("  END OF REVERSE IMAGE SEARCH FORENSIC LOG")
        lines.append(sep_major)
        lines.append("")

        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")
        print(f"  [RIS] Reverse image search log written -> {output_path}")
        return output_path

    def log_skipped(self, claim_text: str, reason: str = "No image provided with news query.", output_path=None):
        """Logs a skipped RIS run when no image is provided."""
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
            "claim_text": claim_text,
            "image_type_desc": "None (Text-only input)",
            "search_url": "N/A",
            "backend_llm": f"Local LLM ({_MODEL_ID})" if USE_LOCAL_LLM else f"OpenRouter ({_MODEL_ID})",
            "serpapi_status": f"Skipped — {reason}",
            "raw_matches": [],
            "visual_filter": {"status": f"Skipped — {reason}"},
            "semantic_filter": {"status": f"Skipped — {reason}"},
            "visual_agent": {
                "model": "N/A", 
                "category": "Skipped", 
                "is_out_of_context": False,
                "confidence": "N/A",
                "summary": f"Reverse Image Search was not performed because {reason.lower()}."
            },
            "judge_summary": f"No Reverse Image Search evidence available ({reason.lower()}). Evaluation will rely entirely on textual and factual evidence.",
            "final_result": {"status": "skipped", "category": "N/A", "is_out_of_context": False, "confidence": "N/A", "matches_count": 0}
        }
        return self.save_ris_log(log_data, output_path=output_path)
