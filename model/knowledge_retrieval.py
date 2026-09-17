"""
model/knowledge_retrieval.py

Knowledge Retrieval Pipeline for RAHMAD.
Provides verified, external supporting knowledge to the Judge module.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import requests
from duckduckgo_search import DDGS
import trafilatura
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
load_dotenv()

from bs4 import BeautifulSoup
from backend.domain_whitelist import is_domain_whitelisted

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
        "X-Title": "RAHMAD Knowledge Retrieval"
    }

class KnowledgeRetriever:
    def __init__(self, similarity_threshold: float = 0.35, max_age_years: int = 3, max_articles: int = 5):
        self.similarity_threshold = similarity_threshold
        self.max_age_years = max_age_years
        self.max_articles = max_articles
        
        # Load the embedding model for semantic verification
        # Using a small, efficient model suitable for this task
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            log.info("Successfully loaded sentence-transformers model.")
        except Exception as e:
            log.error(f"Failed to load sentence-transformers: {e}")
            self.embedder = None
            
        # Check API key if not using local LLM
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not USE_LOCAL_LLM and not self.api_key:
            log.warning("OPENROUTER_API_KEY not found in environment. LLM Synthesis will fail.")

    def retrieve(self, claim: str) -> str:
        """
        Orchestrates the knowledge retrieval pipeline and returns the summary text.
        """
        detailed = self.retrieve_detailed(claim)
        return detailed["summary"]

    def retrieve_detailed(self, claim: str) -> dict:
        """
        Orchestrates the knowledge retrieval pipeline and returns structured metadata
        including claim, articles retrieved (headline, source), and condensed summary.
        """
        print("\n" + "=" * 50)
        print("  KNOWLEDGE RETRIEVAL PIPELINE")
        print("=" * 50)
        print(f"  Claim Query : '{claim[:70]}...'")
        log.info(f"Starting Knowledge Retrieval for claim: '{claim}'")
        result_info = {
            "claim": claim,
            "articles": [],
            "summary": "No external supporting knowledge available.",
            "llm_used": "Local LLM" if USE_LOCAL_LLM else "OpenRouter"
        }
        
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
            "claim": claim,
            "llm_used": result_info["llm_used"],
            "urls_found_count": 0,
            "raw_urls": [],
            "urls_whitelisted_count": 0,
            "whitelisted_urls": [],
            "articles_extracted": [],
            "semantic_snippets": [],
            "summary": "",
            "status": "in_progress"
        }
        
        try:
            # 1. Link Retrieval
            print("  [Step 1/5] Searching external news sources...")
            urls = self.search_claim(claim)
            log_data["urls_found_count"] = len(urls)
            log_data["raw_urls"] = urls
            if not urls:
                print("  [Step 1/5] [X] No URLs found.")
                print("=" * 50 + "\n")
                result_info["summary"] = "No external URLs found for the claim."
                log_data["status"] = "failed - no urls"
                self.save_kr_log(log_data)
                return result_info
            print(f"  [Step 1/5] [OK] Found {len(urls)} URLs.")
                
            # 2. Domain Filtering
            print("  [Step 2/5] Applying strict domain whitelist filter...")
            whitelisted_urls = self.filter_domains(urls)
            log_data["urls_whitelisted_count"] = len(whitelisted_urls)
            log_data["whitelisted_urls"] = whitelisted_urls
            if not whitelisted_urls:
                print("  [Step 2/5] [X] No URLs passed domain whitelist.")
                print("=" * 50 + "\n")
                result_info["summary"] = "No URLs passed the domain whitelist filter."
                log_data["status"] = "failed - no whitelisted urls"
                self.save_kr_log(log_data)
                return result_info
            print(f"  [Step 2/5] [OK] {len(whitelisted_urls)} whitelisted URLs survived.")
                
            # 3. Metadata Extraction & Heuristic Filtering
            print("  [Step 3/5] Scraping metadata & applying heuristics...")
            articles = self.metadata_filter(whitelisted_urls)
            log_data["articles_extracted"] = articles
            if not articles:
                print("  [Step 3/5] [X] No articles survived metadata filtering.")
                print("=" * 50 + "\n")
                result_info["summary"] = "No articles passed the metadata heuristic filter."
                log_data["status"] = "failed - no metadata"
                self.save_kr_log(log_data)
                return result_info
            print(f"  [Step 3/5] [OK] {len(articles)} verified article(s) extracted.")
                
            # Store headline and source for output logging
            result_info["articles"] = [
                {"headline": art.get("title", "No Title"), "source": art.get("url", ""), "date": art.get("date", "")}
                for art in articles
            ]
            for i, art in enumerate(articles, 1):
                title_str = art.get('title', 'No Title').encode('ascii', 'ignore').decode('ascii')
                print(f"       {i}. {title_str} ({art.get('url', '')})")
                
            # 4. Semantic Verification
            print(f"  [Step 4/5] Computing semantic similarity (Threshold >= {self.similarity_threshold})...")
            snippets = self.semantic_filter(claim, articles)
            log_data["semantic_snippets"] = snippets
            if not snippets:
                print("  [Step 4/5] [X] No snippets passed semantic threshold.")
                print("=" * 50 + "\n")
                result_info["summary"] = "No text snippets passed the semantic similarity threshold."
                log_data["status"] = "failed - no semantic snippets"
                self.save_kr_log(log_data)
                return result_info
            print(f"  [Step 4/5] [OK] {len(snippets)} snippet(s) passed similarity check.")
                
            # 5. LLM Synthesis
            print(f"  [Step 5/5] Synthesizing summary via {'Local LLM' if USE_LOCAL_LLM else 'OpenRouter LLM'}...")
            summary = self.synthesize(claim, snippets)
            result_info["summary"] = summary
            log_data["summary"] = summary
            log_data["status"] = "success"
            
            summary_preview = summary[:120].encode('ascii', 'ignore').decode('ascii')
            print("=" * 50)
            print("  KNOWLEDGE RETRIEVAL COMPLETE")
            print(f"  Summary Preview: {summary_preview}...")
            print("=" * 50 + "\n")
            
            self.save_kr_log(log_data)
            return result_info
            
        except Exception as e:
            log.error(f"Error in KR pipeline: {e}")
            log_data["status"] = f"error: {e}"
            self.save_kr_log(log_data)
            result_info["summary"] = f"Knowledge Retrieval encountered an error: {e}"
            return result_info

    def search_claim(self, claim: str, max_results: int = 10) -> List[str]:
        """
        Queries DuckDuckGo HTML endpoint and DDGS for news articles matching the claim.
        Returns a list of direct article URLs.
        """
        log.info(f"Searching news sources for: '{claim}'")
        urls = []
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        
        # 1. Primary: DuckDuckGo HTML endpoint (Zero rate limits, direct URLs)
        try:
            resp = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": claim},
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", class_="result__url"):
                    href = a.get("href", "")
                    if href and href.startswith("http") and href not in urls:
                        urls.append(href)
        except Exception as e:
            log.warning(f"DuckDuckGo HTML search error: {e}")

        # 2. Secondary: DDGS package fallback
        if len(urls) < 5:
            try:
                ddgs = DDGS()
                try:
                    results = list(ddgs.news(claim, max_results=max_results))
                    for res in results:
                        if 'url' in res and res['url'] not in urls:
                            urls.append(res['url'])
                except Exception as ne:
                    log.warning(f"DDGS news search error: {ne}")
            except Exception as e:
                log.error(f"DDGS search error: {e}")
                
        log.info(f"Retrieved {len(urls)} URLs across sources.")
        return urls

    def filter_domains(self, urls: List[str]) -> List[str]:
        """
        Applies a strict, rule-based filter against a whitelist of verified news outlets.
        """
        filtered = [url for url in urls if is_domain_whitelisted(url)]
        log.info(f"Domain filtering: {len(urls)} -> {len(filtered)} surviving URLs.")
        return filtered

    def metadata_filter(self, urls: List[str]) -> List[Dict]:
        """
        Extracts metadata (title, text, date) via a Scrapy subprocess spider,
        filters out outdated articles, and deduplicates by title.
        """
        if not urls:
            return []

        articles = []
        seen_titles = set()
        current_year = datetime.now().year

        project_root = Path(__file__).resolve().parent.parent
        spider_file = project_root / "backend" / "news_spider.py"

        raw_articles = []

        # Execute Scrapy spider inside an isolated temporary directory to keep output/ clean
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_urls_file = tmp_path / "kr_target_urls.json"
            results_file = tmp_path / "kr_results.json"

            # 1. Write target URLs to JSON file
            try:
                with open(target_urls_file, "w", encoding="utf-8") as f:
                    json.dump(urls, f, indent=2)
            except Exception as e:
                log.error(f"Failed to write target URLs to {target_urls_file}: {e}")
                return []

            # 2. Execute Scrapy spider as an isolated subprocess
            cmd = [
                sys.executable, "-m", "scrapy", "runspider",
                str(spider_file),
                "-a", f"urls_file={target_urls_file}",
                "-O", str(results_file),
                "--nolog"
            ]

            log.info(f"Running Scrapy spider subprocess for {len(urls)} URLs...")
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
                if res.returncode != 0:
                    log.warning(f"Scrapy subprocess exited with code {res.returncode}: {res.stderr}")
            except subprocess.TimeoutExpired:
                log.warning("Scrapy subprocess timed out after 25s.")
            except Exception as e:
                log.error(f"Error running Scrapy subprocess: {e}")

            # 3. Read extracted results
            if results_file.exists():
                try:
                    with open(results_file, "r", encoding="utf-8") as f:
                        raw_articles = json.load(f)
                except Exception as e:
                    log.error(f"Failed to parse Scrapy results JSON: {e}")

        # Fallback to direct extraction if Scrapy didn't return any items
        if not raw_articles:
            log.info("Scrapy returned 0 items. Attempting direct fallback extraction...")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            for u in urls:
                try:
                    resp = requests.get(u, headers=headers, timeout=8)
                    if resp.status_code == 200:
                        t, txt, dt = "", "", ""
                        try:
                            ext = trafilatura.extract(resp.text, output_format="json", include_comments=False)
                            if ext:
                                d = json.loads(ext)
                                t = d.get("title", "") or ""
                                txt = d.get("raw_text", "") or ""
                                dt = d.get("date", "") or ""
                        except Exception:
                            pass
                        if not t or not txt:
                            s = BeautifulSoup(resp.text, "html.parser")
                            if not t and s.title:
                                t = s.title.text.strip()
                            if not txt:
                                paras = [p.text.strip() for p in s.find_all("p") if len(p.text.strip()) > 30]
                                txt = "\n\n".join(paras)
                        if t and txt and len(txt.strip()) >= 100:
                            raw_articles.append({"url": u, "title": t, "text": txt, "date": dt})
                except Exception as ex:
                    log.debug(f"Direct fallback failed for {u}: {ex}")

        # 4. Deduplication, age validation, and selection
        for art in raw_articles:
            title = art.get("title", "").strip()
            text = art.get("text", "").strip()
            date = art.get("date", "").strip()
            url = art.get("url", "").strip()

            if not text or not title or len(text) < 100:
                continue

            # Deduplication
            title_lower = title.lower()
            if title_lower in seen_titles:
                continue
            seen_titles.add(title_lower)

            # Timestamp check (naive year check)
            if date:
                try:
                    pub_year = int(date.split('-')[0])
                    if (current_year - pub_year) > self.max_age_years:
                        log.debug(f"Article '{title}' is outdated ({pub_year}). Skipping.")
                        continue
                except Exception:
                    pass

            articles.append({
                "title": title,
                "text": text,
                "url": url,
                "date": date
            })

            if len(articles) >= self.max_articles:
                break

        log.info(f"Metadata filtering: {len(articles)} articles survived.")
        return articles

    def semantic_filter(self, claim: str, articles: List[Dict]) -> List[str]:
        """
        Calculates cosine similarity between claim and article text.
        Returns all top relevant text snippets per article that pass the threshold.
        """
        if not self.embedder:
            log.warning("Embedder not loaded, skipping semantic filter.")
            return [" ".join(art['text'].split()[:300]) for art in articles]
            
        snippets = []
        try:
            claim_embedding = self.embedder.encode([claim])
            
            for art in articles:
                text = art['text']
                # Split text into paragraphs/chunks
                chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 50]
                if not chunks:
                    chunks = [text[:1000]]
                    
                chunk_embeddings = self.embedder.encode(chunks)
                similarities = cosine_similarity(claim_embedding, chunk_embeddings)[0]
                
                # Keep top matching chunks (up to 3 per article) above similarity threshold
                passing_indices = [idx for idx, score in enumerate(similarities) if score >= self.similarity_threshold]
                passing_indices = sorted(passing_indices, key=lambda idx: similarities[idx], reverse=True)[:3]
                
                for idx in passing_indices:
                    snippets.append(f"[{art['title']}] {chunks[idx]}")
                    log.debug(f"Chunk from '{art['title']}' passed with score {similarities[idx]:.2f}.")
                    
        except Exception as e:
            log.error(f"Error during semantic filtering: {e}")
            
        log.info(f"Semantic filtering: {len(snippets)} snippets survived.")
        return snippets

    def synthesize(self, claim: str, snippets: List[str]) -> str:
        """
        Calls OpenRouter API to produce a detailed, key-facts synthesis outlining
        all critical entities, dates, rulings, and figures across all articles.
        """
        if not USE_LOCAL_LLM and not self.api_key:
            return "\n\n".join(snippets)
            
        context = "\n\n".join([f"Snippet {i+1}:\n{s}" for i, s in enumerate(snippets)])
        
        system_prompt = (
            "You are an expert research analyst. Your task is to analyze verified news article snippets "
            "and construct a comprehensive, detailed factual outline that covers ALL key facts, entities "
            "(names, roles, organizations), dates, monetary amounts, court rulings, and event timelines.\n"
            "CRITICAL RULES:\n"
            "1. ONLY include facts explicitly stated in the snippets.\n"
            "2. DO NOT infer, guess, or extrapolate dates, locations, or outcomes.\n"
            "3. If a specific date or year is not mentioned, state 'Date not specified'.\n"
            "4. Provide a full, structured summary of the evidence presented without adding external knowledge."
        )
        
        user_prompt = (
            f"Target Claim: {claim}\n\n"
            f"Verified Article Snippets:\n{context}\n\n"
            "Please provide a comprehensive key-fact outline containing:\n"
            "1. Key Entities, Names, and Roles involved\n"
            "2. Monetary Amounts, Legal Rulings, or Official Outcomes\n"
            "3. Timeline of Events and Corroborating Evidence\n\n"
            "Remember: DO NOT hallucinate or infer information (especially dates or years) not explicitly found in the snippets."
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

        for model in models_to_try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1200
            }
            try:
                log.info(f"Calling {'Local LLM' if USE_LOCAL_LLM else 'OpenRouter API'} ({model}) for Knowledge Retrieval synthesis...")
                resp = requests.post(_API_URL, headers=headers, json=payload, timeout=40)
                resp.raise_for_status()
                
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    summary = data["choices"][0]["message"]["content"]
                    log.info(f"LLM Synthesis complete via {model}.")
                    return summary.strip()
                else:
                    log.warning(f"Model {model} response missing choices: {data}")
            except Exception as e:
                log.warning(f"Error during LLM Synthesis with {model}: {e}")

        return "\n\n".join(snippets) # Fallback

    def save_kr_log(self, log_data: dict, output_path=None):
        """
        Log all Knowledge Retrieval stages (Search, Filter, Extraction, Semantic, LLM)
        into KR_output.txt inside the output/ folder.
        """
        from pathlib import Path
        if output_path is None:
            project_root = Path(__file__).resolve().parent.parent
            output_dir = project_root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "KR_output.txt"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        now_str = log_data.get("timestamp") or datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        lines = []
        sep_major = "=" * 80
        sep_minor = "-" * 60

        lines.append(sep_major)
        lines.append(f"  KNOWLEDGE RETRIEVAL (KR) FORENSIC LOG  —  {now_str}")
        lines.append(sep_major)
        lines.append("")

        lines.append("[1]  INPUT CLAIM")
        lines.append(sep_minor)
        lines.append(f"    Original Claim     : {log_data.get('claim', 'N/A')}")
        lines.append(f"    Execution Backend  : {log_data.get('llm_used', 'N/A')}")
        lines.append("")

        lines.append("[2]  LINK RETRIEVAL & DOMAIN FILTERING")
        lines.append(sep_minor)
        raw_urls = log_data.get("raw_urls", [])
        whitelisted = log_data.get("whitelisted_urls", [])
        lines.append(f"    Total URLs Retrieved (Pre-Filter) : {len(raw_urls)}")
        if raw_urls:
            lines.append("    All Retrieved URLs (Pre-Filter Status):")
            for idx, u in enumerate(raw_urls, 1):
                status_tag = "[WHITELISTED]" if u in whitelisted else "[FILTERED OUT]"
                lines.append(f"      {idx:2d}. {status_tag:<15} {u}")
        lines.append("")
        lines.append(f"    URLs Passing Whitelist : {len(whitelisted)}")
        if whitelisted:
            for idx, u in enumerate(whitelisted, 1):
                lines.append(f"      {idx}. {u}")
        lines.append("")

        lines.append("[3]  METADATA EXTRACTION")
        lines.append(sep_minor)
        articles = log_data.get("articles_extracted", [])
        lines.append(f"    Articles Successfully Extracted : {len(articles)}")
        for idx, art in enumerate(articles, 1):
            lines.append(f"      {idx}. Title  : {art.get('title', 'N/A')}")
            lines.append(f"         Source : {art.get('url', 'N/A')}")
            lines.append(f"         Date   : {art.get('date', 'N/A')}")
        lines.append("")

        lines.append(f"[4]  SEMANTIC VERIFICATION (all-MiniLM-L6-v2) - Threshold >= {self.similarity_threshold}")
        lines.append(sep_minor)
        snippets = log_data.get("semantic_snippets", [])
        lines.append(f"    Snippets Passed : {len(snippets)}")
        for idx, snip in enumerate(snippets, 1):
            lines.append(f"      Snippet #{idx}:")
            for ln in str(snip).strip().splitlines():
                lines.append(f"        {ln}")
        lines.append("")

        lines.append("[5]  LLM SYNTHESIS & CONDENSED SUMMARY")
        lines.append(sep_minor)
        summary = log_data.get("summary", "")
        lines.append("    Generated Key-Facts Summary:")
        if summary:
            for s_line in str(summary).strip().splitlines():
                lines.append(f"      {s_line}")
        else:
            lines.append("      (No summary produced)")
        lines.append("")
        
        lines.append("[6]  FINAL PIPELINE STATUS")
        lines.append(sep_minor)
        lines.append(f"    Status : {log_data.get('status', 'Unknown')}")
        lines.append("")

        lines.append(sep_major)
        lines.append("  END OF KNOWLEDGE RETRIEVAL FORENSIC LOG")
        lines.append(sep_major)
        lines.append("")

        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")
        print(f"  [KR] Knowledge retrieval log written -> {output_path}")
        return output_path
