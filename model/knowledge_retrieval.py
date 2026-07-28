"""
model/knowledge_retrieval.py

Knowledge Retrieval Pipeline for RAHMAD.
Provides verified, external supporting knowledge to the Judge module.
"""

import json
import logging
import os
from datetime import datetime
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

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODEL_ID       = "nvidia/nemotron-3-ultra-550b-a55b:free"

class KnowledgeRetriever:
    def __init__(self, similarity_threshold: float = 0.35, max_age_years: int = 3):
        self.similarity_threshold = similarity_threshold
        self.max_age_years = max_age_years
        
        # Load the embedding model for semantic verification
        # Using a small, efficient model suitable for this task
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            log.info("Successfully loaded sentence-transformers model.")
        except Exception as e:
            log.error(f"Failed to load sentence-transformers: {e}")
            self.embedder = None
            
        # Get OpenRouter API key
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
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
            "summary": "No external supporting knowledge available."
        }
        
        # 1. Link Retrieval
        print("  [Step 1/5] Searching external news sources...")
        urls = self.search_claim(claim)
        if not urls:
            print("  [Step 1/5] [X] No URLs found.")
            print("=" * 50 + "\n")
            result_info["summary"] = "No external URLs found for the claim."
            return result_info
        print(f"  [Step 1/5] [OK] Found {len(urls)} URLs.")
            
        # 2. Domain Filtering
        print("  [Step 2/5] Applying strict domain whitelist filter...")
        whitelisted_urls = self.filter_domains(urls)
        if not whitelisted_urls:
            print("  [Step 2/5] [X] No URLs passed domain whitelist.")
            print("=" * 50 + "\n")
            result_info["summary"] = "No URLs passed the domain whitelist filter."
            return result_info
        print(f"  [Step 2/5] [OK] {len(whitelisted_urls)} whitelisted URLs survived.")
            
        # 3. Metadata Extraction & Heuristic Filtering
        print("  [Step 3/5] Scraping metadata & applying heuristics...")
        articles = self.metadata_filter(whitelisted_urls)
        if not articles:
            print("  [Step 3/5] [X] No articles survived metadata filtering.")
            print("=" * 50 + "\n")
            result_info["summary"] = "No articles passed the metadata heuristic filter."
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
        if not snippets:
            print("  [Step 4/5] [X] No snippets passed semantic threshold.")
            print("=" * 50 + "\n")
            result_info["summary"] = "No text snippets passed the semantic similarity threshold."
            return result_info
        print(f"  [Step 4/5] [OK] {len(snippets)} snippet(s) passed similarity check.")
            
        # 5. LLM Synthesis
        print("  [Step 5/5] Synthesizing summary via OpenRouter LLM...")
        summary = self.synthesize(claim, snippets)
        result_info["summary"] = summary
        
        summary_preview = summary[:120].encode('ascii', 'ignore').decode('ascii')
        print("=" * 50)
        print("  KNOWLEDGE RETRIEVAL COMPLETE")
        print(f"  Summary Preview: {summary_preview}...")
        print("=" * 50 + "\n")
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
        if len(urls) < 3:
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
        Extracts metadata (title, text, date), removes non-English sources, 
        filters out outdated articles, and deduplicates by title.
        """
        articles = []
        seen_titles = set()
        current_year = datetime.now().year
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        
        for url in urls:
            try:
                title, text, date = "", "", ""
                
                # Fetch page HTML directly
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    continue
                    
                html_content = resp.text
                
                # Attempt trafilatura extraction first
                try:
                    result = trafilatura.extract(html_content, output_format="json", include_comments=False)
                    if result:
                        data = json.loads(result)
                        title = data.get("title", "")
                        text = data.get("raw_text", "")
                        date = data.get("date", "")
                except Exception:
                    pass

                # Fallback to BeautifulSoup extraction if trafilatura missed text/title
                if not text or not title:
                    soup = BeautifulSoup(html_content, "html.parser")
                    if not title and soup.title:
                        title = soup.title.text.strip()
                    if not text:
                        paragraphs = [p.text.strip() for p in soup.find_all("p") if len(p.text.strip()) > 30]
                        text = "\n\n".join(paragraphs)

                if not text or not title or len(text.strip()) < 100:
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
                        pass # Ignore parsing errors, keep article
                        
                articles.append({
                    "title": title,
                    "text": text,
                    "url": url,
                    "date": date
                })
                
                # Stop if we have 3 valid articles for performance
                if len(articles) >= 3:
                    break
                    
            except Exception as e:
                log.error(f"Error extracting metadata for {url}: {e}")
                
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
        if not self.api_key:
            return "\n\n".join(snippets)
            
        context = "\n\n".join([f"Snippet {i+1}:\n{s}" for i, s in enumerate(snippets)])
        
        system_prompt = (
            "You are an expert research analyst. Your task is to analyze verified news article snippets "
            "and construct a comprehensive, detailed factual outline that covers ALL key facts, entities "
            "(names, roles, organizations), dates, monetary amounts, court rulings, and event timelines. "
            "Do not give a brief overview; provide a full, structured summary of all evidence."
        )
        
        user_prompt = (
            f"Target Claim: {claim}\n\n"
            f"Verified Article Snippets:\n{context}\n\n"
            "Please provide a comprehensive key-fact outline containing:\n"
            "1. Key Entities, Names, and Roles involved\n"
            "2. Monetary Amounts, Legal Rulings, or Official Outcomes\n"
            "3. Timeline of Events and Corroborating Evidence"
        )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rah-mad", 
            "X-Title": "RAHMAD Knowledge Retrieval"
        }
        
        models_to_try = [
            _MODEL_ID,
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "inclusionai/ling-3.0-flash:free",
            "poolside/laguna-m.1:free",
        ]

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
                log.info(f"Calling OpenRouter API ({model}) for Knowledge Retrieval synthesis...")
                resp = requests.post(_OPENROUTER_URL, headers=headers, json=payload, timeout=40)
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
