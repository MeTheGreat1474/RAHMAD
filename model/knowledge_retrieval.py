"""
model/knowledge_retrieval.py

Knowledge Retrieval Pipeline for RAHMAD.
Provides verified, external supporting knowledge to the Judge module.
"""

import concurrent.futures
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
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

import urllib.parse
from bs4 import BeautifulSoup
from backend.domain_whitelist import is_domain_whitelisted, domain_tier, registrable_domain

log = logging.getLogger(__name__)

# Social media / non-article domains to exclude from search results
SOCIAL_MEDIA_DOMAINS = {
    "facebook.com", "threads.net", "threads.com", "twitter.com", "x.com",
    "instagram.com", "tiktok.com", "reddit.com", "youtube.com", "linkedin.com",
    "pinterest.com", "weibo.com"
}

def is_social_media_url(url: str) -> bool:
    """Returns True if the URL points to a social media or user-generated platform."""
    if not url:
        return False
    try:
        netloc = urllib.parse.urlparse(url).netloc.lower()
        return any(soc == netloc or netloc.endswith("." + soc) for soc in SOCIAL_MEDIA_DOMAINS)
    except Exception:
        return False


def build_publisher_stats(urls: List[str]) -> dict:
    """
    Build publisher statistics from a list of whitelisted URLs.
    Returns a dict with distinct publisher count, tier breakdown, and corroboration strength.
    """
    from collections import defaultdict
    
    domain_to_tier = {}
    domain_counts = defaultdict(int)
    
    for url in urls:
        tier = domain_tier(url)
        if tier is None:
            continue
        rd = registrable_domain(url)
        if rd:
            domain_counts[rd] += 1
            # Keep the minimum (highest priority) tier seen for this domain
            if rd not in domain_to_tier or tier < domain_to_tier[rd]:
                domain_to_tier[rd] = tier
    
    # Build domain list with tier and count
    domains_list = []
    tier1_count = 0
    tier2_count = 0
    tier3_count = 0
    
    for rd, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        tier = domain_to_tier.get(rd, 0)
        domains_list.append({"domain": rd, "tier": tier, "count": count})
        if tier == 1:
            tier1_count += 1
        elif tier == 2:
            tier2_count += 1
        elif tier == 3:
            tier3_count += 1
    
    # Independent distinct domains = Tier 1 + Tier 2 (Tier 3 never corroborates)
    independent_domains = tier1_count + tier2_count
    total_distinct = len(domain_counts)
    
    # Corroboration rule: MULTI_PUBLISHER if (>=1 Tier 1 AND >=2 independent) OR (>=2 Tier 2)
    if (tier1_count >= 1 and independent_domains >= 2) or (tier2_count >= 2):
        corroboration = "MULTI_PUBLISHER"
    else:
        corroboration = "SINGLE_PUBLISHER"
    
    return {
        "distinct_publishers": total_distinct,
        "domains": domains_list,
        "tier1": tier1_count,
        "tier2": tier2_count,
        "tier3": tier3_count,
        "corroboration": corroboration,
    }


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

        # Search depth management. A single search often returns mostly
        # non-whitelisted domains (aggregators, blogs), so we keep searching
        # with broader and then site-restricted queries.
        #
        #   target_whitelisted_urls : aim for this many whitelisted articles
        #   min_whitelisted_urls    : hard floor — below this, keep searching
        #   expansion_time_budget   : wall-clock cap for the expansion phase;
        #                             on timeout we proceed with what we have
        self.target_whitelisted_urls = 10
        self.min_whitelisted_urls = 3
        # attempt 1 = initial search, then up to 4 expansion passes.
        self.max_search_attempts = 5
        self.expansion_time_budget = 60.0
        self.initial_max_results = 10
        self.expanded_max_results = 20
        
        # Load the embedding model for semantic verification
        # Using a small, efficient model suitable for this task
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            log.info("Successfully loaded sentence-transformers model.")
        except Exception as e:
            log.error(f"Failed to load sentence-transformers: {e}")
            self.embedder = None
            
        # Store article titles found during search or fallback extraction
        self.search_titles: Dict[str, str] = {}

        # Check API key if not using local LLM
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not USE_LOCAL_LLM and not self.api_key:
            log.warning("OPENROUTER_API_KEY not found in environment. LLM Synthesis will fail.")

    def extract_title_from_url_slug(self, url: str, claim_text: str = "") -> str:
        """
        Derives a clean, readable news headline from a URL path slug.
        Used as a robust fallback for unextractable whitelisted links (e.g. Reuters).
        """
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.replace('www.', '').split('.')[0].upper()
            path = parsed.path.strip('/')
            segments = [s for s in path.split('/') if s]
            if not segments:
                return f"{domain} News Report"

            slug = segments[-1]
            slug = re.sub(r'\.(html|htm|php|asp|aspx|amp)$', '', slug, flags=re.I)
            slug = re.sub(r'[-_]\d{4}[-_]\d{1,2}[-_]\d{1,2}', '', slug)
            slug = re.sub(r'[-_]\d{8,}', '', slug)

            # If slug is just an ID (e.g. idRW138209092026RP1), check video/watch or preceding segments
            if re.match(r'^(id)?[a-z0-9_-]{5,}$', slug, re.I) and not any(c in slug for c in '-_ '):
                if any(v in [s.lower() for s in segments] for v in ('video', 'watch', 'multimedia')):
                    first_line = claim_text.strip().splitlines()[0] if claim_text else ""
                    clean_claim = re.sub(r'^[A-Z\s/]+,\s*[A-Za-z]+\s*\d+\s*\([A-Za-z]+\)\s*[-–—]\s*', '', first_line).strip()
                    if clean_claim and len(clean_claim) > 10:
                        return f"{clean_claim} (Video)"
                    return f"{domain} Video Report"
                meaningful = [s for s in segments[:-1] if s.lower() not in ('article', 'articles', 'news', 'watch', 'video', 'story', 'post', 'posts', 'view')]
                if meaningful:
                    slug = f"{meaningful[-1]} {slug}"
                else:
                    return f"{domain} News Coverage"

            words = re.split(r'[-_]+', slug)
            words = [w.capitalize() for w in words if w]
            title = ' '.join(words)
            return title if len(title) > 3 else f"{domain} News Report"
        except Exception:
            return "Verified News Article"


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
            self.search_titles = {}
            urls = self.search_claim(claim, max_results=self.initial_max_results)
            log_data["urls_found_count"] = len(urls)
            log_data["raw_urls"] = urls
            if not urls:
                print("  [Step 1/5] [X] No URLs found.")
                print("=" * 50 + "\n")
                result_info["summary"] = "No external URLs found for the claim."
                # Empty publisher stats and evidence block
                empty_stats = {"distinct_publishers": 0, "domains": [], "tier1": 0, "tier2": 0, "tier3": 0, "corroboration": "NONE"}
                result_info["publisher_stats"] = empty_stats
                log_data["publisher_stats"] = empty_stats
                result_info["evidence_block"] = (
                    f"Distinct publishers: 0 (Tier 1: 0 | Tier 2: 0)\n"
                    f"Publisher domains: \n"
                    f"Corroboration strength: NONE\n\n"
                    f"=== RETRIEVED REPORT SNIPPETS (cite by ID as [S1], [S2], ...) ===\n"
                    f"No snippet-level evidence retrieved.\n\n"
                    f"=== SYNTHESIS ===\n{result_info['summary']}"
                )
                log_data["status"] = "failed - no urls"
                self.save_kr_log(log_data)
                return result_info
            print(f"  [Step 1/5] [OK] Found {len(urls)} URLs.")
                
            # 2. Domain Filtering
            print("  [Step 2/5] Applying strict domain whitelist filter...")
            whitelisted_urls = self.filter_domains(urls)

            # 2b. Search expansion. The first query typically returns mostly
            # non-whitelisted aggregators/blogs. Keep searching (broader query,
            # then site-restricted batches) until we have
            # `target_whitelisted_urls` whitelisted articles, the attempts run
            # out, or `expansion_time_budget` seconds elapse. Minimum viable
            # coverage is `min_whitelisted_urls`; below that we still keep
            # going, but the time budget is the fail-safe.
            search_attempts = 1
            expansion_log = []
            expansion_started = time.monotonic()
            deadline = expansion_started + self.expansion_time_budget
            expansion_stop_reason = "target reached" if len(whitelisted_urls) >= self.target_whitelisted_urls else ""

            while (len(whitelisted_urls) < self.target_whitelisted_urls
                   and search_attempts < self.max_search_attempts):
                remaining = deadline - time.monotonic()
                if remaining <= 1.0:
                    expansion_stop_reason = "time budget exhausted"
                    log.warning("KR expansion stopped: %.1fs budget exhausted with %d whitelisted URL(s).",
                                self.expansion_time_budget, len(whitelisted_urls))
                    print(f"  [Step 2/5] [!] Expansion budget exhausted "
                          f"({self.expansion_time_budget:.0f}s) — using {len(whitelisted_urls)} whitelisted URL(s).")
                    break

                search_attempts += 1
                print(f"  [Step 2/5] {len(whitelisted_urls)} whitelisted URL(s) "
                      f"(target {self.target_whitelisted_urls}) — expansion search "
                      f"{search_attempts}/{self.max_search_attempts}, {remaining:.0f}s left...")
                try:
                    extra_urls = self._run_expansion_with_deadline(claim, search_attempts - 1, remaining)
                except Exception as exp_err:
                    log.warning(f"KR expansion search {search_attempts} failed: {exp_err}")
                    extra_urls = []

                new_urls = [u for u in extra_urls if u not in urls]
                urls.extend(new_urls)
                whitelisted_urls = self.filter_domains(urls)
                expansion_log.append({
                    "attempt": search_attempts,
                    "query": getattr(self, "last_search_query", ""),
                    "new_urls": len(new_urls),
                    "whitelisted_after": len(whitelisted_urls),
                })
                print(f"  [Step 2/5] Expansion search {search_attempts} added "
                      f"{len(new_urls)} new URL(s) — {len(whitelisted_urls)} whitelisted now.")

            if not expansion_stop_reason:
                if len(whitelisted_urls) >= self.target_whitelisted_urls:
                    expansion_stop_reason = "target reached"
                elif search_attempts >= self.max_search_attempts:
                    expansion_stop_reason = "attempt cap reached"
                else:
                    expansion_stop_reason = "time budget exhausted"
            if len(whitelisted_urls) < self.min_whitelisted_urls:
                print(f"  [Step 2/5] [!] Only {len(whitelisted_urls)} whitelisted URL(s) "
                      f"(minimum {self.min_whitelisted_urls}) — proceeding with what was found.")
                log.warning("KR below minimum whitelisted coverage: %d < %d (stop: %s).",
                            len(whitelisted_urls), self.min_whitelisted_urls, expansion_stop_reason)

            log_data["search_attempts"] = search_attempts
            log_data["expansion_log"] = expansion_log
            log_data["expansion_stop_reason"] = expansion_stop_reason
            log_data["expansion_elapsed_s"] = round(time.monotonic() - expansion_started, 1)
            log_data["urls_found_count"] = len(urls)
            log_data["raw_urls"] = urls
            log_data["urls_whitelisted_count"] = len(whitelisted_urls)
            log_data["whitelisted_urls"] = whitelisted_urls
            if not whitelisted_urls:
                print("  [Step 2/5] [X] No URLs passed domain whitelist.")
                print("=" * 50 + "\n")
                result_info["summary"] = "No URLs passed the domain whitelist filter."
                # Empty publisher stats and evidence block
                empty_stats = {"distinct_publishers": 0, "domains": [], "tier1": 0, "tier2": 0, "tier3": 0, "corroboration": "NONE"}
                result_info["publisher_stats"] = empty_stats
                log_data["publisher_stats"] = empty_stats
                result_info["evidence_block"] = (
                    f"Distinct publishers: 0 (Tier 1: 0 | Tier 2: 0)\n"
                    f"Publisher domains: \n"
                    f"Corroboration strength: NONE\n\n"
                    f"=== RETRIEVED REPORT SNIPPETS (cite by ID as [S1], [S2], ...) ===\n"
                    f"No snippet-level evidence retrieved.\n\n"
                    f"=== SYNTHESIS ===\n{result_info['summary']}"
                )
                log_data["status"] = "failed - no whitelisted urls"
                self.save_kr_log(log_data)
                return result_info
            print(f"  [Step 2/5] [OK] {len(whitelisted_urls)} whitelisted URLs survived.")
                
            # 3. Metadata Extraction & Heuristic Filtering
            print("  [Step 3/5] Scraping metadata & applying heuristics...")
            extracted_articles = self.metadata_filter(whitelisted_urls)
            log_data["articles_extracted"] = extracted_articles

            # Build unified list of all whitelisted articles for supporting references (both extracted and unextractable)
            extracted_by_url = {art["url"]: art for art in extracted_articles}
            all_whitelisted_articles = []
            unextractable_whitelisted = []
            seen_urls = set()
            seen_titles = set()

            for u in whitelisted_urls:
                if u in seen_urls:
                    continue
                seen_urls.add(u)

                if u in extracted_by_url:
                    art = extracted_by_url[u]
                    title = art.get("title", "")
                    date_val = art.get("date", "")
                    is_extracted = True
                else:
                    # Unextractable article (e.g. Reuters 401 anti-bot, paywall, video page)
                    is_extracted = False
                    date_val = ""
                    # Try to get title from search engine metadata or direct scrape
                    title = self.search_titles.get(u, "")
                    if not title:
                        title = self.extract_title_from_url_slug(u, claim)
                    else:
                        title = re.sub(
                            r'\s*[-–—|]\s*(Reuters|The Associated Press|AP News|US News|BBC News|CNN|Bloomberg|Financial Times|The Guardian).*$',
                            '', title, flags=re.I
                        ).strip()

                title_norm = title.lower().strip()
                if title_norm in seen_titles and is_extracted is False:
                    continue
                seen_titles.add(title_norm)

                item = {
                    "headline": title if title else "Verified News Report",
                    "source": u,
                    "date": date_val,
                    "extracted": is_extracted
                }
                all_whitelisted_articles.append(item)
                if not is_extracted:
                    unextractable_whitelisted.append(item)

            log_data["unextractable_whitelisted"] = unextractable_whitelisted
            result_info["articles"] = all_whitelisted_articles

            if not extracted_articles:
                if all_whitelisted_articles:
                    print(f"  [Step 3/5] [!] Text extraction restricted by publisher policies for {len(all_whitelisted_articles)} article(s). Preserving supporting links.")
                    for i, art in enumerate(all_whitelisted_articles, 1):
                        title_str = art.get('headline', 'No Title').encode('ascii', 'ignore').decode('ascii')
                        print(f"       {i}. {title_str} [Supporting Link - Scraping Restricted] ({art.get('source', '')})")
                    result_info["summary"] = (
                        "Verified news articles from whitelisted reporting outlets were identified for this claim, "
                        "but full textual extraction was restricted by publisher anti-scraping protections (e.g. Reuters). "
                        "Please refer to the verified supporting links for primary reporting."
                    )
                    # Build publisher stats and evidence block even without semantic snippets
                    stats = build_publisher_stats(log_data.get("whitelisted_urls", []))
                    result_info["publisher_stats"] = stats
                    log_data["publisher_stats"] = stats
                    result_info["evidence_block"] = (
                        f"Distinct publishers: {stats['distinct_publishers']} (Tier 1: {stats['tier1']} | Tier 2: {stats['tier2']})\n"
                        f"Publisher domains: {', '.join(d['domain'] + ' (Tier ' + str(d['tier']) + ')' for d in stats['domains'] if d['tier'] <= 2)}\n"
                        f"Corroboration strength: {stats['corroboration']}\n\n"
                        f"=== RETRIEVED REPORT SNIPPETS (cite by ID as [S1], [S2], ...) ===\n"
                        f"No snippet-level evidence retrieved.\n\n"
                        f"=== SYNTHESIS ===\n{result_info['summary']}"
                    )
                    log_data["status"] = "completed - supporting links preserved"
                    self.save_kr_log(log_data)
                    return result_info
                else:
                    print("  [Step 3/5] [X] No articles survived metadata filtering.")
                    print("=" * 50 + "\n")
                    result_info["summary"] = "No articles passed the metadata heuristic filter."
                    log_data["status"] = "failed - no metadata"
                    self.save_kr_log(log_data)
                    return result_info

            print(f"  [Step 3/5] [OK] {len(extracted_articles)} article(s) text-extracted, {len(all_whitelisted_articles)} supporting reference(s) preserved.")
            for i, art in enumerate(all_whitelisted_articles, 1):
                title_str = art.get('headline', 'No Title').encode('ascii', 'ignore').decode('ascii')
                status_note = "[Verified]" if art.get("extracted") else "[Supporting Link]"
                print(f"       {i}. {title_str} {status_note} ({art.get('source', '')})")

            # 4. Semantic Verification
            print(f"  [Step 4/5] Computing semantic similarity (Threshold >= {self.similarity_threshold})...")
            snippets = self.semantic_filter(claim, extracted_articles)

            log_data["semantic_snippets"] = snippets
            if not snippets:
                print("  [Step 4/5] [X] No snippets passed semantic threshold.")
                print("=" * 50 + "\n")
                result_info["summary"] = "No text snippets passed the semantic similarity threshold."
                # Build publisher stats and evidence block even without semantic snippets
                stats = build_publisher_stats(log_data.get("whitelisted_urls", []))
                result_info["publisher_stats"] = stats
                log_data["publisher_stats"] = stats
                result_info["evidence_block"] = (
                    f"Distinct publishers: {stats['distinct_publishers']} (Tier 1: {stats['tier1']} | Tier 2: {stats['tier2']})\n"
                    f"Publisher domains: {', '.join(d['domain'] + ' (Tier ' + str(d['tier']) + ')' for d in stats['domains'] if d['tier'] <= 2)}\n"
                    f"Corroboration strength: {stats['corroboration']}\n\n"
                    f"=== RETRIEVED REPORT SNIPPETS (cite by ID as [S1], [S2], ...) ===\n"
                    f"No snippet-level evidence retrieved.\n\n"
                    f"=== SYNTHESIS ===\n{result_info['summary']}"
                )
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
            
            # Build publisher stats and evidence block from whitelisted URLs
            stats = build_publisher_stats(log_data.get("whitelisted_urls", []))
            result_info["publisher_stats"] = stats
            log_data["publisher_stats"] = stats
            
            # Build evidence block for Judge consumption
            snippet_lines = []
            for idx, snip in enumerate(snippets, 1):
                # Collapse internal newlines to spaces, keep one line per snippet
                one_line = " ".join(str(snip).strip().splitlines())
                snippet_lines.append(f"[S{idx}] {one_line}")
            
            evidence_block = (
                f"Distinct publishers: {stats['distinct_publishers']} (Tier 1: {stats['tier1']} | Tier 2: {stats['tier2']})\n"
                f"Publisher domains: {', '.join(d['domain'] + ' (Tier ' + str(d['tier']) + ')' for d in stats['domains'] if d['tier'] <= 2)}\n"
                f"Corroboration strength: {stats['corroboration']}\n\n"
                f"=== RETRIEVED REPORT SNIPPETS (cite by ID as [S1], [S2], ...) ===\n"
            )
            if snippet_lines:
                evidence_block += "\n".join(snippet_lines) + "\n\n"
            else:
                evidence_block += "No snippet-level evidence retrieved.\n\n"
            evidence_block += f"=== SYNTHESIS ===\n{summary}"
            
            result_info["evidence_block"] = evidence_block
            
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

    @staticmethod
    def _build_search_query(claim: str) -> str:
        """Reduce the article text to a concise search query (usually the headline)."""
        # Clean and isolate concise search query (prefer headline, strip datelines/wire prefixes)
        lines = [l.strip() for l in claim.strip().splitlines() if l.strip()]
        search_query = lines[0] if lines else claim[:120].strip()
        # Remove common news wire datelines like "SINGAPORE/JAKARTA, Sept 9 (Reuters) -"
        search_query = re.sub(r"^[A-Z\s/]+,\s*[A-Za-z]+\s*\d+\s*\([A-Za-z]+\)\s*[-–—]\s*", "", search_query).strip()
        # If still over 140 chars, truncate to first complete sentence or 140 chars
        if len(search_query) > 140:
            first_sent = search_query.split(".")[0].strip()
            if len(first_sent) > 20:
                search_query = first_sent
            else:
                search_query = search_query[:140].strip()
        return search_query

    def search_claim(self, claim: str, max_results: int = 10, query_override: str | None = None) -> List[str]:
        """
        Queries search providers (DuckDuckGo HTML, DDGS, Google News RSS, and SerpAPI Google News fallback)
        for news articles matching the claim.
        Returns a list of direct article URLs.

        `query_override` replaces the headline-derived query — used by the
        expansion passes to run broader and site-restricted searches.
        """
        # 1. Clean and isolate concise search query (prefer headline, strip datelines/wire prefixes)
        search_query = (query_override or self._build_search_query(claim)).strip()

        log.info(f"Searching news sources for query: '{search_query}' (original claim length: {len(claim)})")
        print(f"       Search Query: '{search_query}'")
        self.last_search_query = search_query
        urls = []
        # Titles accumulate across expansion passes so preserved supporting
        # links keep their headlines; retrieve_detailed resets it once.
        if not isinstance(getattr(self, "search_titles", None), dict):
            self.search_titles = {}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
        
        # 1. Primary: DuckDuckGo HTML endpoint
        try:
            resp = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": search_query},
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for res_div in soup.find_all("div", class_="result"):
                    a_link = res_div.find("a", class_="result__url")
                    a_title = res_div.find("a", class_="result__a")
                    href = ""
                    if a_link and a_link.get("href"):
                        href = a_link.get("href", "").strip()
                    elif a_title and a_title.get("href"):
                        href = a_title.get("href", "").strip()
                    if "uddg=" in href:
                        try:
                            pq = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                            if "uddg" in pq:
                                href = pq["uddg"][0]
                        except Exception:
                            pass
                    if href and href.startswith("http") and not is_social_media_url(href):
                        if href not in urls:
                            urls.append(href)
                        if a_title and a_title.text.strip():
                            self.search_titles[href] = a_title.text.strip()
                # Fallback URL parsing if div.result wasn't matched
                for a in soup.find_all("a", class_="result__url"):
                    href = a.get("href", "").strip()
                    if "uddg=" in href:
                        try:
                            pq = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                            if "uddg" in pq:
                                href = pq["uddg"][0]
                        except Exception:
                            pass
                    if href and href.startswith("http") and not is_social_media_url(href) and href not in urls:
                        urls.append(href)
        except Exception as e:
            log.warning(f"DuckDuckGo HTML search error: {e}")

        # 2. Secondary: DDGS package fallback
        if len(urls) < 5:
            try:
                ddgs = DDGS()
                try:
                    results = list(ddgs.news(search_query, max_results=max_results))
                    for res in results:
                        u = res.get('url', '')
                        title = res.get('title', '')
                        if u and u.startswith("http") and not is_social_media_url(u):
                            if u not in urls:
                                urls.append(u)
                            if title:
                                self.search_titles[u] = title.strip()
                except Exception as ne:
                    log.warning(f"DDGS news search error: {ne}")
            except Exception as e:
                log.error(f"DDGS search error: {e}")

        # 3. Tertiary: Google News RSS fallback (resilient, no API key required, bypasses IP bans)
        if len(urls) < 5:
            try:
                log.info(f"Using Google News RSS fallback for query: '{search_query}'")
                print(f"       [Google News RSS Fallback] Querying Google News RSS feed...")
                import xml.etree.ElementTree as ET
                try:
                    from googlenewsdecoder import gnewsdecoder
                except ImportError:
                    gnewsdecoder = None

                q_encoded = urllib.parse.quote(search_query)
                rss_url = f"https://news.google.com/rss/search?q={q_encoded}&hl=en-US&gl=US&ceid=US:en"
                rss_resp = requests.get(rss_url, headers=headers, timeout=10)
                if rss_resp.status_code == 200:
                    root = ET.fromstring(rss_resp.content)
                    items = root.findall(".//item")
                    for item in items[:max_results]:
                        link_elem = item.find("link")
                        title_elem = item.find("title")
                        item_title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                        if link_elem is not None and link_elem.text:
                            raw_link = link_elem.text.strip()
                            resolved_url = raw_link
                            if gnewsdecoder:
                                try:
                                    decoded = gnewsdecoder(raw_link)
                                    if (decoded.get("status") or decoded.get("success")) and decoded.get("decoded_url"):
                                        resolved_url = decoded["decoded_url"]
                                except Exception as de:
                                    log.debug(f"Google News RSS decode error: {de}")
                            
                            if (resolved_url.startswith("http") 
                                    and not is_social_media_url(resolved_url)):
                                if resolved_url not in urls:
                                    urls.append(resolved_url)
                                if item_title:
                                    self.search_titles[resolved_url] = item_title
                log.info(f"Google News RSS retrieved {len(urls)} URLs so far.")
            except Exception as rss_err:
                log.warning(f"Google News RSS fallback error: {rss_err}")

        # 4. Quaternary: SerpAPI Google News fallback (resilient to bot-blocking and rate limits)
        if len(urls) < 3:
            serpapi_key = os.environ.get("SERPAPI_API_KEY")
            if serpapi_key:
                try:
                    from serpapi import GoogleSearch
                    log.info(f"Using SerpAPI Google News fallback for query: '{search_query}'")
                    print(f"       [SerpAPI Fallback] Querying Google News via SerpAPI...")
                    clean_query = f"{search_query} -site:facebook.com -site:twitter.com -site:x.com -site:threads.net -site:threads.com -site:instagram.com"
                    search = GoogleSearch({
                        "engine": "google",
                        "q": clean_query,
                        "tbm": "nws",
                        "api_key": serpapi_key,
                        "num": max_results,
                    })
                    res_dict = search.get_dict()
                    for item in res_dict.get("news_results", []):
                        link = item.get("link")
                        title = item.get("title", "")
                        if link and link.startswith("http") and not is_social_media_url(link):
                            if link not in urls:
                                urls.append(link)
                            if title:
                                self.search_titles[link] = title.strip()
                    for item in res_dict.get("organic_results", []):
                        link = item.get("link")
                        title = item.get("title", "")
                        if link and link.startswith("http") and not is_social_media_url(link):
                            if link not in urls:
                                urls.append(link)
                            if title:
                                self.search_titles[link] = title.strip()
                    log.info(f"SerpAPI retrieved {len(urls)} URLs so far.")
                except Exception as se:
                    log.warning(f"SerpAPI search fallback error: {se}")
                
        # Final safety filter: remove any social media or non-news links
        urls = [u for u in urls if not is_social_media_url(u)]
        log.info(f"Retrieved {len(urls)} URLs across sources.")
        return urls

    # Outlets targeted by the site-restricted expansion passes. Batch 1 is
    # wire/major outlets (MSN News included), batch 2 is trade/financial/crypto,
    # batch 3 is additional Tier-1 majors so later passes cover new outlets
    # instead of repeating the same results.
    _SITE_RESTRICTED_BATCHES = (
        ("reuters.com", "apnews.com", "bloomberg.com", "cnbc.com", "msn.com",
         "yahoo.com", "nytimes.com", "bbc.com", "theguardian.com", "wsj.com"),
        ("decrypt.co", "coindesk.com", "cointelegraph.com", "theblock.co",
         "cryptoslate.com", "marketwatch.com", "barrons.com", "forbes.com",
         "businessinsider.com", "usatoday.com"),
        ("cnn.com", "nbcnews.com", "cbsnews.com", "npr.org", "latimes.com",
         "time.com", "politico.com", "washingtonpost.com", "telegraph.co.uk",
         "independent.co.uk"),
    )

    @classmethod
    def _expansion_query(cls, base: str, attempt: int) -> str:
        """Query for expansion pass `attempt` (1-based)."""
        if attempt <= 1:
            return f"{base} news"
        batch_index = attempt - 2
        if batch_index < len(cls._SITE_RESTRICTED_BATCHES):
            batch = cls._SITE_RESTRICTED_BATCHES[batch_index]
            return f"{base} ({' OR '.join('site:' + d for d in batch)})"
        # Past the curated batches: broaden rather than repeat.
        return f"{base} latest report"

    def search_claim_expanded(self, claim: str, attempt: int) -> List[str]:
        """
        Extra search pass used when the search is still short of
        `target_whitelisted_urls` whitelisted articles.

        attempt 1 : broader keyword query with a larger result cap.
        attempt 2 : site-restricted to wire/major outlets (incl. MSN News).
        attempt 3 : site-restricted to trade/financial/crypto outlets.
        attempt 4 : site-restricted to additional Tier-1 majors.
        """
        query = self._expansion_query(self._build_search_query(claim), attempt)

        preview = query if len(query) <= 110 else query[:110] + "..."
        print(f"  [Step 1/5] [R] Expansion search #{attempt}: {preview}")
        return self.search_claim(
            claim,
            max_results=self.expanded_max_results,
            query_override=query,
        )

    def _run_expansion_with_deadline(self, claim: str, attempt: int, timeout: float) -> List[str]:
        """
        Run one expansion search, abandoning it once `timeout` seconds elapse.

        Each search pass can make up to four provider calls with 10 s request
        timeouts, so a slow chain could otherwise overshoot the expansion
        budget. The worker thread is left to finish on its own; its only
        side effect is extra entries in `search_titles`.
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self.search_claim_expanded, claim, attempt)
            return list(future.result(timeout=max(1.0, timeout)))
        except concurrent.futures.TimeoutError:
            log.warning("Expansion search %d exceeded its %.1fs slice; abandoning it.", attempt, timeout)
            print(f"  [Step 2/5] [!] Expansion search {attempt} timed out — moving on.")
            return []
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def filter_domains(self, urls: List[str]) -> List[str]:
        """
        Applies a strict, rule-based filter against a whitelist of verified news outlets.
        """
        filtered = [url for url in urls if is_domain_whitelisted(url)]
        # Per-tier counts for logging
        tier_counts = {1: 0, 2: 0, 3: 0}
        for url in filtered:
            t = domain_tier(url)
            if t:
                tier_counts[t] += 1
        log.info(f"Domain filtering: {len(urls)} -> {len(filtered)} surviving URLs (Tier1={tier_counts[1]}, Tier2={tier_counts[2]}, Tier3={tier_counts[3]}).")
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

        # Fallback to direct extraction for any URLs Scrapy did not extract
        extracted_spider_urls = {a.get("url") for a in raw_articles if a.get("url")}
        missing_urls = [u for u in urls if u not in extracted_spider_urls]
        if missing_urls:
            log.info(f"Attempting direct extraction for {len(missing_urls)} URL(s)...")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            for u in missing_urls:
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
                            if not t:
                                og_t = s.find("meta", property="og:title")
                                if og_t and og_t.get("content"):
                                    t = og_t["content"].strip()
                            if not txt:
                                paras = [p.text.strip() for p in s.find_all("p") if len(p.text.strip()) > 30]
                                txt = "\n\n".join(paras)
                        if t and not self.search_titles.get(u):
                            self.search_titles[u] = t
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
                resp = requests.post(_API_URL, headers=headers, json=payload, timeout=120)
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
        lines.append(f"    Search Attempts     : {log_data.get('search_attempts', 1)}")
        lines.append(f"    Whitelist Target    : {getattr(self, 'target_whitelisted_urls', 10)} "
                     f"(minimum {getattr(self, 'min_whitelisted_urls', 3)})")
        if "expansion_stop_reason" in log_data:
            lines.append(f"    Expansion Stopped   : {log_data.get('expansion_stop_reason')} "
                         f"after {log_data.get('expansion_elapsed_s', 0)}s")
        for exp in log_data.get("expansion_log", []) or []:
            q = exp.get("query", "")
            q_preview = q if len(q) <= 90 else q[:90] + "..."
            lines.append(
                f"      - Expansion #{exp.get('attempt')}: +{exp.get('new_urls', 0)} new URL(s), "
                f"{exp.get('whitelisted_after', 0)} whitelisted | query: {q_preview}"
            )
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

        # Publisher Statistics (distinct registrable domains, tiers, corroboration)
        stats = log_data.get("publisher_stats")
        if stats:
            lines.append("[2.1]  PUBLISHER STATISTICS")
            lines.append(sep_minor)
            lines.append(f"    Distinct Publishers : {stats.get('distinct_publishers', 0)}")
            lines.append(f"    Tier 1 (Wire/Major)  : {stats.get('tier1', 0)}")
            lines.append(f"    Tier 2 (Trade/Finance/Crypto) : {stats.get('tier2', 0)}")
            lines.append(f"    Tier 3 (Primary/PR)  : {stats.get('tier3', 0)}")
            lines.append(f"    Corroboration Strength: {stats.get('corroboration', 'UNKNOWN')}")
            lines.append("    Domain Breakdown:")
            for d in stats.get("domains", []):
                lines.append(f"      - {d['domain']} (Tier {d['tier']}) x {d['count']}")
            lines.append("")

        lines.append("[3]  METADATA EXTRACTION")
        lines.append(sep_minor)
        articles = log_data.get("articles_extracted", [])
        lines.append(f"    Articles Successfully Extracted : {len(articles)}")
        for idx, art in enumerate(articles, 1):
            lines.append(f"      {idx}. Title  : {art.get('title', 'N/A')}")
            lines.append(f"         Source : {art.get('url', 'N/A')}")
            lines.append(f"         Date   : {art.get('date', 'N/A')}")

        unextractable = log_data.get("unextractable_whitelisted", [])
        if unextractable:
            lines.append("")
            lines.append(f"    Whitelisted Supporting References (Unextractable / Anti-Bot Protected) : {len(unextractable)}")
            for idx, art in enumerate(unextractable, 1):
                lines.append(f"      {idx}. Title  : {art.get('headline', 'N/A')}")
                lines.append(f"         Source : {art.get('source', 'N/A')}")
                lines.append(f"         Status : Preserved for Supporting Links")
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
