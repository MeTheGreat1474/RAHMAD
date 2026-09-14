"""
backend/news_spider.py

Scrapy Spider for the Knowledge Retrieval module.
Crawls candidate news URLs concurrently and extracts article metadata (title, text, date)
using trafilatura with a BeautifulSoup fallback.
"""

import json
import logging
import scrapy
import trafilatura
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


class NewsArticleSpider(scrapy.Spider):
    name = "news_article_spider"

    custom_settings = {
        'USER_AGENT': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        ),
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_TIMEOUT': 10,
        'RETRY_ENABLED': False,
        'CONCURRENT_REQUESTS': 8,
        'LOG_LEVEL': 'WARNING',
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
        }
    }

    def __init__(self, urls_file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.urls_file = urls_file
        self.start_urls = []

        if urls_file:
            try:
                with open(urls_file, "r", encoding="utf-8") as f:
                    self.start_urls = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load target URLs from {urls_file}: {e}")

    def parse(self, response):
        """
        Extracts title, text, and date using trafilatura with BeautifulSoup fallback.
        """
        html_content = response.text
        title, text, date = "", "", ""

        # 1. Primary extraction via trafilatura
        try:
            res = trafilatura.extract(html_content, output_format="json", include_comments=False)
            if res:
                data = json.loads(res)
                title = data.get("title", "") or ""
                text = data.get("raw_text", "") or ""
                date = data.get("date", "") or ""
        except Exception as e:
            self.logger.debug(f"Trafilatura error on {response.url}: {e}")

        # 2. Secondary fallback via BeautifulSoup
        if not text or not title:
            try:
                soup = BeautifulSoup(html_content, "html.parser")
                if not title and soup.title:
                    title = soup.title.text.strip()
                if not text:
                    paragraphs = [p.text.strip() for p in soup.find_all("p") if len(p.text.strip()) > 30]
                    text = "\n\n".join(paragraphs)
            except Exception as e:
                self.logger.debug(f"BeautifulSoup fallback error on {response.url}: {e}")

        # Basic validity check
        if not text or not title or len(text.strip()) < 100:
            return

        yield {
            "url": response.url,
            "title": title.strip(),
            "text": text.strip(),
            "date": date.strip() if date else ""
        }
