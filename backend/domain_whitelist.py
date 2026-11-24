"""
backend/domain_whitelist.py

Contains the whitelist of verified, reputable news outlets and fact-checkers.
Used by the Knowledge Retrieval pipeline to filter out untrustworthy or unverified sources.

CORROBORATION RULES (for publisher_stats in knowledge_retrieval.py):
- Tier 1 (wire / major outlets): counts as independent corroboration on its own.
- Tier 2 (trade / financial / crypto press): counts as independent corroboration ONLY
  when there are >= 2 DISTINCT registrable domains at Tier 2 (or 1 Tier 1 + 1 Tier 2).
- Tier 3 (primary / official sources): NEVER counts as independent corroboration.
  These are company press releases, SEC filings, or IR pages — an imposter/fabricated
  article can mimic them, so they prove the company SAID something, not that the
  underlying event is independently verified.
"""

# ---- Tier 1: Wire / Major Outlets (independent corroboration) ----
TIER1_DOMAINS = {
    # News Agencies
    "reuters.com",
    "apnews.com",
    "afp.com",
    "bloomberg.com",
    "news.google.com",

    # News portals & aggregators. MSN News (msn.com) republishes Reuters/AP/
    # CNBC wire stories, so its URLs are treated as Tier 1. The subdomain
    # check also covers news.msn.com and www.msn.com/en-us/news/… articles.
    "msn.com",
    "yahoo.com",
    
    # US Outlets
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",
    "npr.org",
    "pbs.org",
    "cnn.com",
    "nbcnews.com",
    "cbsnews.com",
    "abcnews.go.com",
    
    # UK Outlets
    "bbc.com",
    "bbc.co.uk",
    "theguardian.com",
    "ft.com",
    "thetimes.co.uk",
    "independent.co.uk",
    "telegraph.co.uk",
    
    # International / Regional
    "aljazeera.com",
    "theaustralian.com.au",
    "abc.net.au",
    "smh.com.au",
    "scmp.com",
    "channelnewsasia.com",
    "straitstimes.com",
    "thehindu.com",
    "rfi.fr",
    "dw.com",
    "france24.com",
    "euronews.com",
    "thejakartapost.com",
    "kompas.com",
    "tempo.co",
    "antaranews.com",
    "dawn.com",
    "gmanetwork.com",
    
    # Malaysian News Outlets
    "freemalaysiatoday.com",
    "thestar.com.my",
    "malaysiakini.com",
    "nst.com.my",
    "malaymail.com",
    "bernama.com",
    "astroawani.com",
    "theedgemarkets.com",
    
    # Science, Tech & Health
    "nasa.gov",
    "space.com",
    "nature.com",
    "sciencemag.org",
    "scientificamerican.com",
    "newscientist.com",
    "techcrunch.com",
    "wired.com",
    "arstechnica.com",
    "theverge.com",
    "cdc.gov",
    "who.int",
    "nih.gov",
    
    # Additional Major News & Analysis
    "time.com",
    "forbes.com",
    "businessinsider.com",
    "politico.com",
    "usatoday.com",
    "latimes.com",
    
    # Fact-checking organizations
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    
    # Extended Whitelist (major)
    "usnews.com",
    "bangkokpost.com",
    "theedgemalaysia.com",
    "asiaone.com",
}

# ---- Tier 2: Trade / Financial / Crypto Press (needs >= 2 distinct domains) ----
TIER2_DOMAINS = {
    "theblock.co",
    "coindesk.com",
    "cointelegraph.com",
    "cryptoslate.com",
    "unchainedcrypto.com",
    "bitcoinmagazine.com",
    "decrypt.co",
    "crypto.news",
    "cryptobriefing.com",
    "marketwatch.com",
    "barrons.com",
    "theinformation.com",
    "benzinga.com",
    "investing.com",
    "coinmarketcap.com",
    "tipranks.com",
}

# ---- Tier 3: Primary / Official Sources (NEVER corroborate independently) ----
TIER3_DOMAINS = {
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
    "sec.gov",
    "ir.nvidia.com",
}

# Union for backward compatibility with existing code
VERIFIED_DOMAINS = TIER1_DOMAINS | TIER2_DOMAINS | TIER3_DOMAINS


def is_domain_whitelisted(url: str) -> bool:
    """
    Checks if the given URL belongs to a whitelisted domain (any tier).
    """
    import urllib.parse
    
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower().rstrip('.')
        
        # Remove 'www.' if present
        if netloc.startswith("www."):
            netloc = netloc[4:]
            
        # Check if the exact domain or a subdomain matches
        for domain in VERIFIED_DOMAINS:
            if netloc == domain or netloc.endswith("." + domain):
                return True
                
        return False
    except Exception:
        return False


def _get_tier_for_netloc(netloc: str) -> int | None:
    """Internal helper: return 1, 2, or 3 for a normalized netloc, or None if not whitelisted."""
    # netloc is already lowercased, no trailing dot, no www.
    for domain in TIER1_DOMAINS:
        if netloc == domain or netloc.endswith("." + domain):
            return 1
    for domain in TIER2_DOMAINS:
        if netloc == domain or netloc.endswith("." + domain):
            return 2
    for domain in TIER3_DOMAINS:
        if netloc == domain or netloc.endswith("." + domain):
            return 3
    return None


def domain_tier(url: str) -> int | None:
    """
    Returns the tier (1, 2, or 3) for a whitelisted URL, or None if not whitelisted.
    """
    import urllib.parse
    
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower().rstrip('.')
        
        if netloc.startswith("www."):
            netloc = netloc[4:]
            
        return _get_tier_for_netloc(netloc)
    except Exception:
        return None


# Known multi-part public suffixes where registrable domain = 3 labels
_MULTI_PART_SUFFIXES = {
    "co.uk", "com.au", "co.jp", "com.my", "com.sg",
    "com.in", "com.br", "co.za", "com.hk", "com.tw",
    "co.nz", "co.il", "co.kr", "co.th", "co.id",
    "org.uk", "ac.uk", "gov.uk", "ne.jp", "or.jp",
}


def registrable_domain(url: str) -> str:
    """
    Returns the registrable domain for counting distinct publishers.
    Strips leading www., lowercases, then takes the last 2 labels (or 3 for known
    multi-part public suffixes). On parse failure returns the raw netloc lowercased.
    """
    import urllib.parse
    
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower().rstrip('.')
        
        if netloc.startswith("www."):
            netloc = netloc[4:]
        
        labels = netloc.split('.')
        if len(labels) <= 1:
            return netloc
        
        # Check if the last two labels form a known multi-part suffix
        if len(labels) >= 2:
            last_two = labels[-2] + "." + labels[-1]
            if last_two in _MULTI_PART_SUFFIXES and len(labels) >= 3:
                return ".".join(labels[-3:])
        
        return ".".join(labels[-2:])
    except Exception:
        return netloc.lower() if 'netloc' in locals() else ""