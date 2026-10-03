"""
backend/domain_whitelist.py

Contains the whitelist of verified, reputable news outlets and fact-checkers.
Used by the Knowledge Retrieval pipeline to filter out untrustworthy or unverified sources.
"""

VERIFIED_DOMAINS = {
    # News Agencies
    "reuters.com",
    "apnews.com",
    "afp.com",
    "bloomberg.com",
    "news.google.com",
    
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
    "msn.com",
    
    # Fact-checking organizations
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    
    # Extended Whitelist
    "usnews.com",
    "yahoo.com",
    "bangkokpost.com",
    "theedgemalaysia.com",
    "asiaone.com",
}

def is_domain_whitelisted(url: str) -> bool:
    """
    Checks if the given URL belongs to a whitelisted domain.
    """
    import urllib.parse
    
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        
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
