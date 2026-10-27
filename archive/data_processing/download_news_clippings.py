"""
archive/data_processing/download_news_clippings.py

Automated Downloader & Builder for the NewsCLIPpings Multimodal Dataset.
Downloads the official test split metadata from Hugging Face (`g-luo/news-clippings`),
resolves live high-resolution news photographs directly from the publisher CDNs (BBC, The Guardian, Washington Post),
and produces a ready-to-test CSV with balanced Real (0) and Fake/Cheapfake (1) samples.
"""

import os
import sys
import json
import re
import time
import urllib.request
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_DIR = Path(r"C:\Users\ahmad\Documents\Programming\fyp\Dataset\news_clippings")
IMAGES_DIR = DATASET_DIR / "images"
OUTPUT_CSV = PROJECT_ROOT / "testing" / "test_dataset_newsclippings.csv"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


def resolve_article_url(source: str, full_article_path: str):
    """Maps internal VisualNews/NewsCLIPpings article path to public publisher URL."""
    if not full_article_path:
        return None
    
    if source == 'guardian':
        m = re.search(r'items:(.+)-200\.json', full_article_path)
        if m:
            clean_path = m.group(1).replace(':', '/')
            return f"https://www.theguardian.com/{clean_path}"
    elif source == 'bbc':
        m = re.search(r'news:(.+)-200\.json', full_article_path)
        if m:
            clean_path = m.group(1).replace(':', '/')
            return f"https://www.bbc.com/news/{clean_path}"
    elif source == 'washingtonpost':
        m = re.search(r'www\.washingtonpost\.com:(.+?):\.json', full_article_path)
        if not m:
            m = re.search(r'www\.washingtonpost\.com:(.+)', full_article_path)
        if m:
            clean = m.group(1).replace(':', '/')
            clean = re.sub(r'\.json$', '', clean)
            return f"https://www.washingtonpost.com/{clean}"
    return None


def fetch_image_url(article_url: str):
    """Scrapes og:image or twitter:image from article page."""
    try:
        req = urllib.request.Request(article_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
        
        m_img = re.search(r'property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', html)
        if not m_img:
            m_img = re.search(r'content=["\']([^"\']+)["\']\s+property=["\']og:image["\']', html)
        if not m_img:
            m_img = re.search(r'name=["\']twitter:image["\']\s+content=["\']([^"\']+)["\']', html)
        if m_img:
            return m_img.group(1)
    except Exception:
        pass
    return None


def download_image(img_url: str, save_path: Path):
    """Downloads image to disk."""
    try:
        req = urllib.request.Request(img_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            if len(data) > 1000:  # Valid image size
                with open(save_path, 'wb') as f:
                    f.write(data)
                return True
    except Exception:
        pass
    return False


def build_news_clippings_dataset(target_samples: int = 60):
    """Downloads balanced Real & Fake test samples from NewsCLIPpings."""
    print("=" * 70)
    print("  DOWNLOADING & BUILDING NEWSCLIPPINGS MULTIMODAL TEST DATASET")
    print("=" * 70)
    
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/3] Downloading official metadata & test split from Hugging Face...")
    p_meta = hf_hub_download(repo_id='g-luo/news-clippings', filename='metadata/test.json', repo_type='model')
    p_split = hf_hub_download(repo_id='g-luo/news-clippings', filename='data/merged_balanced/test.json', repo_type='model')
    
    with open(p_meta, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    with open(p_split, 'r', encoding='utf-8') as f:
        split = json.load(f)
        
    annotations = split['annotations']
    print(f"Loaded {len(annotations)} test annotations from NewsCLIPpings.")
    
    real_candidates = [a for a in annotations if not a['falsified']]
    fake_candidates = [a for a in annotations if a['falsified']]
    
    target_per_class = target_samples // 2
    print(f"Targeting {target_samples} balanced samples ({target_per_class} Real, {target_per_class} Fake)...")
    
    collected_rows = []
    
    for label, candidates, class_name in [(0, real_candidates, "Real"), (1, fake_candidates, "Fake/Cheapfake")]:
        print(f"\n[2/3] Resolving {target_per_class} {class_name} samples (Label={label})...")
        count = 0
        for ann in candidates:
            if count >= target_per_class:
                break
                
            text_info = meta.get(str(ann['id']))
            img_info = meta.get(str(ann['image_id']))
            
            if not text_info or not img_info:
                continue
                
            title = text_info.get('title', '').strip()
            caption = text_info.get('caption', '').strip()
            if not title or not caption:
                continue
                
            # Check image
            img_id = str(ann['image_id'])
            img_filename = f"nc_{img_id}.jpg"
            img_file_path = IMAGES_DIR / img_filename
            
            if not img_file_path.exists():
                art_url = resolve_article_url(img_info.get('source'), img_info.get('full_article_path'))
                if not art_url:
                    continue
                img_url = fetch_image_url(art_url)
                if not img_url:
                    continue
                success = download_image(img_url, img_file_path)
                if not success:
                    continue
                time.sleep(0.1)  # Polite crawling rate
            
            # Format combined text for RAHMAD
            full_text = f"{title}\n\n{caption}"
            
            collected_rows.append({
                'text': full_text,
                'label': label,
                'image_path': str(img_file_path),
                'title': title,
                'caption': caption,
                'source': text_info.get('source', ''),
                'text_id': ann['id'],
                'image_id': ann['image_id']
            })
            count += 1
            print(f"  [+] {class_name} ({count}/{target_per_class}): ID {ann['id']} -> {title[:60]}...")
            
    df = pd.DataFrame(collected_rows)
    # Shuffle order so classes are intermixed
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[3/3] Successfully saved test dataset with {len(df)} samples to:")
    print(f"  -> {OUTPUT_CSV}")
    print(f"  -> Images directory: {IMAGES_DIR}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print("=" * 70)
    return df


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    build_news_clippings_dataset(target_samples=n)
