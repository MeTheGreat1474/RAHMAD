"""
archive/data_processing/prepare_news_dataset.py

Extracts, cleans, labels, and shuffles True.csv and Fake.csv from:
C:\\Users\\ahmad\\Documents\\Programming\\fyp\\Dataset\\news dataset

Outputs:
  1. C:\\Users\\ahmad\\Documents\\Programming\\fyp\\Dataset\\news dataset\\test_dataset.csv
  2. testing/test_dataset_news.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_DIR = Path(r"C:\Users\ahmad\Documents\Programming\fyp\Dataset\news dataset")
OUTPUT_SOURCE_DIR = SOURCE_DIR / "test_dataset.csv"
OUTPUT_REPO_DIR = PROJECT_ROOT / "testing" / "test_dataset_news.csv"

def prepare_dataset(seed: int = 42):
    print("=" * 60)
    print("  EXTRACTING & MIXING TRUE & FAKE NEWS DATASET")
    print("=" * 60)
    
    true_path = SOURCE_DIR / "True.csv"
    fake_path = SOURCE_DIR / "Fake.csv"
    
    if not true_path.exists() or not fake_path.exists():
        raise FileNotFoundError(f"Missing True.csv or Fake.csv in {SOURCE_DIR}")
        
    print(f"Loading True.csv from {true_path}...")
    df_true = pd.read_csv(true_path)
    df_true["label"] = 0  # 0 = Real
    
    print(f"Loading Fake.csv from {fake_path}...")
    df_fake = pd.read_csv(fake_path)
    df_fake["label"] = 1  # 1 = Fake
    
    print(f"  Raw True count: {len(df_true):,}")
    print(f"  Raw Fake count: {len(df_fake):,}")
    
    # Concatenate
    df_combined = pd.concat([df_true, df_fake], ignore_index=True)
    print(f"  Total concatenated: {len(df_combined):,}")
    
    # Format 'text' column to include Title + Body for complete factual context
    title_clean = df_combined["title"].fillna("").astype(str).str.strip()
    body_clean = df_combined["text"].fillna("").astype(str).str.strip()
    
    df_combined["text"] = title_clean + "\n\n" + body_clean
    
    # Filter empty/short texts (< 50 characters)
    df_combined = df_combined[df_combined["text"].str.len() > 50].reset_index(drop=True)
    print(f"  After removing short texts (< 50 chars): {len(df_combined):,}")
    
    # Deduplicate based on text content
    df_combined = df_combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"  After deduplicating: {len(df_combined):,}")
    
    # Add empty image_path column for pipeline schema compatibility
    df_combined["image_path"] = ""
    
    # Reorder columns for clean inspection
    col_order = ["title", "text", "subject", "date", "label", "image_path"]
    df_combined = df_combined[[col for col in col_order if col in df_combined.columns]]
    
    # Stratified shuffle using fixed seed 42
    df_shuffled = df_combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # Save full combined dataset to source dir
    OUTPUT_SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
    df_shuffled.to_csv(OUTPUT_SOURCE_DIR, index=False, encoding="utf-8-sig")
    print(f"[OK] Full mixed dataset saved to:\n     {OUTPUT_SOURCE_DIR}")
    
    # Also save a standard 20% held-out test split (balanced) to testing/test_dataset_news.csv
    # Take stratified sample of 10,000 articles (5,000 Real, 5,000 Fake) for fast benchmarking
    _, df_test = train_test_split(
        df_shuffled,
        test_size=0.20,
        random_state=seed,
        stratify=df_shuffled["label"]
    )
    df_test = df_test.reset_index(drop=True)
    
    OUTPUT_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(OUTPUT_REPO_DIR, index=False, encoding="utf-8-sig")
    print(f"[OK] 20% test split ({len(df_test):,} articles) saved to:\n     {OUTPUT_REPO_DIR}")
    
    print("\nClass breakdown in test split:")
    print(df_test["label"].value_counts().rename({0: "Real News (0)", 1: "Fake News (1)"}))
    print("\nSubject distribution:")
    print(df_test["subject"].value_counts())
    print("\nDataset preparation complete!")

if __name__ == "__main__":
    prepare_dataset()
