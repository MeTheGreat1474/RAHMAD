import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import io
import re
import json
import time
import requests
import subprocess
import pandas as pd
from PIL import Image

def get_mediaproxy_url(raw_url: str) -> str:
    if not isinstance(raw_url, str) or not raw_url.strip():
        return ""
    clean = raw_url.split("?")[0].strip()
    # Normalize snopes tachyon and uploads
    for prefix in [
        "https://www.snopes.com/tachyon/",
        "http://www.snopes.com/tachyon/",
        "https://snopes.com/tachyon/",
        "http://snopes.com/tachyon/",
        "https://www.snopes.com/uploads/",
        "http://www.snopes.com/uploads/",
        "https://snopes.com/uploads/",
        "http://snopes.com/uploads/"
    ]:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break
            
    # Also handle leading dates or wordpress uploads
    clean = re.sub(r"^wordpress/wp-content/uploads/", "", clean)
    clean = re.sub(r"^images/photos/", "", clean)
    return f"https://mediaproxy.snopes.com/width/1200/https://media.snopes.com/{clean}"

def download_image(url: str, dest_path: Path, headers: dict) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        try:
            with Image.open(dest_path) as im:
                im.verify()
            return True
        except Exception:
            pass
            
    candidate_urls = [
        get_mediaproxy_url(url),
        url.split("?")[0],
        url
    ]
    
    for candidate in candidate_urls:
        if not candidate:
            continue
        try:
            r = requests.get(candidate, headers=headers, timeout=12)
            if r.status_code == 200 and len(r.content) > 1000:
                img = Image.open(io.BytesIO(r.content))
                img.convert("RGB").save(dest_path, "JPEG", quality=92)
                return True
        except Exception:
            continue
    return False

def main():
    print("=" * 70)
    print("  DOWNLOADING FAUXTOGRAPHY (ZLATKOVA ET AL. 2019) DATASET & IMAGES")
    print("=" * 70)

    repo_dir = PROJECT_ROOT.parent.parent / ".gemini" / "antigravity-ide" / "brain" / "fd166934-7396-43fe-887c-fe428159e56f" / "scratch" / "fauxtography_repo"
    if not repo_dir.exists():
        # Fallback check inside scratch
        repo_dir = Path(r"C:\Users\ahmad\.gemini\antigravity-ide\brain\fd166934-7396-43fe-887c-fe428159e56f\scratch\fauxtography_repo")

    dataset_images_dir = Path(r"C:\Users\ahmad\Documents\Programming\fyp\Dataset\images_fauxtography")
    if dataset_images_dir.parent.exists():
        images_dir = dataset_images_dir
    else:
        images_dir = PROJECT_ROOT / "testing" / "images_fauxtography"
    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target Images Directory: {images_dir}")

    # 1. Load official test split
    print("\n[1] Loading official test set (data/processed/test/media_test.csv)...")
    res_test = subprocess.run(["git", "show", "HEAD:data/processed/test/media_test.csv"], cwd=repo_dir, capture_output=True, text=True, errors="replace")
    df_test = pd.read_csv(io.StringIO(res_test.stdout))
    print(f"Loaded {len(df_test)} test claims from paper repository.")

    # 2. Load full snopes dataset for potential balance augmentation
    print("\n[2] Loading full Snopes dataset for verified True class augmentation...")
    res_full = subprocess.run(["git", "show", "HEAD:data/processed/snopes/media_2019-02-02 15:59:21.csv"], cwd=repo_dir, capture_output=True, text=True, errors="replace")
    df_full = pd.read_csv(io.StringIO(res_full.stdout))
    print(f"Loaded {len(df_full)} total claims from Snopes collection.")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.snopes.com/",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
    }

    # Process and download official test set
    processed_records = []
    print("\n[3] Downloading images for official test set...")
    success_count = 0

    for idx, row in df_test.iterrows():
        claim = str(row["claim"]).strip()
        img_url = str(row["img"]).strip()
        label_raw = str(row["label"]).strip().lower()
        slug = str(row.get("url_caption", f"fauxtography_test_{idx}")).strip()
        slug_clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", slug)[:60]
        dest_filename = f"{idx:03d}_{slug_clean}.jpg"
        dest_path = images_dir / dest_filename

        ok = download_image(img_url, dest_path, headers)
        if ok:
            success_count += 1
            # In binary classification:
            # label 1 = Fake (false, mixture)
            # label 0 = Real (true)
            bin_label = 0 if label_raw == "true" else 1
            processed_records.append({
                "id": idx,
                "text": claim,
                "label": bin_label,
                "original_label": label_raw,
                "image_path": str(dest_path.resolve()),
                "image_filename": dest_filename,
                "image_url": img_url,
                "source": "Zlatkova_EMNLP_2019_Snopes_Test"
            })
            print(f"  [{idx+1}/{len(df_test)}] [OK] {label_raw.upper():<7} -> {dest_filename}")
        else:
            print(f"  [{idx+1}/{len(df_test)}] [FAILED] {label_raw.upper():<7} -> {img_url}")

    print(f"\nOfficial Test Set Images: {success_count} / {len(df_test)} successfully downloaded ({success_count/len(df_test)*100:.1f}%).")

    df_test_out = pd.DataFrame(processed_records)
    out_test_path = PROJECT_ROOT / "testing" / "test_dataset_fauxtography_official_test.csv"
    df_test_out.to_csv(out_test_path, index=False)
    print(f"Saved official test split dataset to: {out_test_path}")

    # Now create balanced 60-sample benchmark matching the Phase 3 protocol
    # In test set:
    # We have True claims (~15), False claims (~32), Mixture claims (~25)
    # To have a clean 50/50 Real/Fake split of 60 samples:
    # 30 Real (True) + 30 Fake (False/Mixture)
    print("\n[4] Creating a balanced 60-sample benchmark (30 Real, 30 Fake) for Phase 3 Evaluation...")
    real_candidates = df_test_out[df_test_out["label"] == 0].copy()
    fake_candidates = df_test_out[df_test_out["label"] == 1].copy()

    # If we need more Real samples to reach 30, draw from df_full verified True claims
    additional_needed = 30 - len(real_candidates)
    if additional_needed > 0:
        print(f"  Supplementing {additional_needed} verified True claims from the full dataset...")
        full_trues = df_full[df_full["label"].str.lower() == "true"].reset_index(drop=True)
        supp_count = 0
        for idx2, r_row in full_trues.iterrows():
            if supp_count >= additional_needed:
                break
            claim = str(r_row["claim"]).strip()
            if claim in df_test_out["text"].values:
                continue
            img_url = str(r_row["img"]).strip()
            slug = str(r_row.get("url_caption", f"fauxtography_supp_{idx2}")).strip()
            slug_clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", slug)[:60]
            dest_filename = f"supp_{idx2:04d}_{slug_clean}.jpg"
            dest_path = images_dir / dest_filename

            if download_image(img_url, dest_path, headers):
                supp_record = {
                    "id": len(processed_records),
                    "text": claim,
                    "label": 0,
                    "original_label": "true",
                    "image_path": str(dest_path.resolve()),
                    "image_filename": dest_filename,
                    "image_url": img_url,
                    "source": "Zlatkova_EMNLP_2019_Snopes_Full_True"
                }
                real_candidates = pd.concat([real_candidates, pd.DataFrame([supp_record])], ignore_index=True)
                supp_count += 1
                print(f"  [Supp {supp_count}/{additional_needed}] [OK] TRUE -> {dest_filename}")

    sample_real_30 = real_candidates.sample(n=min(30, len(real_candidates)), random_state=42)
    sample_fake_30 = fake_candidates.sample(n=min(30, len(fake_candidates)), random_state=42)
    df_balanced_60 = pd.concat([sample_real_30, sample_fake_30]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_balanced_60["id"] = range(1, len(df_balanced_60) + 1)

    balanced_path = PROJECT_ROOT / "testing" / "test_dataset_fauxtography.csv"
    df_balanced_60.to_csv(balanced_path, index=False)
    print(f"\n[DONE] Saved balanced 60-sample benchmark dataset to: {balanced_path}")
    print("Label distribution in balanced dataset:")
    print(df_balanced_60["original_label"].value_counts())
    print("\nDataset ready for RAHMAD Phase 3 testing!")

if __name__ == "__main__":
    main()
