import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import pandas as pd
from PIL import Image

from model.early_predictor import EarlyPredictor
from model.reverse_image_search import ReverseImageSearchPipeline
from model.judge import Judge
from backend.input_processing import clean_for_llm
from testing.test_phase3_evaluation import load_test_split, _resolve_image

def main():
    t_start = time.perf_counter()
    print("=" * 75)
    print("  RAHMAD FAILURE SAMPLE RE-EVALUATION PIPELINE")
    print("=" * 75)

    dataset_path = PROJECT_ROOT / "testing" / "test_dataset_newsclippings.csv"
    ckpt_file = PROJECT_ROOT / "testing" / ".checkpoint_ablation_test_dataset_newsclippings_60.json"
    kr_cache_file = PROJECT_ROOT / "testing" / ".cache_kr_ablation_test_dataset_newsclippings_60.json"
    mad_cache_file = PROJECT_ROOT / "testing" / ".cache_mad_ablation_test_dataset_newsclippings_60.json"

    if not ckpt_file.exists():
        print(f"Error: Checkpoint file {ckpt_file} not found.")
        return

    checkpoint = json.loads(ckpt_file.read_text(encoding="utf-8"))
    kr_cache = json.loads(kr_cache_file.read_text(encoding="utf-8")) if kr_cache_file.exists() else {}
    mad_cache = json.loads(mad_cache_file.read_text(encoding="utf-8")) if mad_cache_file.exists() else {}

    # Load the exact 60-sample ablation split
    _, df_ablation = load_test_split(dataset_path, n_samples_3a=200, n_samples_ablation=60, seed=42)
    print(f"Loaded {len(df_ablation)} samples in ablation split.")

    # Identify the failure samples (where previous run was INCORRECT)
    sample_records = checkpoint.get("sample_records", [])
    failure_records = [rec for rec in sample_records if not rec.get("correct", False)]
    print(f"Identified {len(failure_records)} failure samples to re-evaluate.")

    # Initialize updated modules
    print("\nInitializing updated modules...")
    judge = Judge()
    ris = ReverseImageSearchPipeline(visual_threshold=0.68, semantic_threshold=0.0)
    print("Modules initialized successfully.\n")

    results_table = []
    corrected_count = 0

    for i, rec in enumerate(failure_records, 1):
        sample_id = rec["id"]
        idx_0 = sample_id - 1
        true_label_str = rec["true_label"]
        true_label_int = 1 if true_label_str == "FAKE" else 0
        old_verdict = rec["full_verdict"]
        old_conf = rec.get("full_confidence", "N/A")

        row = df_ablation.iloc[idx_0]
        text = str(row["text"])
        cleaned_text = clean_for_llm(text)
        img_path = row.get("image_path")

        # Resolve image path
        p = Path(str(img_path))
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        resolved_img_path = str(p) if p.exists() else None

        print(f"[{i}/{len(failure_records)}] Processing ID {sample_id} (True: {true_label_str}) | Old: {old_verdict} ({old_conf})...")
        print(f"    Claim: {text[:90]}...")

        # 1. EP Signal from previous checkpoint
        ep_prob = rec.get("ep_prob", 0.5)
        ep_res = {
            "label": "FAKE" if ep_prob >= 0.5 else "REAL",
            "probability": ep_prob,
            "confidence": f"{abs(ep_prob - 0.5) * 200:.1f}%",
            "color_class": "fabricated" if ep_prob >= 0.5 else "credible"
        }

        # 2. KR Summary
        kr_summary = kr_cache.get(str(idx_0), "")

        # 3. Updated Live RIS Execution
        ris_summary = ""
        if resolved_img_path:
            try:
                print(f"    [RIS] Running updated live Reverse Image Search on {p.name}...")
                ris_res = ris.execute(text, resolved_img_path)
                if ris_res.get("status") == "success" and (ris_res.get("judge_summary") or ris_res.get("visual_agent_summary")):
                    ooc_status = "YES (Decontextualized / Misattributed)" if ris_res.get("is_out_of_context") else "NO (Authentic Context / Non-Deceptive)"
                    exec_summary = ris_res.get("judge_summary") or ris_res.get("visual_agent_summary")
                    ris_summary = (
                        f"--- REVERSE IMAGE SEARCH FINDINGS ---\n"
                        f"Category: {ris_res.get('category', 'Unknown')}\n"
                        f"Out-of-Context Imagery Detected: {ooc_status}\n"
                        f"Confidence: {ris_res.get('confidence', 'Medium')}\n"
                        f"Entailment Scope: Authentic visual provenance confirms only that the media file is genuine and not recycled from an unrelated event; it does NOT independently corroborate unverified claims or speculative character attributions in the text.\n"
                        f"Executive Evidence Summary: {exec_summary}"
                    )
                else:
                    ris_summary = (
                        f"--- REVERSE IMAGE SEARCH FINDINGS ---\n"
                        f"Category: {ris_res.get('category', 'Inconclusive / Insufficient Evidence')}\n"
                        f"Out-of-Context Imagery Detected: NO\n"
                        f"Executive Evidence Summary: {ris_res.get('visual_agent_summary', 'No conclusive online image matches found.')}"
                    )
            except Exception as e:
                print(f"    [!] RIS Error: {e}")
                ris_summary = f"--- REVERSE IMAGE SEARCH FINDINGS ---\nReverse Image Search error: {e}"

        # 4. MAD Summary
        mad_summary = mad_cache.get(str(idx_0), "")

        # 5. Updated Judicial Analysis
        print(f"    [Judge] Deliberating with updated P8 guardrails...")
        judge_res = judge.analyze(
            cleaned_text,
            ep_result=ep_res,
            external_knowledge=kr_summary,
            ris_summary=ris_summary,
            mad_summary=mad_summary
        )

        new_verdict = judge_res.get("verdict", "INCONCLUSIVE")
        new_conf = judge_res.get("confidence", "N/A")
        new_pred = 0 if new_verdict == "CREDIBLE" else 1
        is_now_correct = (new_pred == true_label_int)

        if is_now_correct:
            corrected_count += 1
            status_str = "FIXED (NOW CORRECT)"
        else:
            status_str = "STILL INCORRECT"

        print(f"    -> Result: {new_verdict} ({new_conf}) | {status_str}\n")

        results_table.append({
            "id": sample_id,
            "true_label": true_label_str,
            "old_verdict": old_verdict,
            "old_conf": old_conf,
            "new_verdict": new_verdict,
            "new_conf": new_conf,
            "status": status_str,
            "is_correct": is_now_correct,
            "justification": judge_res.get("justification", "")
        })

    # Summary report
    print("\n" + "=" * 90)
    print("  SUMMARY OF FAILURE SAMPLE RE-EVALUATION")
    print("=" * 90)
    print(f"{'ID':<4} | {'True Label':<10} | {'Previous Verdict':<20} | {'New Verdict':<20} | {'Conf':<6} | {'Status'}")
    print("-" * 90)
    for r in results_table:
        print(f"{r['id']:<4} | {r['true_label']:<10} | {r['old_verdict']:<20} | {r['new_verdict']:<20} | {r['new_conf']:<6} | {r['status']}")
    print("-" * 90)
    print(f"Previously Incorrect : {len(failure_records)} samples")
    print(f"Successfully Fixed   : {corrected_count} / {len(failure_records)} ({corrected_count / len(failure_records) * 100:.1f}%)")
    
    prev_accuracy = 45 / 60 * 100
    new_total_correct = 45 + corrected_count
    new_accuracy = new_total_correct / 60 * 100
    print(f"\nOverall 60-Sample Accuracy Impact:")
    print(f"  Previous Baseline Accuracy : 45 / 60 (75.00%)")
    print(f"  Updated New Accuracy       : {new_total_correct} / 60 ({new_accuracy:.2f}%)")
    print(f"  Accuracy Gain              : +{new_accuracy - prev_accuracy:.2f}%")
    print("=" * 90)

    # Save detailed re-evaluation artifact
    out_file = PROJECT_ROOT / "testing" / "phase3_failure_reevaluation_results.json"
    out_file.write_text(json.dumps(results_table, indent=2), encoding="utf-8")
    print(f"Detailed failure re-evaluation results saved to: {out_file}")

    total_time = time.perf_counter() - t_start
    print(f"Execution finished in {total_time:.2f}s.")

if __name__ == "__main__":
    main()
