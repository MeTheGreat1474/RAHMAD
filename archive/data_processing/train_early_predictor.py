"""
RAHMAD - Early Predictor Training Script (DirectML Edition)
============================================================
Architecture:
  - CLIP (openai/clip-vit-base-patch32) with LoRA injected on BOTH
    vision encoder and text encoder attention layers
  - Single-layer MLP: Linear(1024 → 1) + Sigmoid
  - Binary classification: 0 = Real, 1 = Fake

AMD GPU via DirectML (WSL2 / Windows):
  - pip install torch-directml
  - Device: privateuseone:0
"""

import os
import re
import logging
import warnings
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 1. DEVICE SETUP — DirectML > CUDA > CPU
# ══════════════════════════════════════════════════════════════════

def get_device():
    # Try DirectML first (AMD GPU in WSL/Windows)
    try:
        import torch_directml
        device = torch_directml.device()
        log.info(f"Using DirectML (AMD GPU): {device}")
        return device, "directml"
    except ImportError:
        pass

    # Fall back to CUDA (NVIDIA or ROCm)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device, "cuda"

    # CPU last resort
    log.warning("No GPU found — running on CPU (will be very slow)")
    return torch.device("cpu"), "cpu"


# ══════════════════════════════════════════════════════════════════
# 2. ARGS
# ══════════════════════════════════════════════════════════════════

def get_args():
    parser = argparse.ArgumentParser(description="RAHMAD Early Predictor Training")

    # Paths
    parser.add_argument(
        "--csv_path",
        type=str,
        default=r"C:\Users\ahmad\Documents\Programming\fyp\Dataset\super_dataset.csv",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=r"C:\Users\ahmad\Documents\Programming\fyp\Dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./early_predictor_output",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g. ./early_predictor_output/best_checkpoint.pt)",
    )

    # Model
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")

    # LoRA
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-2)
    parser.add_argument("--max_text_length", type=int,   default=77)
    parser.add_argument("--num_workers",     type=int,   default=0,
                        help="Keep 0 for WSL stability")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--patience",        type=int,   default=3)

    # Dataset cap — useful for quick sanity checks before full training
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap total samples (e.g. 50000). None = use full dataset",
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════
# 3. TEXT PREPROCESSING  (CleanText — FYP report §3.3.1)
# ══════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)                      # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)        # URLs
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)                        # Emojis
    text = re.sub(r"[^\w\s.,!?']", " ", text)                 # Punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# ══════════════════════════════════════════════════════════════════
# 4. DATASET
# ══════════════════════════════════════════════════════════════════

class FakeNewsDataset(Dataset):
    def __init__(self, df, processor, dataset_root, max_length=77):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.dataset_root = Path(dataset_root)
        self.max_length = max_length
        self._placeholder = Image.new("RGB", (224, 224), color=(0, 0, 0))

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path):
        if not isinstance(rel_path, str) or rel_path.strip() == "":
            return self._placeholder
        full_path = self.dataset_root / rel_path.strip()
        try:
            return Image.open(full_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            return self._placeholder

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        text  = clean_text(str(row["text"]))
        image = self._load_image(str(row["image_path"]))
        label = float(row["label"])

        encoded = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values":   encoded["pixel_values"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.float32),
        }


# ══════════════════════════════════════════════════════════════════
# 5. MODEL
# ══════════════════════════════════════════════════════════════════

class EarlyPredictor(nn.Module):
    """
    CLIP (frozen) + LoRA on both encoders → concat → Linear(1024→1) + Sigmoid
    """
    def __init__(self, clip_model, lora_cfg):
        super().__init__()
        self.clip = get_peft_model(clip_model, lora_cfg)
        proj_dim  = clip_model.config.projection_dim   # 512
        fused_dim = proj_dim * 2                       # 1024

        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 1),
            nn.Sigmoid(),
        )

        log.info(f"EarlyPredictor ready | fused_dim={fused_dim}")
        self.clip.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, pixel_values):
        text_features  = self.clip.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        fused = torch.cat([text_features, image_features], dim=-1)
        return self.mlp(fused).squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# 6. LoRA CONFIG
# ══════════════════════════════════════════════════════════════════

def build_lora_config(r, alpha, dropout):
    return LoraConfig(
        task_type=None,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=[
            "visual_projection",
            "vision_model.encoder.layers.*.self_attn.q_proj",
            "vision_model.encoder.layers.*.self_attn.v_proj",
            "text_projection",
            "text_model.encoder.layers.*.self_attn.q_proj",
            "text_model.encoder.layers.*.self_attn.v_proj",
        ],
    )


# ══════════════════════════════════════════════════════════════════
# 7. METRICS
# ══════════════════════════════════════════════════════════════════

def compute_metrics(labels, probs, threshold=0.5):
    preds  = (np.array(probs) >= threshold).astype(int)
    labels = np.array(labels).astype(int)
    acc    = accuracy_score(labels, preds)
    f1     = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "f1": f1, "auc_roc": auc}


# ══════════════════════════════════════════════════════════════════
# 8. TRAIN / EVAL LOOPS
# ══════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, backend):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False, dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values   = batch["pixel_values"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        probs = model(input_ids, attention_mask, pixel_values)
        loss  = criterion(probs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # DirectML doesn't support .item() mid-step efficiently — sync carefully
        loss_val = loss.detach()
        if backend == "directml":
            loss_val = float(loss_val.to("cpu"))
        else:
            loss_val = loss_val.item()

        total_loss += loss_val
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

        pbar.set_postfix(loss=f"{loss_val:.4f}")

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name="val"):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    pbar = tqdm(loader, desc=f"  [{split_name}]", leave=False, dynamic_ncols=True)

    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values   = batch["pixel_values"].to(device)
        labels         = batch["label"].to(device)

        probs = model(input_ids, attention_mask, pixel_values)
        loss  = criterion(probs, labels)

        total_loss += float(loss.detach().to("cpu"))
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


# ══════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device, backend = get_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load CSV ───────────────────────────────────────────────────
    log.info(f"Loading dataset: {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    required = {"text", "label", "image_path"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    # Optional sample cap
    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
        log.info(f"Capped to {args.max_samples} samples")

    log.info(f"Total: {len(df)} | Fake: {df['label'].sum()} | Real: {(df['label']==0).sum()}")

    # ── Split 70 / 10 / 20 ────────────────────────────────────────
    train_df, temp_df = train_test_split(df,      test_size=0.30,    random_state=args.seed, stratify=df["label"])
    val_df,   test_df = train_test_split(temp_df, test_size=(2/3),   random_state=args.seed, stratify=temp_df["label"])
    log.info(f"Split → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── CLIP ───────────────────────────────────────────────────────
    log.info(f"Loading CLIP: {args.clip_model}")
    processor  = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPModel.from_pretrained(args.clip_model)

    # ── DataLoaders ───────────────────────────────────────────────
    # pin_memory must be False for DirectML
    pin = (backend == "cuda")

    train_ds = FakeNewsDataset(train_df, processor, args.dataset_root, args.max_text_length)
    val_ds   = FakeNewsDataset(val_df,   processor, args.dataset_root, args.max_text_length)
    test_ds  = FakeNewsDataset(test_df,  processor, args.dataset_root, args.max_text_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    # ── Model ─────────────────────────────────────────────────────
    lora_cfg = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model    = EarlyPredictor(clip_model, lora_cfg).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.BCELoss()

    # ── Resume from checkpoint ────────────────────────────────────
    start_epoch  = 1
    best_val_f1  = -1.0
    if args.resume and Path(args.resume).exists():
        log.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["val_f1"]
        log.info(f"Resumed at epoch {start_epoch} | best val F1 so far: {best_val_f1:.4f}")
        model.to(device)

    # ── Training loop ─────────────────────────────────────────────
    patience_counter = 0

    log.info("=" * 60)
    log.info(f"Starting training on {backend.upper()}")
    log.info("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):

        train_loss, train_m = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, backend)
        val_loss,   val_m   = evaluate(model, val_loader, criterion, device, "val")
        scheduler.step()

        log.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train → Loss: {train_loss:.4f}  Acc: {train_m['accuracy']:.4f}  F1: {train_m['f1']:.4f}  AUC: {train_m['auc_roc']:.4f}"
        )
        log.info(
            f"{'':>14}| "
            f"Val   → Loss: {val_loss:.4f}  Acc: {val_m['accuracy']:.4f}  F1: {val_m['f1']:.4f}  AUC: {val_m['auc_roc']:.4f}"
        )

        # Save best checkpoint
        if val_m["f1"] > best_val_f1:
            best_val_f1      = val_m["f1"]
            patience_counter = 0
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1":              best_val_f1,
                    "val_metrics":         val_m,
                    "args":                vars(args),
                },
                output_dir / "best_checkpoint.pt",
            )
            log.info(f"  ✓ Best model saved  (val F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            log.info(f"  No improvement — patience {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                log.info("Early stopping triggered.")
                break

        # Also save latest checkpoint every epoch so you can always resume
        torch.save(
            {
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1":              best_val_f1,
                "val_metrics":         val_m,
                "args":                vars(args),
            },
            output_dir / "latest_checkpoint.pt",
        )

    # ── Test evaluation ───────────────────────────────────────────
    log.info("=" * 60)
    log.info("Loading best checkpoint → final test evaluation")
    best_ckpt = torch.load(output_dir / "best_checkpoint.pt", map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(device)

    test_loss, test_m = evaluate(model, test_loader, criterion, device, "test")
    log.info(
        f"TEST RESULTS | Loss: {test_loss:.4f} | "
        f"Accuracy: {test_m['accuracy']:.4f} | "
        f"F1: {test_m['f1']:.4f} | "
        f"AUC-ROC: {test_m['auc_roc']:.4f}"
    )

    # Save model artifacts (DirectML workaround)
    log.info("Rebuilding model on CPU for saving...")
    from transformers import CLIPModel as _CLIPModel
    _cpu_clip = _CLIPModel.from_pretrained(args.clip_model)
    _lora_cfg2 = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    _cpu_model = EarlyPredictor(_cpu_clip, _lora_cfg2)
    _saved_ckpt = torch.load(output_dir / "best_checkpoint.pt", map_location="cpu")
    _cpu_model.load_state_dict(_saved_ckpt["model_state_dict"])
    _cpu_model.eval()
    _cpu_model.clip.save_pretrained(output_dir / "lora_adapter")
    torch.save(_cpu_model.mlp.state_dict(), output_dir / "mlp_head.pt")
    processor.save_pretrained(output_dir / "processor")
    log.info(f"Artifacts saved to {output_dir}")
    log.info("Training complete.")


if __name__ == "__main__":
    main()
