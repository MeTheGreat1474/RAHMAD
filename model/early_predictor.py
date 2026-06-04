"""
model/early_predictor.py

Early Predictor — Inference Module for RAHMAD.

Architecture (mirrors train_early_predictor.py exactly):
  CLIP (openai/clip-vit-base-patch32)
    + LoRA adapters on both vision & text encoder attention layers
  → concat text_embeds [512] + image_embeds [512] = fused [1024]
  → Linear(1024 → 1) + Sigmoid
  → probability ∈ [0, 1]   (0 = Real, 1 = Fake)

Saved artifacts expected at:
  data_processing/early_predictor_output/
    lora_adapter/          ← CLIP + LoRA weights (PEFT format)
    mlp_head.pt            ← MLP Sequential state dict
    processor/             ← CLIPProcessor (tokeniser + image processor)

Usage:
    from model.early_predictor import EarlyPredictor

    ep = EarlyPredictor()                   # loads artifacts, auto-selects device
    result = ep.predict(raw_text, raw_image)

    result["probability"]   # float  e.g. 0.87
    result["label"]         # str    "FAKE" | "REAL"
    result["confidence"]    # str    e.g. "87.0%"
    result["pass_to_agents"]  # bool — True if uncertain (triggers MAD pipeline)
"""

import re
import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  (resolved relative to this file → project root stays portable)
# ---------------------------------------------------------------------------
_PROJECT_ROOT  = Path(__file__).resolve().parent.parent   # RAHMAD/
_ARTIFACTS_DIR = _PROJECT_ROOT / "data_processing" / "early_predictor_output"

LORA_ADAPTER_PATH = _ARTIFACTS_DIR / "lora_adapter"
MLP_HEAD_PATH     = _ARTIFACTS_DIR / "mlp_head.pt"
PROCESSOR_PATH    = _ARTIFACTS_DIR / "processor"

# ---------------------------------------------------------------------------
# Model constants (must match training config)
# ---------------------------------------------------------------------------
_BASE_CLIP_MODEL = "openai/clip-vit-base-patch32"
_PROJ_DIM        = 512    # CLIP projection dim for vit-base-patch32
_FUSED_DIM       = _PROJ_DIM * 2   # 1024
_MAX_TEXT_LEN    = 77

# ---------------------------------------------------------------------------
# Inference decision thresholds
# ---------------------------------------------------------------------------
FAKE_THRESHOLD      = 0.60   # p >= 0.60  → FAKE (high confidence)
REAL_THRESHOLD      = 0.40   # p <= 0.40  → REAL (high confidence)
# 0.40 < p < 0.60  → UNCERTAIN → escalate to Multi-Agent Debate


# ===========================================================================
# 1. TEXT CLEANING  (must match train_early_predictor.py clean_text())
# ===========================================================================

def _clean_text_for_clip(text: str) -> str:
    """
    Regex-based text cleaner — identical to the one used during training so
    that inference inputs match the distribution the model was trained on.

    Removes HTML tags, URLs, emojis, and non-standard punctuation; collapses
    whitespace; lowercases.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove emojis
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
    text = emoji_pattern.sub(" ", text)
    # Keep only word chars, spaces, and basic punctuation
    text = re.sub(r"[^\w\s.,!?']", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# ===========================================================================
# 2. MLP HEAD  (identical to training)
# ===========================================================================

class _MLPHead(nn.Module):
    """Single-layer MLP: Linear(1024 → 1) + Sigmoid."""

    def __init__(self, input_dim: int = _FUSED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ===========================================================================
# 3. EARLY PREDICTOR  (inference wrapper)
# ===========================================================================

class EarlyPredictor:
    """
    Loads the trained CLIP+LoRA+MLP checkpoint and runs fake-news inference.

    The model is loaded once at construction time and reused for every call
    to predict(), making it efficient when processing multiple articles.

    Parameters
    ----------
    device : str | None
        Force a specific device ("cpu", "cuda", "privateuseone:0" for
        DirectML).  Defaults to auto-detection: DirectML → CUDA → CPU.
    fake_threshold : float
        Probability above which the article is labelled FAKE (default 0.60).
    real_threshold : float
        Probability below which the article is labelled REAL (default 0.40).
        Anything between the two thresholds is UNCERTAIN.
    """

    def __init__(
        self,
        device: str | None = None,
        fake_threshold: float = FAKE_THRESHOLD,
        real_threshold: float = REAL_THRESHOLD,
    ):
        self.fake_threshold = fake_threshold
        self.real_threshold = real_threshold
        self.device = self._resolve_device(device)

        log.info(f"[EarlyPredictor] Initialising on device: {self.device}")
        self._validate_artifacts()

        self.processor  = self._load_processor()
        self.clip       = self._load_clip_with_lora()
        self.mlp        = self._load_mlp()

        log.info("[EarlyPredictor] Ready.")

    # ------------------------------------------------------------------
    # Private — setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(requested: str | None) -> torch.device:
        if requested is not None:
            return torch.device(requested)
        # Try DirectML (AMD GPU on Windows / WSL2)
        try:
            import torch_directml
            dev = torch_directml.device()
            log.info("[EarlyPredictor] Using DirectML (AMD GPU).")
            return dev
        except ImportError:
            pass
        if torch.cuda.is_available():
            log.info("[EarlyPredictor] Using CUDA GPU.")
            return torch.device("cuda")
        log.info("[EarlyPredictor] Using CPU.")
        return torch.device("cpu")

    @staticmethod
    def _validate_artifacts():
        missing = []
        for path in (LORA_ADAPTER_PATH, MLP_HEAD_PATH, PROCESSOR_PATH):
            if not path.exists():
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(
                "[EarlyPredictor] Missing trained artifact(s):\n"
                + "\n".join(f"  • {p}" for p in missing)
                + "\nRun train_early_predictor.py first to generate these files."
            )

    def _load_processor(self) -> CLIPProcessor:
        log.info(f"[EarlyPredictor] Loading processor from: {PROCESSOR_PATH}")
        return CLIPProcessor.from_pretrained(str(PROCESSOR_PATH))

    def _load_clip_with_lora(self) -> PeftModel:
        log.info(f"[EarlyPredictor] Loading base CLIP: {_BASE_CLIP_MODEL}")
        base = CLIPModel.from_pretrained(_BASE_CLIP_MODEL)

        log.info(f"[EarlyPredictor] Applying LoRA adapter: {LORA_ADAPTER_PATH}")
        peft_model = PeftModel.from_pretrained(
            base,
            str(LORA_ADAPTER_PATH),
            is_trainable=False,
        )
        peft_model.eval()
        peft_model.to(self.device)
        return peft_model

    def _load_mlp(self) -> _MLPHead:
        log.info(f"[EarlyPredictor] Loading MLP head: {MLP_HEAD_PATH}")
        mlp = _MLPHead(input_dim=_FUSED_DIM)
        state = torch.load(MLP_HEAD_PATH, map_location="cpu", weights_only=True)
        # The training script saved mlp.state_dict() where mlp is an nn.Sequential,
        # so keys are "0.weight" and "0.bias" — remap to our "net.0.*" namespace.
        remapped = {f"net.{k}": v for k, v in state.items()}
        mlp.load_state_dict(remapped)
        mlp.eval()
        mlp.to(self.device)
        return mlp

    # ------------------------------------------------------------------
    # Private — inference helpers
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self,
        raw_text: str,
        raw_image: Image.Image,
    ) -> dict[str, torch.Tensor]:
        """
        Clean text (training-consistent) and run CLIPProcessor to get tensors.
        """
        cleaned = _clean_text_for_clip(raw_text)

        encoded = self.processor(
            text=cleaned,
            images=raw_image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=_MAX_TEXT_LEN,
        )
        return {
            "input_ids":      encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "pixel_values":   encoded["pixel_values"].to(self.device),
        }

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> float:
        """
        Run CLIP encoders → concatenate → MLP → scalar probability.
        """
        prob, _, _, _ = self._forward_verbose(input_ids, attention_mask, pixel_values)
        return prob

    def _forward_verbose(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as _forward() but also returns the intermediate CLIP tensors
        for logging and inspection.

        Returns
        -------
        tuple:
            probability  : float          — scalar ∈ [0, 1]
            text_embeds  : torch.Tensor   — [1, 512]
            image_embeds : torch.Tensor   — [1, 512]
            fused        : torch.Tensor   — [1, 1024]
        """
        with torch.no_grad():
            outputs = self.clip(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            text_embeds  = outputs.text_embeds    # [1, 512]
            image_embeds = outputs.image_embeds   # [1, 512]

        fused = torch.cat([text_embeds, image_embeds], dim=-1)  # [1, 1024]
        prob  = self.mlp(fused)                                  # [1] or scalar

        return float(prob.squeeze().cpu()), text_embeds, image_embeds, fused

    @staticmethod
    def _build_verdict(probability: float, fake_thr: float, real_thr: float) -> dict:
        """Map a raw probability to a structured verdict dict."""
        confidence_pct = f"{probability * 100:.1f}%"

        if probability >= fake_thr:
            label       = "FAKE"
            color_class = "manipulated"
            pass_to_agents = False
        elif probability <= real_thr:
            label       = "REAL"
            color_class = "real"
            pass_to_agents = False
        else:
            label       = "UNCERTAIN"
            color_class = "inconclusive"
            pass_to_agents = True   # escalate to Multi-Agent Debate

        return {
            "probability":    probability,
            "label":          label,
            "confidence":     confidence_pct,
            "color_class":    color_class,
            "pass_to_agents": pass_to_agents,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        raw_text: str,
        raw_image: Image.Image,
    ) -> dict:
        """
        Run end-to-end inference on a news article.

        Parameters
        ----------
        raw_text : str
            Raw article text (from input_retrieval.load_text()).
        raw_image : PIL.Image.Image
            Raw article image in RGB mode (from input_retrieval.load_image()).

        Returns
        -------
        dict
            probability    : float   — raw score ∈ [0, 1]  (1 = fake)
            label          : str     — "FAKE" | "REAL" | "UNCERTAIN"
            confidence     : str     — formatted percentage
            color_class    : str     — "manipulated" | "real" | "inconclusive"
            pass_to_agents : bool    — True when UNCERTAIN; triggers MAD pipeline
        """
        log.info("[EarlyPredictor] Preparing inputs…")
        inputs = self._prepare_inputs(raw_text, raw_image)

        log.info("[EarlyPredictor] Running forward pass…")
        probability = self._forward(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
        )

        result = self._build_verdict(probability, self.fake_threshold, self.real_threshold)

        log.info(
            f"[EarlyPredictor] Result → p={probability:.4f}  "
            f"label={result['label']}  pass_to_agents={result['pass_to_agents']}"
        )
        return result

    def predict_from_tensors(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> dict:
        """
        Run inference on already-processed tensors (from input_processing.py).

        Use this variant when you have already called
        backend.input_processing.preprocess_inputs() and want to avoid
        running the processor a second time.

        Note: tensors must have been produced by CLIPProcessor (or equivalent
        CLIPTokenizer + CLIPImageProcessor) with the same model variant.

        Parameters
        ----------
        input_ids      : torch.Tensor  [1, 77]
        attention_mask : torch.Tensor  [1, 77]
        pixel_values   : torch.Tensor  [1, 3, 224, 224]

        Returns
        -------
        dict  (same structure as predict())
        """
        input_ids      = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        pixel_values   = pixel_values.to(self.device)

        probability = self._forward(input_ids, attention_mask, pixel_values)
        return self._build_verdict(probability, self.fake_threshold, self.real_threshold)

    def predict_from_tensors_verbose(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as predict_from_tensors() but also returns the raw CLIP embeddings
        for pipeline logging.

        Parameters
        ----------
        input_ids      : torch.Tensor  [1, 77]
        attention_mask : torch.Tensor  [1, 77]
        pixel_values   : torch.Tensor  [1, 3, 224, 224]

        Returns
        -------
        tuple:
            result       : dict          — same as predict_from_tensors()
            text_embeds  : torch.Tensor  — [1, 512]  CLIP text encoder output
            image_embeds : torch.Tensor  — [1, 512]  CLIP image encoder output
            fused        : torch.Tensor  — [1, 1024] concatenated embedding
        """
        input_ids      = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        pixel_values   = pixel_values.to(self.device)

        probability, text_embeds, image_embeds, fused = self._forward_verbose(
            input_ids, attention_mask, pixel_values
        )
        result = self._build_verdict(probability, self.fake_threshold, self.real_threshold)
        return result, text_embeds, image_embeds, fused


# ---------------------------------------------------------------------------
# Quick self-test — run directly:  python -m model.early_predictor
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from backend.input_retrieval import load_inputs

    print("Loading inputs from res/…")
    raw_text, raw_image = load_inputs()

    print("Initialising Early Predictor…")
    ep = EarlyPredictor()

    print("Running prediction…")
    result = ep.predict(raw_text, raw_image)

    print("\n=== Early Predictor Output ===")
    print(f"  Probability    : {result['probability']:.4f}")
    print(f"  Label          : {result['label']}")
    print(f"  Confidence     : {result['confidence']}")
    print(f"  Pass to Agents : {result['pass_to_agents']}")
