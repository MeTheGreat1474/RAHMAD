"""
backend/analyzer.py

Entry point for the RAHMAD analysis pipeline.

analyze_text() is called by app.py on every POST to /api/analyze.
It decodes the frontend's base64 image, runs the Early Predictor, and
prints the result to the terminal.  The frontend still receives a mock
response for now — the real verdict will be wired up in a later step.
"""

import base64
import io
import logging
import urllib.request

from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load the Early Predictor once — not at import time so Flask starts
# fast; the model loads on the very first request instead.
# ---------------------------------------------------------------------------
_predictor = None

def _get_predictor():
    global _predictor
    if _predictor is None:
        log.info("[analyzer] Loading Early Predictor for the first time…")
        from model.early_predictor import EarlyPredictor
        _predictor = EarlyPredictor()
        log.info("[analyzer] Early Predictor ready.")
    return _predictor


# ---------------------------------------------------------------------------
# Image decoding — frontend sends a base64 data URL, e.g.:
#   "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
# ---------------------------------------------------------------------------
def _decode_image(image_data: str) -> Image.Image | None:
    """
    Decode an image from either a base64 data-URL or a plain HTTP URL
    into a PIL RGB Image.  Returns None on failure.
    """
    if not image_data:
        return None
    try:
        if image_data.startswith("data:"):
            # Base64 data-URL: "data:<mime>;base64,<data>"
            raw_bytes = base64.b64decode(image_data.split(",", 1)[1])
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        elif image_data.startswith("http://") or image_data.startswith("https://"):
            # Plain URL (e.g. image already served by Flask from /res/)
            with urllib.request.urlopen(image_data) as resp:
                raw_bytes = resp.read()
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        else:
            log.warning("[analyzer] Unrecognised image format — skipping.")
            return None

        log.info(f"[analyzer] Image decoded — size={image.size}, mode={image.mode}")
        return image

    except Exception as exc:
        log.warning(f"[analyzer] Could not decode image: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main entry point called by app.py
# ---------------------------------------------------------------------------
def analyze_text(input_text: str, image_data: str = None) -> dict:
    """
    Receive raw text and optional base64 image from the frontend,
    run the Early Predictor, and print the result to the terminal.

    The frontend mock response is returned unchanged for now.
    """

    # ── 1. Decode image ───────────────────────────────────────────────────
    image = _decode_image(image_data)

    # ── 2. Run Early Predictor ────────────────────────────────────────────
    if len(input_text.strip()) < 10:
        print("\n[analyzer] Input too short — skipping Early Predictor.")
    elif image is None:
        print("\n[analyzer] No image provided — skipping Early Predictor.")
    else:
        try:
            ep     = _get_predictor()
            result = ep.predict(input_text, image)

            print("\n" + "=" * 50)
            print("  EARLY PREDICTOR RESULT")
            print("=" * 50)
            print(f"  Probability    : {result['probability']:.4f}")
            print(f"  Label          : {result['label']}")
            print(f"  Confidence     : {result['confidence']}")
            print(f"  Pass to Agents : {result['pass_to_agents']}")
            print("=" * 50 + "\n")

        except Exception as exc:
            log.error(f"[analyzer] Early Predictor failed: {exc}", exc_info=True)

    # ── 3. Return mock response to frontend (unchanged for now) ───────────
    if len(input_text) < 50:
        return {
            "verdict": "INCONCLUSIVE",
            "color_class": "inconclusive",
            "confidence": "45.0%",
            "justification": (
                "The provided text is too short to accurately assess its "
                "veracity using the multi-agent debate framework."
            ),
            "key_facts": ["Text length is insufficient for contextual analysis."],
            "links": [],
        }

    return {
        "verdict": "MANIPULATED CONTENT",
        "color_class": "manipulated",
        "confidence": "98.4%",
        "justification": (
            "The text was processed by the Python backend. This demonstrates "
            "that the LLM engine successfully received the input and returned "
            "a verified mock response."
        ),
        "key_facts": [
            "Input Received: The Flask backend successfully intercepted the POST request.",
            f"Image Status: {'Image decoded successfully' if image else 'No image provided'}.",
            "Early Predictor: Running — check terminal for result.",
            "Dynamic Output: This JSON payload was constructed dynamically.",
        ],
        "links": [
            {
                "source": "Python Backend",
                "title": f"Analyzed input starting with: '{input_text[:40]}…'",
            },
        ],
    }
