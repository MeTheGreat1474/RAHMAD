"""
backend/analyzer.py

Entry point for the RAHMAD analysis pipeline.

analyze_text() is called by app.py on every POST to /api/analyze.

Full pipeline
─────────────
  1. Decode the frontend's base64 image (or URL) → PIL Image
  2. input_processing.preprocess_inputs()
       → llm_text      (cleaned full text for the Judge LLM)
       → CLIP tensors  (input_ids, attention_mask, pixel_values for EP)
  3. EarlyPredictor.predict_from_tensors_verbose()
       → probability, label, confidence, pass_to_agents
       → text_embeds, image_embeds, fused  (for logging)
  4. Judge.analyze(llm_text, ep_result)
       → verdict, color_class, confidence, justification, key_facts, links
  5. PipelineLogger writes output_log.txt
  6. Return Judge's structured response to the frontend (JSON).

No-image fallback
─────────────────
  If the frontend sends no image, the CLIP-based EP is skipped and the Judge
  receives a neutral EP signal so it can still produce a text-only verdict.
  The log reflects the skip.
"""

import base64
import io
import logging
import urllib.request

from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load heavy models once — not at import time so Flask starts fast.
# ---------------------------------------------------------------------------
_predictor = None
_judge     = None
_retriever = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        log.info("[analyzer] Loading Early Predictor…")
        from model.early_predictor import EarlyPredictor
        _predictor = EarlyPredictor()
        log.info("[analyzer] Early Predictor ready.")
    return _predictor


def _get_judge():
    global _judge
    if _judge is None:
        log.info("[analyzer] Loading Judge Module…")
        from model.judge import Judge
        _judge = Judge()
        log.info("[analyzer] Judge Module ready.")
    return _judge


def _get_retriever():
    global _retriever
    if _retriever is None:
        log.info("[analyzer] Loading Knowledge Retriever…")
        from model.knowledge_retrieval import KnowledgeRetriever
        _retriever = KnowledgeRetriever()
        log.info("[analyzer] Knowledge Retriever ready.")
    return _retriever


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------

def _decode_image(image_data: str) -> Image.Image | None:
    """
    Decode an image from a base64 data-URL or a plain HTTP URL
    into a PIL RGB Image.  Returns None on failure or if no data provided.
    """
    if not image_data:
        return None
    try:
        if image_data.startswith("data:"):
            raw_bytes = base64.b64decode(image_data.split(",", 1)[1])
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        elif image_data.startswith("http://") or image_data.startswith("https://"):
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
# Neutral EP signal used when no image is available
# ---------------------------------------------------------------------------
_NO_IMAGE_EP = {
    "probability":    0.5,
    "label":          "UNKNOWN",
    "confidence":     "N/A",
    "pass_to_agents": True,
    "note":           "EP skipped — no image provided; Judge will rely on text only.",
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_text(input_text: str, image_data: str = None) -> dict:
    """
    Run the full RAHMAD pipeline and return a structured verdict.

    Parameters
    ----------
    input_text : str
        Raw article text from the frontend.
    image_data : str | None
        Base64 data-URL or HTTP URL of the article image, or None.

    Returns
    -------
    dict
        verdict, color_class, confidence, justification, key_facts, links,
        ep_signal  — all ready for jsonify() in app.py.
    """
    from backend.pipeline_logger import PipelineLogger
    plog = PipelineLogger()

    # ── Guard: article too short ──────────────────────────────────────────
    if len(input_text.strip()) < 10:
        log.info("[analyzer] Input too short — returning INCONCLUSIVE.")
        plog.save()
        return {
            "verdict":       "INCONCLUSIVE",
            "color_class":   "inconclusive",
            "confidence":    "0.0%",
            "justification": (
                "The provided text is too short to analyse. "
                "Please submit a complete article."
            ),
            "key_facts":  ["Input text is too short for analysis."],
            "links":      [],
            "ep_signal":  {},
        }

    # ── 1. Decode image ───────────────────────────────────────────────────
    image = _decode_image(image_data)

    # ── Log stage 1: raw input ────────────────────────────────────────────
    plog.log_preprocessing_input(input_text, image)

    # ── 2. Pre-process inputs ─────────────────────────────────────────────
    from backend.input_processing import preprocess_inputs, clean_for_llm

    if image is not None:
        try:
            log.info("[analyzer] Running full pre-processing (text + image)…")
            processed = preprocess_inputs(input_text, image)
            llm_text  = processed["llm_text"]
            has_clip  = True
            plog.log_preprocessing_output(processed)
        except Exception as exc:
            log.warning(f"[analyzer] preprocess_inputs failed: {exc} — falling back to text-only.")
            llm_text = clean_for_llm(input_text)
            has_clip = False
            plog.log_ep_skipped(f"preprocess_inputs failed: {exc}")
    else:
        log.info("[analyzer] No image — cleaning text for LLM only.")
        llm_text = clean_for_llm(input_text)
        has_clip = False
        plog.log_ep_skipped("no image provided")

    # ── 3 & 4. Early Predictor ────────────────────────────────────────────
    ep_result = _NO_IMAGE_EP.copy()

    if has_clip:
        try:
            ep = _get_predictor()
            ep_result, text_embeds, image_embeds, fused = ep.predict_from_tensors_verbose(
                processed["input_ids"],
                processed["attention_mask"],
                processed["pixel_values"],
            )
            log.info(
                f"[analyzer] EP → label={ep_result['label']}  "
                f"p={ep_result['probability']:.4f}  "
                f"pass_to_agents={ep_result['pass_to_agents']}"
            )
            print("\n" + "=" * 50)
            print("  EARLY PREDICTOR RESULT")
            print("=" * 50)
            print(f"  Probability    : {ep_result['probability']:.4f}")
            print(f"  Label          : {ep_result['label']}")
            print(f"  Confidence     : {ep_result['confidence']}")
            print(f"  Pass to Agents : {ep_result['pass_to_agents']}")
            print("=" * 50 + "\n")

            # Log CLIP tensors and EP verdict
            plog.log_clip_output(text_embeds, image_embeds, fused)
            plog.log_ep_output(ep_result)

        except Exception as exc:
            log.error(f"[analyzer] Early Predictor failed: {exc}", exc_info=True)
            plog.log_ep_skipped(f"EarlyPredictor error: {exc}")

    # ── 4.5. Knowledge Retrieval ──────────────────────────────────────────
    external_knowledge = ""
    kr_articles = []
    try:
        retriever = _get_retriever()
        claim_query = processed["lead_text"] if (has_clip and "lead_text" in processed) else input_text[:300]
        kr_data = retriever.retrieve_detailed(claim_query)
        external_knowledge = kr_data["summary"]
        kr_articles = kr_data.get("articles", [])
        plog.log_knowledge_retrieval(kr_data)
    except Exception as exc:
        log.error(f"[analyzer] Knowledge Retrieval failed: {exc}", exc_info=True)
        plog.log_knowledge_retrieval({
            "claim": input_text[:300],
            "articles": [],
            "summary": f"Knowledge Retrieval error: {exc}"
        })

    # ── 5. Judge Module ───────────────────────────────────────────────────
    try:
        from model.judge import _SYSTEM_PROMPT   # module-level constant

        print("\n" + "=" * 50)
        print("  JUDGE MODULE (CLOUD LLM REASONING LAYER)")
        print("=" * 50)
        if external_knowledge:
            print("  [OK] Forwarding synthesized Knowledge Summary to Judge context.")
        else:
            print("  [!] No external knowledge retrieved. Proceeding with article text & EP signal.")
            
        print(f"  Sending request to OpenRouter Cloud LLM... Please wait...")
        
        judge  = _get_judge()
        result = judge.analyze(llm_text, ep_result, external_knowledge=external_knowledge)
        
        # Populate frontend links from Knowledge Retrieval articles
        if kr_articles:
            formatted_links = []
            from urllib.parse import urlparse
            for a in kr_articles:
                url = a.get("source") or a.get("url", "#")
                headline = a.get("headline") or a.get("title", url)
                domain = urlparse(url).netloc.replace("www.", "").upper() if url and url.startswith("http") else "EXTERNAL SOURCE"
                formatted_links.append({
                    "source": domain,
                    "title": headline,
                    "url": url
                })
            result["links"] = formatted_links

        # Log Judge input (prompt) and output (raw response + parsed)
        plog.log_judge_input(_SYSTEM_PROMPT, judge.last_user_message)
        plog.log_judge_output(judge.last_raw_response, result)

        log.info(
            f"[analyzer] Judge → {result['verdict']} ({result['confidence']})"
        )
        print("\n" + "=" * 50)
        print("  JUDGE MODULE RESULT")
        print("=" * 50)
        print(f"  Verdict     : {result['verdict']}")
        print(f"  Confidence  : {result['confidence']}")
        print(f"  Color Class : {result['color_class']}")
        print("=" * 50 + "\n")

        plog.save()
        return result

    except Exception as exc:
        log.error(f"[analyzer] Judge Module failed: {exc}", exc_info=True)
        plog.save()
        return {
            "verdict":       "INCONCLUSIVE",
            "color_class":   "inconclusive",
            "confidence":    "0.0%",
            "justification": f"Analysis pipeline encountered an unexpected error: {exc}",
            "key_facts":     ["Pipeline error — check server logs for details."],
            "links":         [],
            "ep_signal":     ep_result,
        }
