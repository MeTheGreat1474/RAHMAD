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
import os
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

_ris_pipeline = None
def _get_ris_pipeline():
    global _ris_pipeline
    if _ris_pipeline is None:
        log.info("[analyzer] Loading Reverse Image Search Pipeline…")
        from model.reverse_image_search import ReverseImageSearchPipeline
        _ris_pipeline = ReverseImageSearchPipeline()
        log.info("[analyzer] Reverse Image Search Pipeline ready.")
    return _ris_pipeline

_mad_orchestrator = None
def _get_mad_orchestrator():
    global _mad_orchestrator
    if _mad_orchestrator is None:
        log.info("[analyzer] Loading MAD Orchestrator…")
        from model.mad_orchestrator import OrchestratorAgent
        _mad_orchestrator = OrchestratorAgent()
        log.info("[analyzer] MAD Orchestrator ready.")
    return _mad_orchestrator


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
    "note":           "EP failed — no image provided; Judge will rely on text only.",
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
    import time
    t_pipeline_start = time.perf_counter()

    from backend.pipeline_logger import PipelineLogger
    plog = PipelineLogger()

    # ── Guard: article too short ──────────────────────────────────────────
    if len(input_text.strip()) < 10:
        log.info("[analyzer] Input too short — returning INCONCLUSIVE.")
        plog.save()
        yield {"type": "result", "data": {
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
}
    yield {"type": "status", "step": "PRE-PROCESSING INPUT...", "status": "IN PROGRESS"}
    # ── 1. Decode image & 2. Pre-process inputs ────────────────────────────
    t_prep_start = time.perf_counter()
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
            prep_runtime = time.perf_counter() - t_prep_start
            plog.log_preprocessing_output(processed, runtime=prep_runtime)
            print(f"  [Step 1 & 2] Input Pre-processing completed in {prep_runtime:.2f}s")
        except Exception as exc:
            log.warning(f"[analyzer] preprocess_inputs failed: {exc} — falling back to text-only.")
            llm_text = clean_for_llm(input_text)
            has_clip = False
            prep_runtime = time.perf_counter() - t_prep_start
            plog.log_ep_skipped(f"preprocess_inputs failed: {exc}", runtime=prep_runtime)
            print(f"  [Step 1 & 2] Input Pre-processing failed in {prep_runtime:.2f}s — falling back to text-only.")
    else:
        log.info("[analyzer] No image — cleaning text for LLM only.")
        llm_text = clean_for_llm(input_text)
        has_clip = False
        prep_runtime = time.perf_counter() - t_prep_start
        plog.log_ep_skipped("no image provided", runtime=prep_runtime)
        print(f"  [Step 1 & 2] Text Pre-processing completed in {prep_runtime:.2f}s (No image provided)")
    yield {"type": "status", "step": "PRE-PROCESSING INPUT...", "status": "OK"}

    yield {"type": "status", "step": "RUNNING EARLY PREDICTOR...", "status": "IN PROGRESS"}
    # ── 3 & 4. Early Predictor ────────────────────────────────────────────
    t_ep_start = time.perf_counter()
    ep_result = _NO_IMAGE_EP.copy()

    if has_clip:
        try:
            ep = _get_predictor()
            ep_result, text_embeds, image_embeds, fused = ep.predict_from_tensors_verbose(
                processed["input_ids"],
                processed["attention_mask"],
                processed["pixel_values"],
            )
            ep_runtime = time.perf_counter() - t_ep_start
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
            print(f"  Runtime        : {ep_runtime:.2f}s")
            print("=" * 50 + "\n")

            # Log CLIP tensors and EP verdict
            plog.log_clip_output(text_embeds, image_embeds, fused)
            plog.log_ep_output(ep_result, runtime=ep_runtime)
            yield {"type": "status", "step": "RUNNING EARLY PREDICTOR...", "status": "OK"}

        except Exception as exc:
            ep_runtime = time.perf_counter() - t_ep_start
            log.error(f"[analyzer] Early Predictor failed: {exc}", exc_info=True)
            plog.log_ep_skipped(f"EarlyPredictor error: {exc}", runtime=ep_runtime)
            print(f"  [Step 3 & 4] Early Predictor failed in {ep_runtime:.2f}s: {exc}")
            yield {"type": "status", "step": "RUNNING EARLY PREDICTOR...", "status": "FAIL"}
    else:
        ep_runtime = time.perf_counter() - t_ep_start
        print(f"  [Step 3 & 4] Early Predictor failed: No image provided (Runtime: {ep_runtime:.2f}s)")
        yield {"type": "status", "step": "RUNNING EARLY PREDICTOR...", "status": "FAIL"}

    yield {"type": "status", "step": "FETCHING EVIDENCE...", "status": "IN PROGRESS"}
    # ── 4.5. Knowledge Retrieval ──────────────────────────────────────────
    t_kr_start = time.perf_counter()
    external_knowledge = ""
    kr_articles = []
    try:
        retriever = _get_retriever()
        # Use input_text[:800] to preserve headline on line 1 for search,
        # and provide substantive claim context for semantic verification
        claim_query = input_text[:800].strip()
        kr_data = retriever.retrieve_detailed(claim_query)
        external_knowledge = kr_data["summary"]
        kr_articles = kr_data.get("articles", [])
        kr_runtime = time.perf_counter() - t_kr_start
        plog.log_knowledge_retrieval(kr_data, runtime=kr_runtime)
        print(f"  [Step 4.5] Knowledge Retrieval completed in {kr_runtime:.2f}s ({len(kr_articles)} article(s) retrieved)")
        yield {"type": "status", "step": "FETCHING EVIDENCE...", "status": "OK"}
    except Exception as exc:
        kr_runtime = time.perf_counter() - t_kr_start
        log.error(f"[analyzer] Knowledge Retrieval failed: {exc}", exc_info=True)
        plog.log_knowledge_retrieval({
            "claim": input_text[:300],
            "articles": [],
            "summary": f"Knowledge Retrieval error: {exc}"
        }, runtime=kr_runtime)
        print(f"  [Step 4.5] Knowledge Retrieval failed in {kr_runtime:.2f}s: {exc}")
        yield {"type": "status", "step": "FETCHING EVIDENCE...", "status": "FAIL"}

    yield {"type": "status", "step": "SEARCHING FOR SIMILAR IMAGES...", "status": "IN PROGRESS"}
    # ── 4.6. Reverse Image Search (RIS) ───────────────────────────────────
    t_ris_start = time.perf_counter()
    ris_summary = ""
    ris_data_for_log = None
    if has_clip and image_data and (image_data.startswith("http://") or image_data.startswith("https://") or image_data.startswith("data:")):
        try:
            print("  [Step 4.6] Initializing Reverse Image Search pipeline...")
            ris_pipeline = _get_ris_pipeline()
            ris_data = ris_pipeline.execute(input_text, image_data)
            ris_data_for_log = ris_data
            if ris_data["status"] == "success" and (ris_data.get("judge_summary") or ris_data.get("visual_agent_summary")):
                ooc_status = "YES (Decontextualized / Misattributed)" if ris_data.get("is_out_of_context") else "NO (Authentic Context / Non-Deceptive)"
                exec_summary = ris_data.get("judge_summary") or ris_data.get("visual_agent_summary")
                ris_summary = (
                    f"\n\n--- REVERSE IMAGE SEARCH FINDINGS ---\n"
                    f"Category: {ris_data['category']}\n"
                    f"Out-of-Context Imagery Detected: {ooc_status}\n"
                    f"Confidence: {ris_data.get('confidence', 'Medium')}\n"
                    f"Executive Evidence Summary for Judge:\n{exec_summary}"
                )
            ris_runtime = time.perf_counter() - t_ris_start
            print(f"  [Step 4.6] Reverse Image Search completed in {ris_runtime:.2f}s")
            yield {"type": "status", "step": "SEARCHING FOR SIMILAR IMAGES...", "status": "OK"}
        except Exception as exc:
            ris_runtime = time.perf_counter() - t_ris_start
            log.error(f"[analyzer] Reverse Image Search failed: {exc}", exc_info=True)
            ris_summary = f"\n\n--- REVERSE IMAGE SEARCH FINDINGS ---\nRIS encountered an error: {exc}"
            ris_data_for_log = {"status": "failed", "visual_agent_summary": f"RIS encountered an error: {exc}"}
            print(f"  [X] RIS Pipeline Error in {ris_runtime:.2f}s: {exc}")
            yield {"type": "status", "step": "SEARCHING FOR SIMILAR IMAGES...", "status": "FAIL"}
    else:
        ris_runtime = time.perf_counter() - t_ris_start
        print(f"  [Step 4.6] Reverse Image Search failed: No image provided (Runtime: {ris_runtime:.2f}s)")
        yield {"type": "status", "step": "SEARCHING FOR SIMILAR IMAGES...", "status": "FAIL"}
        ris_summary = "\n\n--- REVERSE IMAGE SEARCH FINDINGS ---\nReverse Image Search failed: No image provided."
        ris_data_for_log = {"status": "failed", "visual_agent_summary": "Reverse Image Search failed: No image provided."}
        try:
            ris_pipeline = _get_ris_pipeline()
            ris_pipeline.log_skipped(input_text, reason="No image provided with news query.")
        except Exception as exc:
            log.warning(f"[analyzer] Could not log skipped RIS: {exc}")
            
    # Log RIS output
    plog.log_ris_output(ris_data_for_log, runtime=ris_runtime)

    yield {"type": "status", "step": "DEBATING...", "status": "IN PROGRESS"}
    # ── 4.7. Multi-Agent Debate (MAD) ─────────────────────────────────────
    t_mad_start = time.perf_counter()
    mad_summary = ""
    agents_info = []
    try:
        print("  [Step 4.7] Initializing Multi-Agent Debate (MAD) locally...")
        orchestrator = _get_mad_orchestrator()
        mad_summary, agents_info = orchestrator.run_debate(input_text, external_knowledge)
        mad_runtime = time.perf_counter() - t_mad_start
        print(f"  [Step 4.7] Multi-Agent Debate completed in {mad_runtime:.2f}s")
        yield {"type": "status", "step": "DEBATING...", "status": "OK"}
    except Exception as exc:
        mad_runtime = time.perf_counter() - t_mad_start
        log.error(f"[analyzer] MAD failed: {exc}", exc_info=True)
        mad_summary = f"MAD encountered an error: {exc}"
        print(f"  [X] MAD Pipeline Error in {mad_runtime:.2f}s: {exc}")
        yield {"type": "status", "step": "DEBATING...", "status": "FAIL"}
        
    plog.log_mad_output(mad_summary, agents_info, runtime=mad_runtime)

    yield {"type": "status", "step": "SYNTHESIZING RESULTS...", "status": "IN PROGRESS"}
    # ── 5. Judge Module ───────────────────────────────────────────────────
    t_judge_start = time.perf_counter()
    try:
        from model.judge import _SYSTEM_PROMPT, _CHIEF_JUDGE_PROMPT   # module-level constants

        judge_mode = os.environ.get("JUDGE_MODE", "single").strip().lower()
        mode_labels = {
            "panel":            "Multi-Judge Panel (Specialized Personas + Chief Synthesizer)",
            "self_consistency": "Self-Consistency Majority Vote (n=3)",
            "single":           "Standard Single-Pass Reasoning",
        }
        mode_label = mode_labels.get(judge_mode, f"Custom Mode ({judge_mode})")

        print("\n" + "=" * 50)
        print("  JUDGE MODULE (CLOUD LLM REASONING LAYER)")
        print(f"  Architecture Pipeline : {mode_label}")
        print("=" * 50)
        if external_knowledge or ris_summary or mad_summary:
            print("  [OK] Forwarding synthesized KR, RIS, and MAD context to Judge.")
        else:
            print("  [!] No external context retrieved. Proceeding with article text & EP signal.")
            
        print(f"  Sending request to OpenRouter Cloud LLM... Please wait...")
        
        judge = _get_judge()
        if judge_mode == "panel":
            result = judge.panel_analyze(
                llm_text,
                ep_result,
                external_knowledge=external_knowledge.strip(),
                ris_summary=ris_summary.strip(),
                mad_summary=mad_summary.strip()
            )
            logged_system_prompt = _CHIEF_JUDGE_PROMPT
        elif judge_mode == "self_consistency":
            result = judge.self_consistency_analyze(
                llm_text,
                ep_result,
                n=3,
                external_knowledge=external_knowledge.strip(),
                ris_summary=ris_summary.strip(),
                mad_summary=mad_summary.strip()
            )
            logged_system_prompt = _SYSTEM_PROMPT
        else:
            result = judge.analyze(
                llm_text, 
                ep_result, 
                external_knowledge=external_knowledge.strip(),
                ris_summary=ris_summary.strip(),
                mad_summary=mad_summary.strip()
            )
            logged_system_prompt = _SYSTEM_PROMPT
        
        # Populate frontend links from Knowledge Retrieval articles
        if kr_articles:
            formatted_links = []
            from urllib.parse import urlparse
            for a in kr_articles[:6]:
                url = a.get("source") or a.get("url", "#")
                headline = a.get("headline") or a.get("title", url)
                domain = urlparse(url).netloc.replace("www.", "").upper() if url and url.startswith("http") else "EXTERNAL SOURCE"
                formatted_links.append({
                    "source": domain,
                    "title": headline,
                    "url": url
                })
            result["links"] = formatted_links

        judge_runtime = time.perf_counter() - t_judge_start

        # Log Judge input (prompt) and output (raw response + parsed)
        plog.log_judge_input(logged_system_prompt, judge.last_user_message)
        plog.log_judge_output(judge.last_raw_response, result, runtime=judge_runtime)

        # Write dedicated forensic log to output/JUDGE_output.txt
        try:
            judge.save_judge_log(
                result=result,
                mode=judge_mode,
                llm_text=llm_text,
                ep_result=ep_result,
                external_knowledge=external_knowledge.strip(),
                ris_summary=ris_summary.strip(),
                mad_summary=mad_summary.strip(),
                runtime=judge_runtime,
            )
        except Exception as log_err:
            log.warning(f"[analyzer] Could not write JUDGE_output.txt: {log_err}")

        log.info(
            f"[analyzer] Judge → {result['verdict']} ({result['confidence']})"
        )
        print("\n" + "=" * 50)
        print("  JUDGE MODULE RESULT")
        print("=" * 50)
        print(f"  Verdict     : {result['verdict']}")
        print(f"  Confidence  : {result['confidence']}")
        print(f"  Color Class : {result['color_class']}")
        print(f"  Runtime     : {judge_runtime:.2f}s")
        print("=" * 50 + "\n")

        total_runtime = time.perf_counter() - t_pipeline_start
        print("=" * 50)
        print("  PIPELINE MODULE RUNTIMES SUMMARY")
        print("=" * 50)
        print(f"  1. Pre-processing       : {prep_runtime:.2f}s")
        print(f"  2. Early Predictor      : {ep_runtime:.2f}s")
        print(f"  3. Knowledge Retrieval  : {kr_runtime:.2f}s")
        print(f"  4. Reverse Image Search : {ris_runtime:.2f}s")
        print(f"  4.5. Multi-Agent Debate : {mad_runtime:.2f}s")
        print(f"  5. Judge Module         : {judge_runtime:.2f}s")
        print("-" * 50)
        print(f"  Total Pipeline Runtime  : {total_runtime:.2f}s")
        print("=" * 50 + "\n")

        plog.save()
        yield {"type": "status", "step": "SYNTHESIZING RESULTS...", "status": "OK"}
        yield {"type": "status", "step": "OUTPUTTING PREDICTION...", "status": "IN PROGRESS"}
        yield {"type": "status", "step": "OUTPUTTING PREDICTION...", "status": "OK"}
        yield {"type": "result", "data": result}

    except Exception as exc:
        judge_runtime = time.perf_counter() - t_judge_start
        total_runtime = time.perf_counter() - t_pipeline_start
        log.error(f"[analyzer] Judge Module failed: {exc}", exc_info=True)
        plog.save()
        yield {"type": "result", "data": {
            "verdict":       "INCONCLUSIVE",
            "color_class":   "inconclusive",
            "confidence":    "0.0%",
            "justification": f"Analysis pipeline encountered an unexpected error: {exc}",
            "key_facts":     ["Pipeline error — check server logs for details."],
            "links":         [],
            "ep_signal":     ep_result,
        }
}
