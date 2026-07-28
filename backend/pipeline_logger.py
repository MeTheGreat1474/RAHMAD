"""
backend/pipeline_logger.py

Structured pipeline logger for RAHMAD.

Captures every stage of the analysis pipeline and writes a human-readable
log to output_log.txt in the project root after each request.

Stages logged:
  [1] Pre-processing — raw input  (text preview + image info)
  [2] Pre-processing — output     (llm_text, lead_text, tensor shapes/values)
  [3] CLIP encoder   — output     (text_embeds, image_embeds, fused tensor)
  [4] Early Predictor — output    (probability, label, confidence, pass_to_agents)
  [5] Judge (LLM)   — input       (full system prompt + full user message)
  [6] Judge (LLM)   — output      (raw API response + parsed verdict)

Usage (from analyzer.py):
    from backend.pipeline_logger import PipelineLogger

    plog = PipelineLogger()
    plog.log_preprocessing_input(raw_text, image)
    plog.log_preprocessing_output(processed)
    plog.log_clip_output(text_embeds, image_embeds, fused)
    plog.log_ep_output(ep_result)
    plog.log_judge_input(system_prompt, user_message)
    plog.log_judge_output(raw_response, parsed_result)
    plog.save()
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Log file location — project root (next to app.py)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH      = _PROJECT_ROOT / "output_log.txt"

_SEP_MAJOR = "=" * 80
_SEP_MINOR = "-" * 60


# ===========================================================================
# PipelineLogger
# ===========================================================================

class PipelineLogger:
    """
    Accumulates structured log lines for one analysis run, then flushes the
    full log to output_log.txt.  Create a new instance per request.
    """

    def __init__(self):
        self._lines: list[str] = []
        self._start            = datetime.now()
        ts = self._start.strftime("%Y-%m-%d  %H:%M:%S")
        self._w(_SEP_MAJOR)
        self._w(f"  RAHMAD PIPELINE LOG  —  {ts}")
        self._w(_SEP_MAJOR)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _w(self, line: str = "") -> None:
        self._lines.append(line)

    def _section(self, number: int, title: str) -> None:
        self._w()
        self._w(f"[{number}]  {title}")
        self._w(_SEP_MINOR)

    def _kv(self, key: str, value: object, indent: int = 4) -> None:
        self._w(f"{' ' * indent}{key:<22}: {value}")

    def _block(self, label: str, content: str, indent: int = 4) -> None:
        pad = " " * indent
        self._w(f"{pad}--- {label} ---")
        for line in str(content).splitlines():
            self._w(f"{pad}  {line}")

    @staticmethod
    def _tensor_summary(t: torch.Tensor, n_vals: int = 8) -> str:
        """Return shape, stats, and first N values of a tensor as a string.
        Handles both floating-point tensors (CLIP embeddings, pixel_values)
        and integer tensors (input_ids, attention_mask — norm is not defined
        for these, so dtype + min/max is shown instead).
        """
        cpu   = t.detach().cpu()
        flat  = cpu.flatten().tolist()
        more  = f", … ({len(flat) - n_vals} more)" if len(flat) > n_vals else ""

        if cpu.is_floating_point():
            vals  = ", ".join(f"{v:+.4f}" for v in flat[:n_vals])
            norm  = float(torch.norm(cpu).item())
            tmin  = float(cpu.min().item())
            tmax  = float(cpu.max().item())
            stats = f"norm={norm:.4f}  min={tmin:+.4f}  max={tmax:+.4f}"
        else:
            # Integer tensor (e.g. input_ids, attention_mask) — norm not applicable
            vals  = ", ".join(str(int(v)) for v in flat[:n_vals])
            tmin  = int(cpu.min().item())
            tmax  = int(cpu.max().item())
            stats = f"dtype={cpu.dtype}  min={tmin}  max={tmax}"

        return (
            f"shape={list(t.shape)}  "
            f"{stats}  "
            f"first_{n_vals}=[{vals}{more}]"
        )

    # ------------------------------------------------------------------
    # Public logging methods  (call in pipeline order)
    # ------------------------------------------------------------------

    def log_preprocessing_input(
        self,
        raw_text: str,
        image: "PILImage.Image | None",
    ) -> None:
        self._section(1, "PRE-PROCESSING  —  INPUT")

        # Text preview (up to 500 chars, newlines escaped)
        preview = raw_text[:500].replace("\n", "\\n")
        if len(raw_text) > 500:
            preview += f"  … [{len(raw_text)} chars total]"
        self._w(f"    raw_text:")
        self._w(f'      "{preview}"')
        self._w()

        # Image info
        if image is not None:
            self._kv("image.size", f"{image.size[0]} × {image.size[1]} px")
            self._kv("image.mode", image.mode)
        else:
            self._kv("image", "None  (no image provided — EP will be skipped)")

    def log_preprocessing_output(self, processed: dict) -> None:
        self._section(2, "PRE-PROCESSING  —  OUTPUT")

        # llm_text
        llm = processed["llm_text"]
        llm_preview = llm[:500].replace("\n", "\\n")
        if len(llm) > 500:
            llm_preview += f"  … [{len(llm)} chars total]"
        self._w(f"    llm_text  (cleaned full text — goes to Judge):")
        self._w(f'      "{llm_preview}"')
        self._w()

        # lead_text
        lead = processed["lead_text"]
        self._w(f"    lead_text  (headline + 1st sentence — goes to CLIP):")
        self._w(f'      "{lead[:300]}"')
        self._w()

        # Tensors
        self._kv("input_ids",      self._tensor_summary(processed["input_ids"],     n_vals=12))
        self._kv("attention_mask", self._tensor_summary(processed["attention_mask"], n_vals=12))
        self._kv("pixel_values",   self._tensor_summary(processed["pixel_values"],   n_vals=6))

    def log_clip_output(
        self,
        text_embeds:  torch.Tensor,
        image_embeds: torch.Tensor,
        fused:        torch.Tensor,
    ) -> None:
        self._section(3, "CLIP ENCODER  —  OUTPUT")
        self._kv("text_embeds  [1×512]",  self._tensor_summary(text_embeds,  n_vals=8))
        self._kv("image_embeds [1×512]",  self._tensor_summary(image_embeds, n_vals=8))
        self._kv("fused        [1×1024]", self._tensor_summary(fused,        n_vals=8))

    def log_ep_output(self, ep_result: dict) -> None:
        self._section(4, "EARLY PREDICTOR  —  OUTPUT")
        prob = ep_result.get("probability", None)
        self._kv("probability",
                 f"{prob:.6f}   (0.0 = Real, 1.0 = Fake)" if prob is not None else "N/A")
        self._kv("label",          ep_result.get("label",          "N/A"))
        self._kv("confidence",     ep_result.get("confidence",     "N/A"))
        self._kv("pass_to_agents", ep_result.get("pass_to_agents", "N/A"))

    def log_ep_skipped(self, reason: str = "no image provided") -> None:
        self._section(3, "CLIP ENCODER  —  SKIPPED")
        self._w(f"    Reason: {reason}")
        self._section(4, "EARLY PREDICTOR  —  SKIPPED")
        self._w(f"    Reason: {reason}")

    def log_knowledge_retrieval(self, kr_data: dict) -> None:
        self._section(5, "KNOWLEDGE RETRIEVAL  —  OUTPUT")
        self._kv("claim_queried", kr_data.get("claim", ""))
        articles = kr_data.get("articles", [])
        self._kv("articles_queried", f"[{len(articles)} article(s) survived whitelist & heuristics]")
        for i, art in enumerate(articles, 1):
            self._w(f"      {i}. Headline : {art.get('headline', 'N/A')}")
            self._w(f"         Source   : {art.get('source', 'N/A')}")
            if art.get("date"):
                self._w(f"         Date     : {art.get('date')}")
        self._w()
        self._block("CONDENSED SUMMARY (LLM)", kr_data.get("summary", ""))

    def log_judge_input(self, system_prompt: str, user_message: str) -> None:
        self._section(6, "JUDGE (LLM)  —  INPUT")
        self._block("SYSTEM PROMPT", system_prompt)
        self._w()
        self._block("USER MESSAGE", user_message)

    def log_judge_output(self, raw_response: str, parsed: dict) -> None:
        self._section(7, "JUDGE (LLM)  —  OUTPUT")
        self._block("RAW API RESPONSE", raw_response)
        self._w()
        self._w(f"    --- PARSED VERDICT ---")
        self._kv("verdict",       parsed.get("verdict",       "?"))
        self._kv("color_class",   parsed.get("color_class",   "?"))
        self._kv("confidence",    parsed.get("confidence",    "?"))
        just = str(parsed.get("justification", ""))
        self._kv("justification", just[:300] + ("…" if len(just) > 300 else ""))
        facts = parsed.get("key_facts", [])
        self._kv("key_facts",     f"[{len(facts)} item(s)]")
        for i, fact in enumerate(facts, 1):
            self._w(f"      {i}. {fact}")

    # ------------------------------------------------------------------
    # Flush to disk
    # ------------------------------------------------------------------

    def save(self) -> Path:
        elapsed = (datetime.now() - self._start).total_seconds()
        self._w()
        self._w(_SEP_MAJOR)
        self._w(f"  END OF LOG  |  total elapsed: {elapsed:.2f}s")
        self._w(_SEP_MAJOR)

        content = "\n".join(self._lines) + "\n"
        LOG_PATH.write_text(content, encoding="utf-8")
        print(f"[pipeline_logger] Log written -> {LOG_PATH}")
        return LOG_PATH
