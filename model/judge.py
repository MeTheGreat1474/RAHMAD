"""
model/judge.py

Judge Module — Cloud LLM reasoning layer for RAHMAD.

Receives the cleaned article text and the Early Predictor's preliminary signal,
then calls the OpenRouter API (openrouter/owl-alpha) to produce a structured
verdict. The EP signal is passed as grounding context — the Judge always makes
the final call regardless of EP confidence.

Prompting techniques applied
─────────────────────────────
  • Constitutional AI   — explicit fact-checking principles in the system prompt
  • Step-Back Prompting — domain orientation before close reading
  • Chain-of-Thought    — mandatory 7-step reasoning protocol
  • Self-Refine         — explicit self-critique step before final verdict
  • Self-Consistency    — optional: N parallel runs + majority vote (see
                          Judge.self_consistency_analyze())

Input:
    llm_text  : str   — cleaned full article text (input_processing.clean_for_llm)
    ep_result : dict  — EarlyPredictor.predict() output:
                          probability, label, confidence, pass_to_agents

Output dict (matches frontend contract exactly):
    verdict       : str   — "SATIRE OR PARODY" | "FALSE CONNECTION" |
                            "MISLEADING CONTENT" | "IMPOSTER CONTENT" |
                            "MANIPULATED CONTENT" | "FABRICATED CONTENT" |
                            "CREDIBLE" | "INCONCLUSIVE"
    color_class   : str   — "satire" | "false-connection" | "misleading" |
                            "imposter" | "manipulated" | "fabricated" |
                            "credible" | "inconclusive"
    confidence    : str   — e.g. "87.0%"
    justification : str   — reasoning paragraph
    key_facts     : list  — 3–5 key observations
    reasoning     : str   — CoT scratchpad (steps 1–7) for transparency
    links         : list  — empty list (no retrieval in prototype)
    ep_signal     : dict  — EP result passed through for frontend transparency

Usage:
    from model.judge import Judge

    judge = Judge()   # reads OPENROUTER_API_KEY from environment

    # Standard single-pass analysis
    verdict = judge.analyze(llm_text, ep_result)

    # Self-Consistency (N=3 parallel runs, majority vote)
    verdict = judge.self_consistency_analyze(llm_text, ep_result, n=3)
"""

import json
import logging
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenRouter config
# ---------------------------------------------------------------------------
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODEL_ID       = "openrouter/owl-alpha"

# ---------------------------------------------------------------------------
# Enhanced System Prompt
# Techniques: Constitutional AI + Step-Back + Chain-of-Thought + Self-Refine
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a professional news fact-checker and investigative media analyst \
working for RAHMAD (Reliable Automated Hybrid Multi-Agent Detection). \
Your role is to classify news articles into misinformation categories \
using structured, evidence-based reasoning.

══════════════════════════════════════════════════
SECTION 1 — YOUR CONSTITUTION
══════════════════════════════════════════════════
Follow these principles at all times before forming any judgment:

  P1. EVIDENCE OVER INTUITION
      Every claim in your verdict must be grounded in specific, \
observable signals from the article text. Do not classify based \
on gut feeling or surface-level tone alone.

  P2. ABSENCE OF EVIDENCE ≠ EVIDENCE OF FALSEHOOD
      Missing citations, unnamed sources, or lack of verifiable \
detail do not automatically indicate fabrication. Consider what \
is plausible for the article's domain and publication type.

  P3. DISTINGUISH WRITING STYLE FROM INTENT
      Sensationalist, emotional, or informal language is a signal, \
not a verdict. Satire uses irony deliberately; misleading content \
uses facts deceptively. Do not conflate the two.

  P4. CONTEXT DETERMINES MEANING
      A factually accurate statement can still be MISLEADING \
CONTENT if it is framed to lead the reader to a false conclusion. \
Always assess framing, not just factual accuracy.

  P5. CALIBRATED CONFIDENCE
      Only exceed 85% confidence when at least three independent \
signals from different categories (style, structure, source, \
content) all point to the same classification. Default to \
INCONCLUSIVE when evidence is mixed or insufficient.

  P6. TREAT EP AS EVIDENCE, NOT AUTHORITY
      The Early Predictor signal is a multimodal binary estimate \
(Real/Fake). Treat it as one useful data point. You may agree \
with it, partially agree, or override it — but always explain why.

══════════════════════════════════════════════════
SECTION 2 — CLASSIFICATION TYPES
══════════════════════════════════════════════════
Classify the article into exactly one of the following, \
or INCONCLUSIVE if none clearly apply.

  1. SATIRE OR PARODY
     Uses humor, irony, or sarcasm to critique or comment on \
current events. No strict intention to deceive, but can be \
mistaken for real news and shared as fact.

  2. FALSE CONNECTION
     Headlines, visuals, or captions do not accurately support \
the article content. Usually employed to create clickbait or \
mislead readers who only skim the headline.

  3. MISLEADING CONTENT
     Real information, facts, or quotes framed in a way that \
leads to a false conclusion. Includes using facts out of context \
to support a specific bias or agenda.

  4. IMPOSTER CONTENT
     Fake information claiming to be from a legitimate, trusted \
source. Includes impersonating an established news outlet, \
government official, or organisation to grant the false \
information a veneer of credibility.

  5. MANIPULATED CONTENT
     Genuine images, documents, or video footage altered to \
deceive audiences. Examples include heavily editing a video out \
of context or using photo-editing software to change the meaning \
of a photograph.

  6. FABRICATED CONTENT
     Entirely invented information, articles, or visuals designed \
to deliberately deceive, manipulate emotions, or push propaganda. \
Frequently involves AI-generated text or deepfake visuals.

  7. CREDIBLE
     No misinformation signals were detected across all analysis \
dimensions — style, framing, sourcing, internal consistency, and \
headline-to-body alignment all appear legitimate. Use this when \
the article reads as a well-sourced, factually grounded news \
report with no manipulation, fabrication, false framing, or \
impersonation patterns. Important: this does not certify the \
article as factually true — only that no misinformation patterns \
were found in the text.

  8. INCONCLUSIVE
     Use this ONLY when signals are genuinely mixed or \
contradictory — some red flags are present but insufficient to \
confidently assign a specific misinformation type. Do NOT use \
INCONCLUSIVE as a fallback for articles that simply appear \
legitimate; use CREDIBLE for those.

══════════════════════════════════════════════════
SECTION 3 — YOUR REASONING PROTOCOL
══════════════════════════════════════════════════
Work through ALL seven steps in order before forming your verdict. \
Capture your thinking in the "reasoning" field of your JSON output.

  STEP 1 — DOMAIN ORIENTATION (Step-Back)
    Before reading closely, ask: What domain does this article \
operate in (health, politics, science, celebrity, finance, \
entertainment)? What does a legitimate article in this domain \
typically look like? What are the common manipulation tactics \
used in this domain? Answering these questions first will anchor \
your analysis in the right frame of reference.

  STEP 2 — CLAIM EXTRACTION
    Identify the 3–5 most important factual claims made in the \
article. For each claim, label it as: VERIFIABLE (can be \
fact-checked), VAGUE (lacks specifics), or UNFALSIFIABLE \
(cannot be disproved by design). A high ratio of vague or \
unfalsifiable claims is a red flag.

  STEP 3 — INTERNAL CONSISTENCY AUDIT
    Check for internal consistency: Does the headline match the \
body? Do quotes match their attributed sources? Are statistics \
internally coherent (e.g. percentages that sum to more than 100)? \
Does the article contradict itself? Flag every mismatch \
with a specific example.

  STEP 4 — FRAMING AND STYLE ANALYSIS
    Assess emotional language, urgency cues ("BREAKING", "SHOCKING"), \
appeals to fear or outrage, and whether the writing style matches \
the claimed publication type. Look for narrative patterns common \
in propaganda: in-group vs. out-group framing, appeals to \
unnamed authorities, and loaded or demonizing language.

  STEP 5 — SOURCE AND ATTRIBUTION ANALYSIS
    Evaluate source credibility and transparency. Are sources \
named and traceable? Are institutions or officials quoted in \
ways that seem authentic? Does the article claim affiliation \
with a known outlet while showing signs of impersonation \
(e.g. slight name variation, unfamiliar domain)?

  STEP 6 — EP SIGNAL INTEGRATION
    Now consider the Early Predictor's signal. State whether \
it agrees or disagrees with your analysis so far. If it agrees, \
note which signals corroborate it. If it disagrees, explicitly \
state why you are overriding it — or whether it is causing you \
to reconsider a signal you underweighted.

  STEP 7 — SELF-CRITIQUE (Self-Refine)
    Before committing to a verdict, challenge yourself: What is \
the strongest argument AGAINST your current classification? \
Could you be misreading satire as fabrication, or vice versa? \
Could cultural context, technical jargon, or incomplete \
information be distorting your reading? If the counter-argument \
is strong enough to change your classification or materially \
reduce your confidence, update accordingly.

══════════════════════════════════════════════════
SECTION 4 — OUTPUT FORMAT
══════════════════════════════════════════════════
Respond with a single valid JSON object only. \
No markdown fences, no prose outside the JSON. \
Use exactly this schema:

{
  "reasoning": "<your step-by-step scratchpad covering steps 1–7, written as 5–8 concise sentences>",
  "verdict": "SATIRE OR PARODY" | "FALSE CONNECTION" | "MISLEADING CONTENT" | "IMPOSTER CONTENT" | "MANIPULATED CONTENT" | "FABRICATED CONTENT" | "CREDIBLE" | "INCONCLUSIVE",
  "color_class": "satire" | "false-connection" | "misleading" | "imposter" | "manipulated" | "fabricated" | "credible" | "inconclusive",
  "confidence": "<number>%",
  "justification": "<2–3 sentence explanation of your final verdict, citing specific signals from the article>",
  "key_facts": ["<observation 1>", "<observation 2>", "<observation 3>"]
}\
"""

_USER_TEMPLATE = """\
=== EARLY PREDICTOR SIGNAL ===
Label       : {ep_label}
Probability : {ep_prob:.4f}  (0.0 = Real, 1.0 = Fake)
Confidence  : {ep_confidence}

Note: This is a preliminary multimodal signal from a CLIP+LoRA model — \
treat it as evidence, not as a definitive answer. You must run your own \
full reasoning protocol before reaching a verdict.

=== ARTICLE TEXT ===
{llm_text}

Now work through your 7-step reasoning protocol and return your verdict \
as a JSON object.\
"""

# Maximum characters of article text sent to the LLM
_MAX_TEXT_CHARS = 6000


# ===========================================================================
# Judge
# ===========================================================================

class Judge:
    """
    Calls OpenRouter (openrouter/owl-alpha) to produce a structured
    fake-news verdict, using the Early Predictor result as grounding context.

    Prompting techniques used in the system prompt
    ───────────────────────────────────────────────
    • Constitutional AI  (Section 1 — explicit principles P1–P6)
    • Step-Back          (Step 1 of reasoning protocol — domain orientation)
    • Chain-of-Thought   (Steps 1–6 — mandatory structured reasoning)
    • Self-Refine        (Step 7 — explicit self-critique before verdict)

    Optional Self-Consistency
    ─────────────────────────
    Use self_consistency_analyze() instead of analyze() to run N parallel
    API calls and aggregate via majority vote. Recommended n=3 or n=5.
    Note: self_consistency_analyze() uses a higher temperature internally
    (default 0.7) to ensure diverse reasoning paths across runs.

    Parameters
    ----------
    api_key : str | None
        OpenRouter API key.  Defaults to the OPENROUTER_API_KEY env var.
    temperature : float
        LLM sampling temperature for single-pass analyze() (default 0.2 —
        low for consistent JSON output). Self-consistency overrides this.
    max_tokens : int
        Maximum tokens in the LLM response (default 1024).
    """

    def __init__(
        self,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self.api_key     = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.temperature = temperature
        self.max_tokens  = max_tokens

        # Set by _build_user_message / _call_api — readable after analyze()
        self.last_user_message: str = ""
        self.last_raw_response: str = ""

        if not self.api_key:
            raise ValueError(
                "[Judge] No API key found. "
                "Set the OPENROUTER_API_KEY environment variable "
                "or pass api_key= when constructing Judge()."
            )

        log.info(f"[Judge] Initialised — model: {_MODEL_ID}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, llm_text: str, ep_result: dict) -> str:
        """Format the user turn from article text and EP result."""
        msg = _USER_TEMPLATE.format(
            ep_label      = ep_result.get("label", "UNKNOWN"),
            ep_prob       = ep_result.get("probability", 0.5),
            ep_confidence = ep_result.get("confidence", "N/A"),
            llm_text      = llm_text[:_MAX_TEXT_CHARS],
        )
        self.last_user_message = msg
        return msg

    def _call_api(self, user_message: str, temperature: float | None = None) -> str:
        """POST to OpenRouter and return the raw content string."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://rahmad.local",
            "X-Title":       "RAHMAD Fake News Detector",
        }
        payload = {
            "model": _MODEL_ID,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens":  self.max_tokens,
        }

        log.info(f"[Judge] Sending request to OpenRouter — model: {_MODEL_ID}")
        response = requests.post(
            _OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]
        log.info(f"[Judge] Response received ({len(content)} chars).")
        self.last_raw_response = content
        return content

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """
        Extract and validate the JSON verdict from the LLM response.
        Handles cases where the model wraps JSON in markdown code fences.
        Now also extracts the 'reasoning' field added by the enhanced prompt.
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                raise ValueError(
                    f"[Judge] Could not extract valid JSON from LLM response:\n{raw}"
                )
            data = json.loads(match.group())

        raw_verdict = str(data.get("verdict", "")).upper()

        _VERDICT_RULES = [
            (["SATIRE", "PARODY"],                          "SATIRE OR PARODY",    "satire"),
            (["FALSE CONNECTION", "FALSE_CONNECTION"],      "FALSE CONNECTION",    "false-connection"),
            (["MISLEADING"],                                "MISLEADING CONTENT",  "misleading"),
            (["IMPOSTER", "IMPOSTOR"],                      "IMPOSTER CONTENT",    "imposter"),
            (["MANIPULATED"],                               "MANIPULATED CONTENT", "manipulated"),
            (["FABRICATED", "FABRICATION", "INVENTED"],     "FABRICATED CONTENT",  "fabricated"),
            (["CREDIBLE", "LEGITIMATE", "NO MISINFORMATION", "REAL NEWS"], "CREDIBLE", "credible"),
        ]

        verdict     = "INCONCLUSIVE"
        color_class = "inconclusive"
        for triggers, canonical, cls in _VERDICT_RULES:
            if any(t in raw_verdict for t in triggers):
                verdict     = canonical
                color_class = cls
                break

        _VALID_CLASSES = {
            "satire", "false-connection", "misleading",
            "imposter", "manipulated", "fabricated", "credible", "inconclusive",
        }
        model_color = data.get("color_class", "")
        if model_color in _VALID_CLASSES:
            color_class = model_color

        key_facts = data.get("key_facts", [])
        if isinstance(key_facts, str):
            key_facts = [key_facts]

        return {
            "verdict":       verdict,
            "color_class":   color_class,
            "confidence":    str(data.get("confidence", "50.0%")),
            "justification": str(data.get("justification", "")),
            "reasoning":     str(data.get("reasoning", "")),   # CoT scratchpad
            "key_facts":     list(key_facts),
            "links":         [],
        }

    @staticmethod
    def _parse_confidence_float(confidence_str: str) -> float:
        """Convert a confidence string like '87.0%' to a float 87.0."""
        try:
            return float(confidence_str.replace("%", "").strip())
        except ValueError:
            return 50.0

    @staticmethod
    def _aggregate_verdicts(results: list[dict]) -> dict:
        """
        Self-Consistency aggregation: take a majority vote on verdict,
        average the confidence scores, and merge key_facts + justifications.

        Parameters
        ----------
        results : list[dict]
            List of parsed verdict dicts from N parallel runs.

        Returns
        -------
        dict
            A single aggregated verdict with majority verdict, averaged
            confidence, and combined key_facts.
        """
        verdicts = [r["verdict"] for r in results]
        vote_counts = Counter(verdicts)
        majority_verdict, vote_count = vote_counts.most_common(1)[0]

        # Use results that match the majority verdict for final output
        majority_results = [r for r in results if r["verdict"] == majority_verdict]

        # Average confidence across majority runs
        avg_confidence = sum(
            Judge._parse_confidence_float(r["confidence"])
            for r in majority_results
        ) / len(majority_results)

        # Collect unique key facts across majority runs
        seen_facts = set()
        merged_facts = []
        for r in majority_results:
            for fact in r.get("key_facts", []):
                if fact not in seen_facts:
                    seen_facts.add(fact)
                    merged_facts.append(fact)

        # Use the highest-confidence run's justification and reasoning
        best_run = max(
            majority_results,
            key=lambda r: Judge._parse_confidence_float(r["confidence"]),
        )

        log.info(
            f"[Judge][Self-Consistency] Vote: {dict(vote_counts)} | "
            f"Majority: {majority_verdict} ({vote_count}/{len(results)}) | "
            f"Avg confidence: {avg_confidence:.1f}%"
        )

        return {
            "verdict":       majority_verdict,
            "color_class":   best_run["color_class"],
            "confidence":    f"{avg_confidence:.1f}%",
            "justification": best_run["justification"],
            "reasoning":     best_run["reasoning"],
            "key_facts":     merged_facts[:5],    # cap at 5
            "links":         [],
            "sc_votes":      dict(vote_counts),   # transparency: vote breakdown
        }

    @staticmethod
    def _fallback_response(error_msg: str) -> dict:
        """Safe fallback verdict returned when the API call or parsing fails."""
        log.warning(f"[Judge] Returning fallback response. Reason: {error_msg}")
        return {
            "verdict":       "INCONCLUSIVE",
            "color_class":   "inconclusive",
            "confidence":    "0.0%",
            "justification": (
                "The Judge Module encountered an error and could not complete "
                f"its analysis: {error_msg}"
            ),
            "reasoning":     "",
            "key_facts":     [
                "Analysis failed — check your OPENROUTER_API_KEY and network connection."
            ],
            "links": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, llm_text: str, ep_result: dict) -> dict:
        """
        Standard single-pass analysis of a news article.

        Applies: Constitutional AI, Step-Back, Chain-of-Thought, Self-Refine
        via the enhanced system prompt. Temperature is kept low (default 0.2)
        for deterministic, consistent JSON output.

        Parameters
        ----------
        llm_text : str
            Cleaned full article text from input_processing.clean_for_llm().
        ep_result : dict
            Output of EarlyPredictor.predict(). Pass {} if EP was skipped.

        Returns
        -------
        dict
            verdict, color_class, confidence, justification, reasoning,
            key_facts, links, ep_signal.
        """
        self.last_raw_response = ""
        self.last_user_message = ""
        try:
            user_message = self._build_user_message(llm_text, ep_result)
            raw_response = self._call_api(user_message)
            result       = self._parse_response(raw_response)

            log.info(
                f"[Judge] Final verdict: {result['verdict']} "
                f"({result['confidence']})"
            )

            result["ep_signal"] = ep_result
            return result

        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            body   = exc.response.text[:200] if exc.response is not None else ""
            log.error(f"[Judge] HTTP {status}: {body}")
            result = self._fallback_response(f"HTTP error {status}")
            result["ep_signal"] = ep_result
            return result

        except Exception as exc:
            log.error(f"[Judge] Unexpected error: {exc}", exc_info=True)
            result = self._fallback_response(str(exc))
            result["ep_signal"] = ep_result
            return result

    def self_consistency_analyze(
        self,
        llm_text: str,
        ep_result: dict,
        n: int = 3,
        sc_temperature: float = 0.7,
    ) -> dict:
        """
        Self-Consistency analysis: run N parallel reasoning paths and
        aggregate via majority vote on the verdict.

        Each run uses a higher temperature (sc_temperature, default 0.7)
        so that different reasoning paths are explored. The final verdict
        is the one that appears most often across all N runs. Confidence
        is averaged across runs that agree on the majority verdict.

        This method is best used when accuracy is critical and the added
        latency (N x single-call time, parallelised) is acceptable.
        Recommended: n=3 (fast) or n=5 (higher reliability).

        Parameters
        ----------
        llm_text : str
            Cleaned full article text.
        ep_result : dict
            Output of EarlyPredictor.predict(). Pass {} if EP was skipped.
        n : int
            Number of independent reasoning paths to sample (default 3).
        sc_temperature : float
            Sampling temperature for each run (default 0.7). Higher than
            the standard temperature to encourage diverse reasoning paths.

        Returns
        -------
        dict
            Aggregated verdict dict. Includes 'sc_votes' key showing the
            full vote breakdown across all N runs for transparency.
        """
        user_message = self._build_user_message(llm_text, ep_result)

        log.info(
            f"[Judge][Self-Consistency] Starting {n} parallel runs "
            f"at temperature={sc_temperature}…"
        )

        results  = []
        failures = 0

        # Run N API calls in parallel using a thread pool
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = {
                executor.submit(
                    self._call_api, user_message, sc_temperature
                ): i
                for i in range(n)
            }
            for future in as_completed(futures):
                run_idx = futures[future]
                try:
                    raw      = future.result()
                    parsed   = self._parse_response(raw)
                    results.append(parsed)
                    log.info(
                        f"[Judge][SC] Run {run_idx + 1}/{n} → "
                        f"{parsed['verdict']} ({parsed['confidence']})"
                    )
                except Exception as exc:
                    failures += 1
                    log.warning(
                        f"[Judge][SC] Run {run_idx + 1}/{n} failed: {exc}"
                    )

        if not results:
            result = self._fallback_response(
                f"All {n} Self-Consistency runs failed."
            )
            result["ep_signal"] = ep_result
            return result

        if failures > 0:
            log.warning(
                f"[Judge][SC] {failures}/{n} runs failed — "
                f"aggregating from {len(results)} successful runs."
            )

        aggregated = self._aggregate_verdicts(results)
        aggregated["ep_signal"] = ep_result
        return aggregated


# ---------------------------------------------------------------------------
# Quick self-test — run directly:  python -m model.judge
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _sample_text = (
        "Scientists at a leading university have discovered that drinking "
        "coffee every morning extends human lifespan by 20 years, according "
        "to a study that has not yet been peer reviewed. The researchers "
        "claim the findings will revolutionise medicine worldwide."
    )
    _sample_ep = {
        "probability":    0.72,
        "label":          "FAKE",
        "confidence":     "72.0%",
        "pass_to_agents": False,
    }

    print("Initialising Judge…")
    j = Judge()

    print("\n--- Standard analyze() ---")
    out = j.analyze(_sample_text, _sample_ep)
    for k, v in out.items():
        print(f"  {k}: {v}")

    print("\n--- Self-Consistency analyze() [n=3] ---")
    out_sc = j.self_consistency_analyze(_sample_text, _sample_ep, n=3)
    for k, v in out_sc.items():
        print(f"  {k}: {v}")
