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
from datetime import datetime
from pathlib import Path

import requests

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)


class JudgeResponseError(RuntimeError):
    """The LLM reply could not be trusted as a complete JSON verdict."""


class JudgeTruncatedResponse(JudgeResponseError):
    """The reply was cut off by the output-token cap (finish_reason == 'length')."""


# ---------------------------------------------------------------------------
# OpenRouter config
# ---------------------------------------------------------------------------
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_DEFAULT_FALLBACKS = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free,inclusionai/ling-3.0-flash:free,poolside/laguna-m.1:free"
_JUDGE_FALLBACK_RAW = os.environ.get("JUDGE_MODEL_FALLBACK") or os.environ.get("OPENROUTER_MODEL_FALLBACK", _DEFAULT_FALLBACKS)
_JUDGE_FALLBACKS = [m.strip() for m in _JUDGE_FALLBACK_RAW.split(",") if m.strip()]

_MODEL_ID = os.environ.get("JUDGE_MODEL") or os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash")

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
Always assess framing.

  P5. CALIBRATED CONFIDENCE
      Only exceed 85% confidence when at least three independent \
signals from different categories (style, structure, source, \
content) all point to the same classification. Default to \
INCONCLUSIVE when evidence is mixed or insufficient.

  P6. TREAT EARLY PREDICTOR AS EVIDENCE, NOT AUTHORITY
      The Early Predictor signal is a preliminary multimodal estimate \
(Real/Fake). Treat it as one useful data point. You may agree \
with it, partially agree, or override it — but always explain why.

  P7. CROSS-VERIFY WITH EXTERNAL KNOWLEDGE EVIDENCE
      Verified external news reports and knowledge retrieval provide \
authoritative evidence extracted from reputable news outlets. \
Treat this external knowledge as ground-truth factual context. \
Cross-reference the core claims in the article text against this knowledge. \
Direct corroboration strongly supports a CREDIBLE classification, \
while direct contradiction signals FABRICATED, MISLEADING, or FALSE CONNECTION.

  P8. CROSS-VERIFY VISUAL CONTEXT VIA REVERSE IMAGE SEARCH
      If Reverse Image Search findings are provided, use them to assess \
whether the article's image is authentic or decontextualized. Distinguish \
carefully between:
      - 'Authentic Context / Matching Event': The image genuinely documents the same \
event, disaster, location, or subject matter, even if headlines vary across news outlets \
or wire photo licensing catalogs (e.g. Reuters, SCMP, Reuters Connect).
      - 'Out-of-Context (Misattributed / Recycled)': The image is demonstrably recycled from \
a completely different historical year, different event, or unrelated location. This \
strongly supports a FALSE CONNECTION or MISLEADING CONTENT classification.
      - 'Generic / Illustrative Imagery': Standard stock or symbolic photography without \
deceptive intent. Treat as neutral visual evidence.
      - 'Inconclusive / Insufficient Evidence': Ambiguous visual provenance; rely \
primarily on textual and debate evidence.

      CRITICAL CAVEAT 1 (AUTHENTIC PHOTO ≠ CREDIBLE TEXT):
      An authentic photograph does NOT certify that the textual claims, captions, or attributions \
are true. Deceptive actors frequently pair genuine historical or wire photos with unverified, \
speculative, or false narratives (Cheapfakes / False Connection). An authentic image only proves that \
the media file itself is not digitally altered or recycled from an unrelated event. If the textual \
claims remain uncorroborated by Knowledge Retrieval (KR) or are challenged as speculative/unverified \
in Debate (MAD), you MUST critically assess whether the article creates a FALSE CONNECTION or \
MISLEADING CONTENT, and NEVER default to CREDIBLE simply because the photograph is authentic.

      CRITICAL CAVEAT 2 (EVIDENTIARY HIERARCHY — FACT-CHECKING OUTRANKS AUTOMATED VISUAL HEURISTICS):
      Multimodal fact-checking requires distinguishing between genuine cheapfakes (unrelated images maliciously attached to stories) \
and authentic archival/historical photos resurfacing online:
      - PRIMARY GROUND TRUTH: Verified external news reports and fact-checking archives (Knowledge Retrieval from Snopes, Reuters, AP, etc.) \
represent authoritative ground truth. If verified external knowledge confirms that the photograph genuinely documents the subject, event, \
or historical person described, this authoritative factual evidence OUTRANKS automated Reverse Image Search heuristics. Automated image \
search algorithms can misinterpret authentic archival photographs, historical videos, or viral event clips as 'recycled' when they re-circulate \
on modern social media. In such cases, you MUST NOT penalize the article as a FALSE CONNECTION.
      - DECONTEXTUALIZED CHEAPFAKES: However, when external knowledge does NOT corroborate the visual match, or when a demonstrably unrelated \
photograph is attached to depict a separate news event, this is a genuine FALSE CONNECTION (Cheapfake). When the image is demonstrably \
out-of-context and unsupported by verified news reporting, classify as FALSE CONNECTION or MISLEADING CONTENT.

  P9. INCORPORATE MULTI-AGENT DEBATE PERSPECTIVES
      If debate analysis findings are provided, use them to understand \
the nuanced arguments from both 'Real News' and 'Fake News' perspectives \
argued by domain experts. This discourse provides structured critical \
analysis of the claims. Integrate these insights into your reasoning.

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

  STEP 6 — MULTIMODAL AND MULTI-AGENT EVIDENCE INTEGRATION
    Evaluate all external evidence streams:
    1. External Knowledge Retrieval: Cross-reference claims extracted in Step 2 \
against the verified external news reports. State clearly if \
external news corroborates, refutes, or is absent for the claims.
    2. Reverse Image Search: If Reverse Image Search evidence is provided, check the \
explicit 'Out-of-Context Imagery Detected' status and category. Apply the Evidentiary Hierarchy: \
If marked as 'Out-of-Context (Misattributed / Recycled)', first check whether verified External Knowledge (KR) \
explicitly corroborates that the claim accurately describes this authentic photograph/event. If KR corroborates the \
authenticity of the photo/event, treat the visual provenance as authentic context (the automated visual agent likely flagged \
archival re-circulation). If KR does NOT corroborate the photo or confirms a mismatch/hoax, classify as FALSE CONNECTION (Cheapfake). \
If marked as 'Authentic Context / Matching Event' or 'Generic / Illustrative Imagery', do NOT penalize an article \
simply because syndicated outlets rephrase headlines or use illustrative photos. \
Perform an Image-to-Text Entailment check: does the authentic image actually substantiate the specific claims or named identities in the text? \
If an authentic photo is paired with uncorroborated, speculative, or fabricated text claims, classify as FALSE CONNECTION or MISLEADING CONTENT, NOT CREDIBLE.
    3. Multi-Agent Debate: If debate findings are provided, leverage \
the structured arguments generated by the domain experts. Synthesize \
their findings regarding the article's credibility.
    4. Early Predictor Signal: State whether the Early Predictor's vision-text score \
agrees or disagrees with your factual and visual evaluation. Synthesize \
all evidence to justify your classification and calibrate final confidence.

  STEP 7 — SELF-CRITIQUE (Self-Refine)
    Before committing to a verdict, challenge yourself: What is \
the strongest argument AGAINST your current classification? \
Could you be misreading satire as fabrication, or vice versa? \
Could cultural context, technical jargon, or incomplete \
information be distorting your reading? If the counter-argument \
is strong enough to change your classification or materially \
reduce your confidence, update accordingly.

══════════════════════════════════════════════════
SECTION 4 — OUTPUT FORMAT & USER-FACING TERMINOLOGY
══════════════════════════════════════════════════
Your output MUST be a single, valid JSON object formatted exactly as follows.
Important: Use markdown bolding (e.g. **key finding** or **critical claim**) to emphasize important phrases, entities, dates, and core evidence across "justification", "reasoning", and "key_facts".

CRITICAL AUDIENCE & TERMINOLOGY RULE:
Your reasoning, justification, and key facts will be read directly by everyday users who do not know internal system terminology.
- NEVER use internal technical acronyms or module codes, specifically:
  • Do NOT say "MAD", "MAD module", or "MAD summary" -> instead use clear phrases like "Debate shows...", "The debate analysis indicates...", or "Multi-Agent Debate arguments show...".
  • Do NOT say "RIS", "RIS module", or "RIS pipeline" -> instead use clear phrases like "Reverse Image Search shows...", "Visual verification indicates...", or "Image provenance checks reveal...".
  • Do NOT say "KR", "KR module", or "KR pipeline" -> instead use clear phrases like "External knowledge retrieval shows...", "Verified news reports confirm...", "External fact-checking reveals...", or "Independent news sources indicate...".
  • Do NOT say "EP" or "EP module" -> instead use "Early Predictor", "Initial screening", or "Preliminary multimodal assessment".
- Explain the evidence naturally and objectively, as a professional investigative journalist or fact-checker explaining findings to the public.

{
  "reasoning": "<your step-by-step scratchpad covering steps 1–7, formatted as structured Markdown bullet points (e.g. '* **Visual Verification:** ...'). Do NOT write a wall of text. Use 1-2 detailed sentences per bullet point, strictly using user-friendly terms like 'Reverse Image Search shows...' or 'Debate shows...'. You MUST explicitly include dedicated bullet points for ALL pipeline components: Factual Assessment, Linguistic Assessment, Logical Assessment, Debate Analysis, Early Predictor (initial screening), External Knowledge Retrieval (verified news), and Reverse Image Search (visual provenance). Summarize their results transparently, even if they contradict your final verdict.>",
  "verdict": "SATIRE OR PARODY" | "FALSE CONNECTION" | "MISLEADING CONTENT" | "IMPOSTER CONTENT" | "MANIPULATED CONTENT" | "FABRICATED CONTENT" | "CREDIBLE" | "INCONCLUSIVE",
  "color_class": "satire" | "false-connection" | "misleading" | "imposter" | "manipulated" | "fabricated" | "credible" | "inconclusive",
  "confidence": "<number>%",
  "justification": "<3–5 sentence comprehensive explanation of your final verdict, detailing the main evidence and your reasoning clearly, using **bold** for critical signals and key entities, strictly using user-friendly terms like 'Reverse Image Search shows...' or 'Debate shows...'>",
  "key_facts": ["<observation 1 with **key points** bolded>", "<observation 2>", "<observation 3>"]
}
"""

_USER_TEMPLATE = """\
=== KNOWLEDGE RETRIEVAL EVIDENCE ===
{external_knowledge}

Note: The numbered [S1], [S2], … blocks are verbatim snippets retrieved \
from reputable whitelisted news outlets via semantic similarity filtering — \
these are the only citable evidence. The "SYNTHESIS" section is an \
LLM-generated outline of those snippets, NOT independent evidence: never \
cite it as corroboration and never treat a detail that appears only in the \
synthesis as covered.

=== REVERSE IMAGE SEARCH EVIDENCE ===
{ris_summary}

=== MULTI-AGENT DEBATE SUMMARY ===
{mad_summary}

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
as a JSON object.
CRITICAL REMINDER: Write in clear language for end users. Do NOT use technical acronyms like MAD, RIS, KR, or EP in your output text. Instead, use natural phrases such as "Debate shows...", "Reverse Image Search shows...", "External knowledge retrieval shows...", and "Early Predictor signal indicates...".\
"""

# Maximum characters of article text sent to the LLM
_MAX_TEXT_CHARS = 6000

# ---------------------------------------------------------------------------
# Multi-Judge Panel Prompts (Specialized Personas & Chief Synthesizer)
# ---------------------------------------------------------------------------

_FACT_CHECKER_PROMPT = """\
You are the Senior Fact-Checker and Evidence Verifier on the RAHMAD judicial panel.
Your sole mission is to assess the FACTUAL ACCURACY, PROVENANCE, and EXTERNAL CORROBORATION of the claims made in the news article.

Evidence available to you:
1. Knowledge Retrieval Evidence (ground-truth external reports from reputable news sources) — each snippet is tagged with a source reference like [S1], [S2], etc.
2. Reverse Image Search Evidence (visual provenance, authenticity vs. out-of-context recycling)
3. Multi-Agent Debate Summary (critical arguments from Real News and Fake News domain debaters)

Evaluation Protocol:
- Build a ledger of only the article's 3-6 CORE verifiable claims (dates, events, people, actions, outcomes). Do NOT create a separate entry for every minor figure or detail; group details under the claim they support.
- For EACH core claim, check the Knowledge Retrieval snippets: Is the claim SUPPORTED (a snippet directly confirms it), CONTRADICTED (a snippet explicitly refutes it), or NOT_COVERED (no snippet addresses it)?
- CRITICAL CONSISTENCY RULE: Your "factual_analysis" MUST be derived from the ledger. If the analysis says retrieved reports corroborate, match, track, or confirm an event, that event MUST appear as a SUPPORTED ledger entry with its snippet ref. Never describe corroboration in prose while marking the same claim NOT_COVERED.
- CRITICAL: You MUST cite the specific snippet reference(s) (e.g., [S1], [S2]) that justify SUPPORTED or CONTRADICTED status. A claim is CONTRADICTED ONLY if a retrieved snippet explicitly refutes it — internal knowledge or absence of coverage does NOT constitute contradiction.
- CITATION IS NOT OPTIONAL: if you write any note saying a retrieved report "confirms", "corroborates", "supports", "matches", "tracks", or "documents" a claim, that claim's status MUST be SUPPORTED and evidence_ref MUST name the [S#] snippet that does so. A note claiming corroboration with an empty evidence_ref is invalid and will be downgraded to NOT_COVERED.
- A detail that appears only in the SYNTHESIS outline (not in any numbered [S#] snippet) is NOT covered by evidence. Judge coverage against the numbered snippets only.
- Analyze Reverse Image Search findings: Is the image authentic to the event, or misattributed/recycled?
- Evaluate Multi-Agent Debate arguments: Which side presents stronger factual documentation?
- Your risk score will be computed automatically from the validated evidence ledger; do not invent a score.

CRITICAL AUDIENCE & TERMINOLOGY RULE:
Your factual analysis will be presented to end users.
NEVER use internal system acronyms like "KR", "KR module", "RIS", "RIS module", "MAD", "MAD module", or "EP".
Instead, use user-friendly phrases:
- "Reverse Image Search shows..." or "Visual verification indicates..." (instead of RIS)
- "Debate shows..." or "The multi-agent debate indicates..." (instead of MAD)
- "External news reports show..." or "Knowledge retrieval indicates..." (instead of KR)
- "Initial screening indicates..." (instead of EP)

Output Format:
You must output a single, valid JSON object with an EVIDENCE LEDGER:
{
  "factual_analysis": "<concise 3-5 sentence analysis of claim corroboration, image authenticity, and external evidence alignment using user-friendly phrasing>",
  "claim_ledger": [
    {
      "claim": "<core factual claim from the article>",
      "status": "SUPPORTED" | "CONTRADICTED" | "NOT_COVERED",
      "evidence_ref": "<snippet ref(s) like [S1], [S2] that justify the status — REQUIRED for SUPPORTED/CONTRADICTED>",
      "notes": "<optional brief note>"
    }
  ],
  "prior_conflicts": [
    "<any internal knowledge that conflicts with a SUPPORTED claim, flagged but NOT scored>"
  ],
  "image_assessment": "<authentic | out-of-context | recycled | unverified>",
  "factual_risk_score": null
}
"""

_LINGUIST_PROMPT = """\
You are the Senior Linguist and Stylometrics Analyst on the RAHMAD judicial panel.
Your sole mission is to assess the LINGUISTIC PATTERNS, EMOTIONAL MANIPULATION, TONE, and RHETORICAL FRAMING of the news article.
Do NOT attempt to fact-check external claims; focus strictly on HOW the text is written.

Evaluation Protocol:
- Emotional manipulation: Does the article deploy sensationalism, rage-bait, panic, excessive exclamation, or loaded adjectives?
- Headline vs. Body alignment: Does the headline make exaggerated, clickbait, or unsupported claims compared to the body text?
- Genre & Stylistic consistency: Does the text adhere to objective journalistic standards, or does it exhibit hallmarks of satire, parody, partisan agitprop, or conspiratorial writing?
- Semantic coherence: Is the language logically coherent, or does it rely on vague innuendo and rhetorical sleight-of-hand?
- Assign a Linguistic Risk Score from 1 to 10 (1 = highly objective, neutral, professional journalistic tone; 10 = extreme clickbait, inflammatory propaganda, deceptive framing, or obvious satire).

Output Format:
You must output a single, valid JSON object:
{
  "stylistic_analysis": "<concise 3-5 sentence analysis of tone, emotional cues, headline framing, and rhetorical markers>",
  "detected_devices": ["<e.g. Sensationalism, Loaded Language, Fear Appeal, Irony/Satirical Tropes>"],
  "emotional_manipulation_level": "Low" | "Medium" | "High",
  "linguistic_risk_score": <integer from 1 to 10>
}
"""

_LOGICIAN_PROMPT = """\
You are the Senior Logician and Intent Analyst on the RAHMAD judicial panel.
Your sole mission is to assess the INTERNAL CONSISTENCY, ARGUMENT COHERENCE, and COMMUNICATIVE INTENT of the news article using ONLY the provided text and the evidence snippets tagged [S1], [S2], etc.

Evaluation Protocol:
- Internal consistency: Do the statements within the article contradict one another? Do reported numbers/statistics add up?
- Source attribution: Are quoted authorities, institutions, or witnesses specific and verifiable against the provided snippets? Flag any attribution that has NO supporting snippet as "unverified". Do NOT judge credibility from internal knowledge.
- Communicative intent: What is the primary purpose of the piece? (To inform objectively, to satirize current events, to generate clickbait ad revenue, to deceive/defame, or to spread propaganda?)
- Logical fallacies: Detect false dichotomies, non-sequiturs, strawman arguments, or guilt-by-association.
- Every contradiction or fallacy you cite MUST end with a snippet reference like [S3] that demonstrates the issue. Unsupported assertions will be downgraded.
- Your risk score will be computed automatically from the validated evidence ledger; do not invent a score.

Output Format:
You must output a single, valid JSON object with a LOGIC LEDGER:
{
  "logical_analysis": "<concise 3-5 sentence analysis of internal consistency, sourcing verification, and underlying intent>",
  "logic_ledger": [
    {
      "issue": "<internal contradiction, unverified attribution, or fallacy>",
      "status": "CONTRADICTED" | "UNVERIFIED" | "FALLACY",
      "evidence_ref": "<snippet ref(s) like [S1], [S2] that demonstrate the issue — REQUIRED for CONTRADICTED/FALLACY>",
      "notes": "<optional brief note>"
    }
  ],
  "perceived_intent": "Informational" | "Satirical" | "Deceptive" | "Imposter" | "Sensationalist",
  "logical_risk_score": null
}
"""

_CHIEF_JUDGE_PROMPT = """\
You are the Chief Judicial Magistrate and Synthesis Arbiter for RAHMAD (Reliable Automated Hybrid Multi-Agent Detection).
You preside over a panel of three specialized investigative judges who have analyzed the news article:
1. The Fact-Checker (External Evidence, Reverse Image Search provenance, and Multi-Agent Debate verification) — provides a claim_ledger with SUPPORTED/CONTRADICTED/NOT_COVERED statuses and snippet citations
2. The Linguist (Writing style, emotional manipulation, sensationalism, and rhetoric)
3. The Logician (Internal consistency, source verification against snippets, and communicative intent) — provides a logic_ledger with CONTRADICTED/UNVERIFIED/FALLACY statuses and snippet citations

You also receive the Early Predictor multimodal signal, the original article text, and the computed risk scores derived from the validated evidence ledgers.

Judicial Synthesis Principles:
- Factual Priority: If the Fact-Checker's ledger shows core claims CONTRADICTED (with valid snippet refs) or the image is out-of-context (recycled), strong presumption toward FABRICATED CONTENT, FALSE CONNECTION, or MISLEADING CONTENT.
- Cheapfake / False Connection Caution: An authentic photograph does NOT validate an uncorroborated, speculative, or false textual claim. If the image is authentic but the caption or claim is unsupported (NOT_COVERED) by external news or challenged in debate, classify as FALSE CONNECTION or MISLEADING CONTENT, NOT CREDIBLE.
- Satire/Parody Recognition: If the Linguist and Logician detect satirical markers, irony, and humorous intent, classify as SATIRE OR PARODY even if the claims are factually untrue.
- Imposter Detection: If the Logician detects false impersonation of an established institution or journalist (with snippet evidence), classify as IMPOSTER CONTENT.
- Credibility Standard: When core claims are SUPPORTED with valid snippet refs, no CONTRADICTED core claims, logic_ledger is clean, and linguistic risk is low, classify as CREDIBLE.
- Prior-Knowledge Firewall: "prior_conflicts" are unverified things the fact-checker model believes that disagree with the article. They are NOT evidence. Never cite them in "reasoning", "justification" or "key_facts"; never treat them as a CONTRADICTED claim; never let them by themselves produce FABRICATED CONTENT or MISLEADING CONTENT. If the claim ledger has no CONTRADICTED claim and the only doubt comes from prior knowledge, return INCONCLUSIVE (or FALSE CONNECTION when the images cannot evidence the described event) and state that retrieved reports did not contradict the article.
- Inconclusive Standard: Use INCONCLUSIVE when (a) the Fact-Checker's ledger has no SUPPORTED claims and no CONTRADICTED claims (all NOT_COVERED) AND the subject is a named major entity/event → suspicious (lean FALSE CONNECTION), or (b) signals remain irreconcilably divided without sufficient evidence.

CRITICAL AUDIENCE & TERMINOLOGY RULE:
Your output ("reasoning", "justification", and "key_facts") is displayed directly to the end user.
- NEVER use technical jargon, internal acronyms, or system module names such as "MAD", "MAD module", "RIS", "RIS module", "KR", "KR module", or "EP".
- Instead, express findings in natural language that any reader can understand:
  • Instead of "MAD" or "MAD module", write "Debate shows...", "The debate analysis indicates...", or "Multi-Agent Debate arguments show...".
  • Instead of "RIS" or "RIS module", write "Reverse Image Search shows...", "Visual verification indicates...", or "Image provenance checks reveal...".
  • Instead of "KR" or "KR module", write "External knowledge retrieval indicates...", "Verified news reports show...", or "Independent fact-checking reveals...".
  • Instead of "EP" or "EP module", write "Initial screening indicates..." or "Early Predictor signal shows...".
- The evidence ledgers are internal audit data. NEVER mention "claim ledger", "logic ledger", "claim_ledger", "logic_ledger", or the status codes "SUPPORTED", "CONTRADICTED", "NOT_COVERED" in your output. They are recorded in the forensic log only.
- NEVER cite evidence item numbers such as [S1], [S2] or [S4], and never use the words "snippet" or "snippet reference". Say "verified news reports", "a retrieved report", or "the reporting" instead.
- NEVER use publisher tier vocabulary — no "Tier 1", "Tier 2", "distinct publishers", "publisher statistics", or "corroboration strength". Say "independent outlets" or "multiple independent sources" instead.
- Express ledger outcomes as findings instead: "Verified news reports corroborate the reinstatement and the board's timeline, while the Bitcoin price detail remains unverified." Do not report ledger counts or status tallies.
Write with the objective, accessible tone of a senior investigative fact-checker.

Your output MUST be a single, valid JSON object formatted exactly as follows:
Important: Use markdown bolding (e.g. **key finding** or **critical claim**) to emphasize important phrases across "justification", "reasoning", and "key_facts".

{
  "reasoning": "<synthesized judicial deliberation detailing how the specialist findings were weighed, formatted as structured Markdown bullet points (e.g. '* **Factual Priority:** ...'). Do NOT write a wall of text. Use 1-2 detailed sentences per bullet point, strictly using user-friendly terms like 'Reverse Image Search shows...' or 'Debate shows...'. You MUST explicitly include dedicated bullet points for ALL pipeline components: Factual Assessment (which core claims verified reporting corroborates, contradicts, or leaves unverified — in plain language, never as a ledger table or status tally), Linguistic Assessment, Logical Assessment (internal consistency findings), Debate Analysis, Early Predictor (initial screening), External Knowledge Retrieval (verified news), and Reverse Image Search (visual provenance). Summarize their results transparently, even if they contradict your final verdict.>",
  "verdict": "SATIRE OR PARODY" | "FALSE CONNECTION" | "MISLEADING CONTENT" | "IMPOSTER CONTENT" | "MANIPULATED CONTENT" | "FABRICATED CONTENT" | "CREDIBLE" | "INCONCLUSIVE",
  "color_class": "satire" | "false-connection" | "misleading" | "imposter" | "manipulated" | "fabricated" | "credible" | "inconclusive",
  "confidence": "<number>%",
  "justification": "<4–6 sentence comprehensive explanation of your final verdict, detailing the main evidence and your reasoning clearly, using **bold** for critical signals and key entities, strictly using user-friendly terms like 'Reverse Image Search shows...' or 'Debate shows...'>",
  "key_facts": ["<observation 1 with **key points** bolded>", "<observation 2>", "<observation 3>"]
}
"""


# ===========================================================================
# Helper: User-Friendly Text Sanitizer
# ===========================================================================

def _clean_user_facing_text(text: str) -> str:
    """
    Sanitize text to replace internal acronyms and module references
    (MAD, RIS, KR, EP) with user-friendly terminology.
    """
    if not text:
        return text

    # Remove parenthetical acronym definitions like (RIS), (MAD), (KR), (EP)
    text = re.sub(r"\s*\(\s*(?:RIS|MAD|KR|EP)\s*\)", "", text)

    # Replace explicit module references (case-insensitive)
    text = re.sub(r"(?i)\bthe\s+mad\s+module\b", "the debate analysis", text)
    text = re.sub(r"(?i)\bmad\s+module\b", "debate analysis", text)
    text = re.sub(r"(?i)\bthe\s+ris\s+module\b", "the Reverse Image Search", text)
    text = re.sub(r"(?i)\bris\s+module\b", "Reverse Image Search", text)
    text = re.sub(r"(?i)\bthe\s+kr\s+module\b", "external knowledge retrieval", text)
    text = re.sub(r"(?i)\bkr\s+module\b", "external knowledge retrieval", text)
    text = re.sub(r"(?i)\bthe\s+ep\s+module\b", "the Early Predictor", text)
    text = re.sub(r"(?i)\bep\s+module\b", "Early Predictor", text)

    # Specific common verb / noun patterns
    text = re.sub(r"\bMAD\s+(shows|showed|indicates|indicated|concluded|found|suggests|suggested|argued|argues)\b", r"Debate \1", text)
    text = re.sub(r"\bRIS\s+(shows|showed|indicates|indicated|concluded|found|suggests|suggested|confirmed|confirms)\b", r"Reverse Image Search \1", text)
    text = re.sub(r"\bKR\s+(shows|showed|indicates|indicated|concluded|found|suggests|suggested|confirmed|confirms)\b", r"External knowledge retrieval \1", text)
    text = re.sub(r"\bEP\s+(shows|showed|indicates|indicated|concluded|found|suggests|suggested)\b", r"Early Predictor \1", text)

    # "According to MAD / RIS / KR / EP"
    text = re.sub(r"\bAccording to MAD\b", "According to the debate", text)
    text = re.sub(r"\bAccording to RIS\b", "According to Reverse Image Search", text)
    text = re.sub(r"\bAccording to KR\b", "According to external news reports", text)
    text = re.sub(r"\bAccording to EP\b", "According to initial screening", text)

    # "the MAD", "the RIS", "the KR", "the EP"
    text = re.sub(r"\bthe\s+MAD\b", "the debate analysis", text)
    text = re.sub(r"\bthe\s+RIS\b", "the Reverse Image Search", text)
    text = re.sub(r"\bthe\s+KR\b", "the external knowledge", text)
    text = re.sub(r"\bthe\s+EP\b", "the Early Predictor", text)

    # Standalone acronyms
    text = re.sub(r"\bMAD\b", "Multi-Agent Debate", text)
    text = re.sub(r"\bRIS\b", "Reverse Image Search", text)
    text = re.sub(r"\bKR\b", "Knowledge Retrieval", text)
    text = re.sub(r"\bEP\b", "Early Predictor", text)

    # Evidence-ledger vocabulary is audit data for the forensic log, not for
    # the verdict card. Rewrite tallies into a plain-language finding, e.g.
    # "The claim ledger records 4 SUPPORTED claims, 0 CONTRADICTED claims,
    #  and 3 NOT_COVERED claims." -> "Retrieved reporting corroborates 4 core
    # claims, contradicts 0, and leaves 3 unverified."
    text = re.sub(
        r"(?i)\b(?:the\s+)?claim[\s_]ledger\s+"
        r"(?:records|lists|shows|reports|contains|has)\s+"
        r"(\d+)\s+SUPPORTED\s+claims?,\s*"
        r"(\d+)\s+CONTRADICTED\s+claims?,?\s*(?:and\s+)?"
        r"(\d+)\s+NOT[\s_]COVERED\s+claims?",
        lambda m: (
            f"Retrieved reporting corroborates {m.group(1)} core claim(s), "
            f"contradicts {m.group(2)}, and leaves {m.group(3)} unverified"
        ),
        text,
    )
    text = re.sub(r"(?i)\bclaim[\s_]ledger\b", "evidence review", text)
    text = re.sub(r"(?i)\blogic[\s_]ledger\b", "consistency review", text)
    text = re.sub(r"(?i)\bevidence[\s_]ledger\b", "evidence review", text)
    text = re.sub(r"\bNOT[\s_]COVERED\b", "unverified", text)
    text = re.sub(r"\bSUPPORTED\b", "corroborated", text)
    text = re.sub(r"\bCONTRADICTED\b", "contradicted", text)

    # Evidence item numbers ("[S4]", "(see [S1], [S2])") mean nothing to a reader.
    text = re.sub(r"\s*\(\s*(?:see\s+|cf\.\s*)?\[S\d+\][^()]*\)", "", text)
    text = re.sub(
        r"\s*(?:\[S\d+\]\s*(?:,\s*(?:and\s+|or\s+)?|and\s+|or\s+|&\s*)?)+",
        " ",
        text,
    )

    # "snippet" and publisher-tier vocabulary are internal evidence plumbing.
    text = re.sub(r"(?i)\bsnippet\s+ref(?:erence)?s?\b", "citation", text)
    text = re.sub(r"(?i)\bsnippets\b", "retrieved reports", text)
    text = re.sub(r"(?i)\bsnippet\b", "retrieved report", text)
    # Tier tallies in parentheses ("(Tier 1: 2 | Tier 2: 1)") are noise.
    text = re.sub(r"(?i)\(\s*tiers?\s*[123][^()]*\)", "", text)
    text = re.sub(
        r"(?i)\bcorroboration\s+strength\s*[:=]?\s*multi[\s_-]?publisher\b",
        "multiple independent sources",
        text,
    )
    text = re.sub(
        r"(?i)\bcorroboration\s+strength\s*[:=]?\s*none\b",
        "no independent corroboration",
        text,
    )
    text = re.sub(r"(?i)\bcorroboration\s+strength\b", "source corroboration", text)
    text = re.sub(r"(?i)\bdistinct\s+publishers?\b", "independent outlets", text)
    text = re.sub(r"(?i)\bpublisher\s+(?:stats|statistics|tiers?|count)\b", "source overview", text)
    text = re.sub(r"(?i)\btiered\s+publishers?\b", "independent outlets", text)
    text = re.sub(r"(?i)\bmulti[\s_-]?publisher\b", "multiple independent sources", text)
    text = re.sub(r"(?i)\btier[\s-]?[123]\b", "independent", text)
    text = re.sub(r"(?i)\bindependent\s+(?:independent|outlet)\b", "independent outlet", text)
    text = re.sub(r"(?i)\bretrieved\s+retrieved\b", "retrieved", text)

    # Tidy punctuation/spacing artifacts left by the removals above.
    text = re.sub(r"(?i)\b(?:by|from|in|per|see)\s*([.,;:])", r"\1", text)
    # A preposition orphaned by a removed citation ("corroborated by while …").
    text = re.sub(
        r"(?i)\b(?:by|from|in|per)\s+(?=(?:and|or|but|while|whereas|although|though)\b)",
        "",
        text,
    )
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Restore a capital letter where a substitution began a line/sentence.
    text = re.sub(r"(?m)^(\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)

    return text


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
        Maximum tokens in the LLM response (default 8192 — the reasoning
        model writes its chain of thought into the reply, so a small cap
        truncates the JSON before it is emitted).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ):
        self.api_key     = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model_name  = model_name or _MODEL_ID
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.fallback_models = _JUDGE_FALLBACKS

        # Set by _build_user_message / _call_api — readable after analyze()
        self.last_user_message: str = ""
        self.last_raw_response: str = ""

        if not self.api_key:
            raise ValueError(
                "[Judge] No API key found. "
                "Set the OPENROUTER_API_KEY environment variable "
                "or pass api_key= when constructing Judge()."
            )

        log.info(f"[Judge] Initialised — model: {self.model_name}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, llm_text: str, ep_result: dict, external_knowledge: str = "", ris_summary: str = "", mad_summary: str = "") -> str:
        """Format the user turn from article text, EP result, and external knowledge."""
        # Structured stage outputs are rendered to text before templating.
        def _as_text(value):
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            try:
                return json.dumps(value, indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(value)

        external_knowledge = _as_text(external_knowledge)
        ris_summary = _as_text(ris_summary)
        mad_summary = _as_text(mad_summary)

        msg = _USER_TEMPLATE.format(
            ep_label      = ep_result.get("label", "UNKNOWN"),
            ep_prob       = ep_result.get("probability", 0.5),
            ep_confidence = ep_result.get("confidence", "N/A"),
            external_knowledge = external_knowledge or "No external text knowledge provided.",
            ris_summary   = ris_summary or "No Reverse Image Search data provided.",
            mad_summary   = mad_summary or "No Multi-Agent Debate data provided.",
            llm_text      = llm_text[:_MAX_TEXT_CHARS],
        )
        self.last_user_message = msg
        return msg

    def _call_api(
        self,
        user_message: str,
        temperature: float | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        json_mode: bool = True,
    ) -> str:
        """POST to OpenRouter and return the raw content string with multi-model fallback.

        Asks for a JSON object so the model cannot spend its whole output
        budget narrating. A reply cut off by the token cap is retried once
        on the same model with double the budget before falling back.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://rahmad.local",
            "X-Title":       "RAHMAD Fake News Detector",
        }
        
        models_to_try = [self.model_name]
        for m in self.fallback_models:
            if m not in models_to_try:
                models_to_try.append(m)

        sys_prompt = system_prompt if system_prompt is not None else _SYSTEM_PROMPT

        last_err = None
        budget   = max_tokens or self.max_tokens

        for model in models_to_try:
            use_json_mode = json_mode
            attempt_msg   = user_message

            # Two attempts per model. The second runs only when the first reply
            # was cut off by the token cap (finish_reason == "length"); it keeps
            # the same budget and instead asks for a shorter answer, because the
            # usual cause is the model narrating rather than running out of room.
            for attempt in (1, 2):
                print(
                    f"  [Judge] Querying OpenRouter model '{model}'"
                    + (" (retry, shorter answer)..." if attempt > 1 else "...")
                )
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user",   "content": attempt_msg},
                    ],
                    "temperature": temperature if temperature is not None else self.temperature,
                    "max_tokens":  budget,
                }
                if use_json_mode:
                    # Force a JSON object. Without this the model narrates its
                    # chain of thought into the reply and never emits the JSON.
                    payload["response_format"] = {"type": "json_object"}

                log.info(f"[Judge] Sending request to OpenRouter — model: {model}")
                try:
                    response = requests.post(
                        _OPENROUTER_URL,
                        headers=headers,
                        json=payload,
                        timeout=120,
                    )

                    # Some models/providers reject structured output outright.
                    if response.status_code in (400, 422) and use_json_mode:
                        log.warning(
                            f"[Judge] {model} rejected structured output "
                            f"({response.text[:200]}); retrying without it."
                        )
                        print(f"  [Judge] [!] Structured output unsupported on '{model}'. Retrying in plain mode.")
                        use_json_mode = False
                        continue

                    response.raise_for_status()
                    content = self._extract_content(response.json(), model, budget)
                    self.last_raw_response = content
                    return content

                except JudgeTruncatedResponse as exc:
                    last_err = exc
                    log.warning(f"[Judge] {model}: {exc}")
                    print(f"  [Judge] [!] {exc} — retrying with a shorter-answer instruction.")
                    attempt_msg = (
                        user_message
                        + "\n\nIMPORTANT: Your previous reply hit the output limit before "
                        "finishing. Reply with the JSON object only — no preamble and no "
                        "visible reasoning. Keep 'justification' to 3-5 sentences, "
                        "'reasoning' to seven short bullets, and 'key_facts' to 3-5 items."
                    )
                    continue

                except Exception as exc:
                    log.warning(f"[Judge] Model {model} failed: {exc}")
                    print(f"  [Judge] [!] Model '{model}' failed or timed out. Trying fallback...")
                    last_err = exc
                    break

        raise RuntimeError(f"All OpenRouter LLM models failed. Last error: {last_err}")

    def _extract_content(self, resp_json: dict, model: str, max_tokens: int) -> str:
        """Pull the reply text from an OpenRouter response, rejecting cut-off replies."""
        choices = resp_json.get("choices") or []
        if not choices:
            raise JudgeResponseError(
                f"model {model} returned no choices: {str(resp_json)[:200]}"
            )

        choice  = choices[0]
        msg     = choice.get("message") or {}
        content = msg.get("content")
        if content is None:
            content = msg.get("reasoning") or ""
        content = str(content)
        finish  = choice.get("finish_reason")

        log.info(
            f"[Judge] Response received ({len(content)} chars, "
            f"finish_reason={finish}) from model {model}."
        )

        if finish == "length":
            raise JudgeTruncatedResponse(
                f"reply hit the {max_tokens}-token cap before finishing "
                f"(finish_reason=length, {len(content)} chars)"
            )
        if not content.strip():
            raise JudgeResponseError(
                f"model {model} returned an empty reply (finish_reason={finish})"
            )

        print(f"  [Judge] [OK] Response received from '{model}' ({len(content)} chars).")
        return content

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """
        Extract and validate the JSON verdict from the LLM response.
        Handles cases where the model wraps JSON in markdown code fences,
        emits chain-of-thought preambles, or suffers partial truncation.
        """
        # Remove <think>...</think> reasoning blocks common in modern LLMs
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        
        # Remove markdown code fences and whitespace
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip().rstrip("`").strip()

        data = None

        # 1. Try parsing directly
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 2. Search for candidate JSON object containing "verdict"
        if data is None or not isinstance(data, dict) or "verdict" not in data:
            v_matches = list(re.finditer(r'\{[^{}]*"verdict"[^{}]*\}', cleaned, re.DOTALL))
            for m in reversed(v_matches):
                try:
                    candidate = json.loads(m.group(0))
                    if isinstance(candidate, dict) and "verdict" in candidate:
                        data = candidate
                        break
                except json.JSONDecodeError:
                    pass

        # 3. Greedy balanced-brace extraction around "verdict"
        if data is None:
            v_pos = cleaned.rfind('"verdict"')
            if v_pos != -1:
                start_idx = cleaned.rfind("{", 0, v_pos)
                if start_idx != -1:
                    sub = cleaned[start_idx:]
                    for end in range(len(sub), 0, -1):
                        if sub[end - 1] == '}':
                            try:
                                candidate = json.loads(sub[:end])
                                if isinstance(candidate, dict) and "verdict" in candidate:
                                    data = candidate
                                    break
                            except json.JSONDecodeError:
                                pass

        # 4. Salvage auto-close truncated JSON
        if data is None:
            start_idx = cleaned.find("{")
            if start_idx != -1:
                substring = cleaned[start_idx:]
                for cutoff in range(len(substring), 0, -1):
                    candidate = substring[:cutoff].rstrip(", \n\r\t")
                    open_braces = candidate.count("{") - candidate.count("}")
                    if open_braces > 0:
                        quotes = candidate.count('"') - candidate.count(r'\"')
                        if quotes % 2 != 0:
                            candidate += '"'
                        candidate += "}" * open_braces
                    try:
                        cand_data = json.loads(candidate)
                        if isinstance(cand_data, dict) and "verdict" in cand_data:
                            data = cand_data
                            break
                    except json.JSONDecodeError:
                        continue

        # 5. No prose salvage. A reply without a JSON verdict is rejected so the
        #    caller can retry with a stricter prompt. Scraping prose used to match
        #    verdict words the model merely quoted from its own constitution
        #    (e.g. "... classify as CREDIBLE.") and report a verdict it never reached.
        if data is None or not isinstance(data, dict) or "verdict" not in data:
            raise JudgeResponseError(
                f"No JSON verdict in the LLM reply ({len(raw)} chars) — the reply "
                f"was most likely truncated. Start of reply: {raw[:200]!r}"
            )

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

        justification = _clean_user_facing_text(str(data.get("justification", "")))
        reasoning = _clean_user_facing_text(str(data.get("reasoning", "")))
        clean_facts = [_clean_user_facing_text(str(fact)) for fact in key_facts]

        # The UI renders "reasoning" and "key_facts" directly. A verdict missing
        # either one means the reply was cut short — retry rather than ship
        # half-populated panels.
        missing = []
        if not reasoning.strip():
            missing.append("reasoning")
        if not clean_facts:
            missing.append("key_facts")
        if missing:
            raise JudgeResponseError(
                f"Verdict JSON parsed but {', '.join(missing)} empty — the reply is incomplete."
            )

        return {
            "verdict":       verdict,
            "color_class":   color_class,
            "confidence":    str(data.get("confidence", "50.0%")),
            "justification": justification,
            "reasoning":     reasoning,   # CoT scratchpad
            "key_facts":     list(clean_facts),
            "links":         [],
        }

    def _call_and_parse(
        self,
        user_message: str,
        temperature: float | None = None,
        system_prompt: str | None = None,
        context: str = "Judge",
    ) -> dict:
        """Call the API and return a verified verdict, retrying once on a bad reply.

        A reply that is truncated, empty, or missing the JSON verdict raises
        instead of being scraped for keywords. The retry repeats the same
        message plus an explicit "JSON only" nudge, because the usual cause is
        a model that narrated instead of answering.
        """
        last_err: Exception | None = None

        for attempt in (1, 2):
            raw = self._call_api(user_message, temperature, system_prompt)
            try:
                return self._parse_response(raw)
            except Exception as exc:
                last_err = exc
                log.warning(
                    f"[Judge][{context}] Attempt {attempt}/2 produced no valid "
                    f"verdict: {exc}"
                )
                print(f"  [Judge] [{context}] [!] Invalid reply ({exc}). Retrying with a stricter prompt...")
                user_message = (
                    user_message
                    + "\n\nIMPORTANT: Your previous reply could not be parsed. Reply with ONLY "
                    "the JSON object — no preamble, no markdown fences, no visible "
                    'reasoning — and include the "reasoning" and "key_facts" fields.'
                )

        raise RuntimeError(
            f"[Judge][{context}] No valid JSON verdict after 2 attempts. "
            f"Last error: {last_err}"
        )

    @staticmethod
    def _parse_specialist(raw: str) -> dict:
        """Parse a specialist judge's JSON reply, rejecting cut-off or empty payloads."""
        clean = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
        clean = re.sub(r"```(?:json)?\s*", "", clean).strip().rstrip("`").strip()

        parsed = None
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            start = clean.find("{")
            end   = clean.rfind("}")
            if start != -1 and end > start:
                try:
                    parsed = json.loads(clean[start:end + 1])
                except json.JSONDecodeError:
                    parsed = None

        if not isinstance(parsed, dict) or not parsed:
            raise JudgeResponseError(
                f"no usable JSON object in the specialist reply "
                f"({len(raw or '')} chars): {(raw or '')[:160]!r}"
            )
        return parsed


    @staticmethod
    def _validate_claim_ledger(claim_ledger: list, snippet_refs: set, external_knowledge: str = "") -> tuple[list, dict]:
        """
        Validate the Fact-Checker's claim_ledger.
        Returns (validated_ledger, validation_stats).
        - CONTRADICTED requires evidence_ref that exists in snippet_refs and shares tokens with claim.
        - Invalid CONTRADICTED -> downgraded to NOT_COVERED.
        - SUPPORTED requires evidence_ref in snippet_refs.
        - AUTO-REPAIR: if status is SUPPORTED but evidence_ref is missing/empty,
          attempt to match claim to snippet content from external_knowledge.
        """
        # Parse snippet content from external_knowledge for auto-repair
        snippet_content = {}
        if external_knowledge:
            # Pattern: [S1] snippet text\n[S2] snippet text\n...
            # Use non-greedy match until next [S#] or === boundary
            for m in re.finditer(r"\[S(\d+)\]\s*(.*?)(?=\n\s*\[S\d+\]|\n\s*===|\Z)", external_knowledge, re.DOTALL):
                ref = f"[S{m.group(1)}]"
                content = m.group(2).strip()
                if content:
                    snippet_content[ref] = content
        
        def _auto_match_claim_to_snippets(claim: str, snippet_content: dict) -> list:
            """Find snippet refs whose content genuinely supports the claim.

            Matching is deliberately conservative: a snippet must share at least
            two *distinctive* claim tokens and cover at least 30% of them. That
            rejects snippets that merely repeat an entity name (every Nvidia
            snippet mentions Nvidia/Huang/board), while still accepting real
            corroboration phrased differently.
            """
            if not snippet_content:
                return []

            # Tokens of 3+ chars keep numbers and short but meaningful words
            # ("six", "nov", "ceo") that a 4-char filter would drop.
            claim_tokens = set(re.findall(r"[a-z0-9]{3,}", claim.lower()))

            # Words shared by nearly every news snippet about the subject. They
            # cannot distinguish corroboration from incidental overlap.
            generic_tokens = {
                "nvidia", "huang", "jensen", "board", "said", "says", "reported",
                "reports", "announced", "today", "company", "following",
                "immediately", "effective", "according", "post", "news", "after",
                "before", "that", "this", "these", "those", "with", "from",
                "will", "been", "have", "has", "had", "was", "were", "are",
                "also", "into", "over", "more", "than", "such", "when", "what",
                "which", "their", "there", "about", "would", "could", "should",
                "being", "other", "remain", "remains", "continue", "continues",
                "now", "new",
                # Function words: never evidence of corroboration, but they
                # inflate overlap and let unrelated snippets match.
                "the", "and", "for", "but", "not", "you", "all", "can", "her",
                "his", "out", "one", "get", "use", "may", "its", "our", "who",
                "how", "why", "did", "does", "them", "they", "then", "too",
                "very", "just", "only", "any", "each", "some", "most", "much",
                "many", "both", "onto", "per", "via", "yet", "nor", "off",
            }
            distinctive = claim_tokens - generic_tokens
            base_tokens = distinctive if len(distinctive) >= 3 else claim_tokens

            scored_matches = []
            for ref, content in snippet_content.items():
                content_tokens = set(re.findall(r"[a-z0-9]{3,}", content.lower()))
                overlap = base_tokens & content_tokens
                # Three shared distinctive tokens, or two when the claim is
                # short enough that two is most of it. Two shared words out
                # of five ("chief officer") is coincidental phrasing, not
                # corroboration.
                if len(overlap) < 2:
                    continue
                if len(overlap) < 3 and len(base_tokens) > 4:
                    continue
                ratio = len(overlap) / max(1, len(base_tokens))
                if ratio >= 0.30:
                    scored_matches.append((ratio, len(overlap), ref))

            # Best match first, at most two refs per claim.
            scored_matches.sort(reverse=True)
            return [ref for _, _, ref in scored_matches[:2]]

        validated = []
        stats = {"supported": 0, "contradicted": 0, "not_covered": 0, "downgraded": 0, "missing_refs": 0, "auto_repaired": 0}
        
        for entry in claim_ledger:
            if not isinstance(entry, dict):
                continue
            claim = str(entry.get("claim") or "").strip()
            status = str(entry.get("status") or "NOT_COVERED").upper()
            evidence_ref = str(entry.get("evidence_ref") or "").strip()
            notes = str(entry.get("notes") or "")
            
            if status == "SUPPORTED":
                if evidence_ref and any(ref in evidence_ref for ref in snippet_refs):
                    validated.append({"claim": claim, "status": "SUPPORTED", "evidence_ref": evidence_ref, "notes": notes})
                    stats["supported"] += 1
                else:
                    # AUTO-REPAIR: try to find matching snippets from external_knowledge
                    matched_refs = _auto_match_claim_to_snippets(claim, snippet_content)
                    if matched_refs:
                        repaired_ref = ", ".join(matched_refs)
                        validated.append({"claim": claim, "status": "SUPPORTED", "evidence_ref": repaired_ref, "notes": notes + " [auto-matched snippet ref]"})
                        stats["supported"] += 1
                        stats["auto_repaired"] += 1
                    else:
                        validated.append({"claim": claim, "status": "NOT_COVERED", "evidence_ref": "", "notes": notes + " [missing valid snippet ref]"})
                        stats["not_covered"] += 1
                        stats["missing_refs"] += 1
            elif status == "CONTRADICTED":
                # Validate: evidence_ref must exist in snippet_refs
                if evidence_ref and any(ref in evidence_ref for ref in snippet_refs):
                    validated.append({"claim": claim, "status": "CONTRADICTED", "evidence_ref": evidence_ref, "notes": notes})
                    stats["contradicted"] += 1
                else:
                    # Downgrade invalid CONTRADICTED to NOT_COVERED
                    validated.append({"claim": claim, "status": "NOT_COVERED", "evidence_ref": "", "notes": notes + " [invalid contradiction ref; downgraded]"})
                    stats["not_covered"] += 1
                    stats["downgraded"] += 1
            else:
                # AUTO-REPAIR for NOT_COVERED: if notes indicate corroboration/confirmation,
                # try to match claim to snippet content and upgrade to SUPPORTED
                notes_lower = notes.lower()
                corroboration_indicators = ["confirm", "corroborat", "support", "match", "track", "appear in", "consistent with", "aligns with", "evidence shows", "reports show", "listed", "mentioned", "described in", "documented"]
                if any(indicator in notes_lower for indicator in corroboration_indicators):
                    matched_refs = _auto_match_claim_to_snippets(claim, snippet_content)
                    if matched_refs:
                        repaired_ref = ", ".join(matched_refs)
                        validated.append({"claim": claim, "status": "SUPPORTED", "evidence_ref": repaired_ref, "notes": notes + " [auto-upgraded from NOT_COVERED via note corroboration]"})
                        stats["supported"] += 1
                        stats["auto_repaired"] += 1
                    else:
                        validated.append({"claim": claim, "status": "NOT_COVERED", "evidence_ref": "", "notes": notes})
                        stats["not_covered"] += 1
                else:
                    validated.append({"claim": claim, "status": "NOT_COVERED", "evidence_ref": "", "notes": notes})
                    stats["not_covered"] += 1
        
        return validated, stats

    @staticmethod
    def _compute_factual_risk_score(validated_ledger: list, validation_stats: dict, image_assessment: str) -> int:
        """
        Compute factual risk score from the validated ledger (code, not model).

        Absence of coverage is NOT evidence of fabrication, so NOT_COVERED
        contributions are capped and cannot dominate the score.

        Base: 3
        + CONTRADICTED core claim: +3 each (max +6)
        + NOT_COVERED core claim:  +1 each (max +3)
        - SUPPORTED core claim:    -2 each (max -6)
        + Image out-of-context: +4
        + Image recycled:       +3
        + Image unverified:     +1
        Final clamped to 1-10.
        """
        contradicted = sum(1 for e in validated_ledger if e.get("status") == "CONTRADICTED")
        not_covered = sum(1 for e in validated_ledger if e.get("status") == "NOT_COVERED")
        supported = sum(1 for e in validated_ledger if e.get("status") == "SUPPORTED")

        score = 3  # base
        score += min(6, contradicted * 3)
        score += min(3, not_covered)
        score -= min(6, supported * 2)

        img = (image_assessment or "").lower()
        if "out-of-context" in img or "out of context" in img:
            score += 4
        elif "recycled" in img:
            score += 3
        elif "unverified" in img:
            score += 1

        return max(1, min(10, score))

    @staticmethod
    def _validate_logic_ledger(logic_ledger: list, snippet_refs: set) -> tuple[list, dict]:
        """
        Validate the Logician's logic_ledger.
        CONTRADICTED/FALLACY requires evidence_ref in snippet_refs.
        UNVERIFIED means no snippet supports the attribution.
        Returns (validated_ledger, validation_stats).
        """
        validated = []
        stats = {"contradicted": 0, "unverified": 0, "fallacy": 0, "downgraded": 0}
        
        for entry in logic_ledger:
            issue = entry.get("issue", "").strip()
            status = entry.get("status", "UNVERIFIED").upper()
            evidence_ref = entry.get("evidence_ref", "").strip()
            notes = entry.get("notes", "")
            
            if status in ("CONTRADICTED", "FALLACY"):
                if evidence_ref and any(ref in evidence_ref for ref in snippet_refs):
                    validated.append({"issue": issue, "status": status, "evidence_ref": evidence_ref, "notes": notes})
                    stats[status.lower()] += 1
                else:
                    validated.append({"issue": issue, "status": "UNVERIFIED", "evidence_ref": "", "notes": notes + " [invalid ref; downgraded]"})
                    stats["unverified"] += 1
                    stats["downgraded"] += 1
            else:  # UNVERIFIED
                validated.append({"issue": issue, "status": "UNVERIFIED", "evidence_ref": "", "notes": notes})
                stats["unverified"] += 1
        
        return validated, stats

    @staticmethod
    def _compute_logical_risk_score(validated_logic_ledger: list, validation_stats: dict) -> int:
        """
        Compute logical risk score from validated logic ledger.
        Scale:
        + CONTRADICTED: +3 each
        + FALLACY: +2 each
        + UNVERIFIED: +1 each
        Base: 2
        Clamped 1-10.
        """
        score = 2
        for entry in validated_logic_ledger:
            status = entry.get("status", "UNVERIFIED")
            if status == "CONTRADICTED":
                score += 3
            elif status == "FALLACY":
                score += 2
            elif status == "UNVERIFIED":
                score += 1
        return max(1, min(10, score))

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

    @staticmethod
    def _enforce_constitutional_guardrails(result: dict, ris_summary: str, ep_result: dict, external_knowledge: str = "") -> dict:
        """
        Constitutional Principle P8 Enforcement:
        A factually accurate text narrative never excuses a decontextualized / recycled image.
        If Reverse Image Search confirms 'Out-of-Context Imagery Detected: YES (Decontextualized / Misattributed)'
        with High confidence (or Medium confidence when Early Predictor flags Fake), an article cannot be CREDIBLE.
        Attaching an unrelated or recycled photo to news text is the definition of FALSE CONNECTION (Cheapfake).

        Exception/Guard: If the Judge explicitly acknowledged the visual ambiguity and cited verified external knowledge
        corroborating the claim and authentic photo, trust the deliberate judicial reasoning.
        """
        if not ris_summary or not isinstance(result, dict):
            return result

        verdict = result.get("verdict", "")
        if verdict == "CREDIBLE":
            summary_lower = ris_summary.lower()
            is_ooc = "out-of-context imagery detected: yes" in summary_lower

            if is_ooc:
                reasoning     = str(result.get("reasoning", "")).lower()
                justification = str(result.get("justification", "")).lower()
                combined_text = reasoning + " " + justification

                # 1) Did the Judge acknowledge the RIS out-of-context finding?
                ris_ack_phrases = [
                    "reverse image search", "visual verification",
                    "image provenance", "out-of-context", "out of context",
                    "recycled", "misattributed", "decontextualized",
                    "ris", "visual forensic",
                ]
                judge_acknowledged_ris = any(
                    phrase in combined_text for phrase in ris_ack_phrases
                )

                # 2) Did the Judge cite external knowledge / fact-checks that confirmed authenticity?
                kr_confirm_phrases = [
                    "external knowledge", "knowledge retrieval",
                    "verified news", "confirms", "confirmed",
                    "corroborated", "fact-check", "fact check",
                    "snopes", "reuters", "associated press",
                    "verified", "authentic", "genuine",
                    "credible source", "reliable source",
                    "news reports", "documented", "well-documented",
                ]
                judge_cited_kr_confirmation = any(
                    phrase in combined_text for phrase in kr_confirm_phrases
                )

                if judge_acknowledged_ris and judge_cited_kr_confirmation:
                    log.info(
                        "[Judge] Constitutional Principle P8: Judge explicitly "
                        "considered RIS out-of-context finding but concluded "
                        "CREDIBLE based on stronger verified external evidence. "
                        "Trusting deliberate judicial reasoning."
                    )
                else:
                    log.warning(
                        "[Judge] Constitutional Principle P8 Enforcement: "
                        "Overriding CREDIBLE to FALSE CONNECTION — Judge did "
                        "not explicitly address the RIS out-of-context finding "
                        "with countervailing verified evidence."
                    )
                    result["verdict"] = "FALSE CONNECTION"
                    result["color_class"] = "false-connection"
                    p8_note = (
                        " **[Constitutional Principle P8 Guardrail]**: While "
                        "the textual claims appear plausible or grounded, "
                        "Reverse Image Search forensic checks confirmed that "
                        "the accompanying visual is decontextualized or "
                        "recycled from an unrelated event. Under multimodal "
                        "fact-checking standards, attaching an unrelated image "
                        "to news claims constitutes a **FALSE CONNECTION** "
                        "(Cheapfake)."
                    )
                    result["justification"] = (
                        result.get("justification", "").strip() + p8_note
                    ).strip()
                    facts = result.get("key_facts", [])
                    if isinstance(facts, list):
                        facts.append(
                            "Visual forensic analysis confirmed out-of-context "
                            "imagery, resulting in a **False Connection** verdict."
                        )
                        result["key_facts"] = facts

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, llm_text: str, ep_result: dict, external_knowledge: str = "", ris_summary: str = "", mad_summary: str = "") -> dict:
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
        external_knowledge : str
            Verified external knowledge retrieved for the claim.

        Returns
        -------
        dict
            verdict, color_class, confidence, justification, reasoning,
            key_facts, links, ep_signal.
        """
        self.last_raw_response = ""
        self.last_user_message = ""
        try:
            user_message = self._build_user_message(llm_text, ep_result, external_knowledge, ris_summary, mad_summary)
            result       = self._call_and_parse(user_message, context="Single")
            result       = self._enforce_constitutional_guardrails(result, ris_summary, ep_result)

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
        external_knowledge: str = "",
        ris_summary: str = "",
        mad_summary: str = "",
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
        external_knowledge : str
            Verified external knowledge retrieved for the claim.

        Returns
        -------
        dict
            Aggregated verdict dict. Includes 'sc_votes' key showing the
            full vote breakdown across all N runs for transparency.
        """
        user_message = self._build_user_message(llm_text, ep_result, external_knowledge, ris_summary, mad_summary)

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
                    self._call_and_parse,
                    user_message,
                    sc_temperature,
                    None,
                    f"SC run {i + 1}",
                ): i
                for i in range(n)
            }
            for future in as_completed(futures):
                run_idx = futures[future]
                try:
                    parsed = future.result()
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
        aggregated = self._enforce_constitutional_guardrails(aggregated, ris_summary, ep_result)
        aggregated["ep_signal"] = ep_result
        return aggregated

    def panel_analyze(
        self,
        llm_text: str,
        ep_result: dict,
        external_knowledge: str = "",
        ris_summary: str = "",
        mad_summary: str = "",
    ) -> dict:
        """
        Multi-Judge Panel analysis:
        Executes 3 specialized judicial agents concurrently:
          1. Fact-Checker  (External KR evidence, Reverse Image Search provenance, Multi-Agent Debate)
          2. Linguist      (Stylometrics, emotional manipulation, sensationalism, rhetoric)
          3. Logician      (Internal consistency, source attribution plausibility, communicative intent)
        Then feeds their qualitative reports and risk scores to the Chief Judicial Magistrate
        for final synthesis.

        Parameters
        ----------
        llm_text : str
            Cleaned full article text.
        ep_result : dict
            Output of EarlyPredictor.predict(). Pass {} if EP was skipped.
        external_knowledge : str
            Verified external knowledge retrieved for the claim.
        ris_summary : str
            Reverse Image Search visual provenance summary.
        mad_summary : str
            Multi-Agent Debate argument summary.

        Returns
        -------
        dict
            Synthesized verdict matching frontend schema, plus 'panel_details'
            containing assessments from all 3 specialists.
        """
        self.last_raw_response = ""
        self.last_user_message = ""
        user_message = self._build_user_message(
            llm_text, ep_result, external_knowledge, ris_summary, mad_summary
        )

        log.info("[Judge][Panel] Starting Multi-Judge Panel analysis with 3 specialists…")
        print("  [Judge Panel] Dispatching to 3 Specialized Judges concurrently...")

        specialist_tasks = {
            "fact_checker": (_FACT_CHECKER_PROMPT, "Senior Fact-Checker"),
            "linguist":     (_LINGUIST_PROMPT, "Senior Linguist"),
            "logician":     (_LOGICIAN_PROMPT, "Senior Logician"),
        }

        specialist_results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_role = {
                executor.submit(self._call_api, user_message, 0.2, prompt): (role, label, prompt)
                for role, (prompt, label) in specialist_tasks.items()
            }
            for future in as_completed(future_to_role):
                role, label, prompt = future_to_role[future]
                try:
                    raw_text = future.result()
                    try:
                        clean_raw = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                        clean_raw = re.sub(r"```(?:json)?\s*", "", clean_raw).strip().rstrip("`").strip()
                        specialist_results[role] = self._parse_specialist(clean_raw)
                    except Exception as parse_exc:
                        # Truncated specialist replies used to be flattened to
                        # {"summary": <raw text>}, silently destroying the risk
                        # score and structured findings the chief judge needs.
                        log.warning(f"[Judge Panel] {label} reply unusable: {parse_exc} — retrying once.")
                        print(f"  [Judge Panel] [!] {label} reply unusable ({parse_exc}). Retrying...")
                        retry_raw = self._call_api(
                            user_message
                            + "\n\nIMPORTANT: Reply with ONLY the JSON object for your "
                              "specialist assessment — no preamble, no markdown fences.",
                            0.2,
                            prompt,
                        )
                        try:
                            specialist_results[role] = self._parse_specialist(retry_raw)
                        except Exception as retry_exc:
                            log.warning(f"[Judge Panel] {label} failed twice: {retry_exc}")
                            print(f"  [Judge Panel] [!] {label} failed twice ({retry_exc}).")
                            specialist_results[role] = {
                                "error": f"specialist reply unusable: {retry_exc}",
                                "raw_excerpt": (retry_raw or "")[:400],
                            }
                    print(f"  [Judge Panel] [OK] {label} completed assessment.")
                except Exception as exc:
                    log.warning(f"[Judge Panel] Specialist {label} failed: {exc}")
                    print(f"  [Judge Panel] [!] Specialist {label} failed: {exc}")
                    specialist_results[role] = {"error": str(exc), "risk_score": 5}

        # Sanitize specialist findings so they don't contain raw acronyms
        for role, data in specialist_results.items():
            if isinstance(data, dict):
                for key in ["factual_analysis", "stylistic_analysis", "logical_analysis", "summary"]:
                    if key in data and isinstance(data[key], str):
                        data[key] = _clean_user_facing_text(data[key])
                for key in ["corroborated_claims", "refuted_or_unsubstantiated_claims", "fallacies_or_contradictions"]:
                    if key in data and isinstance(data[key], list):
                        data[key] = [_clean_user_facing_text(str(item)) for item in data[key]]

        # ===== EVIDENCE-LEDGER VALIDATION & CODE-BASED RISK SCORING =====
        # Extract snippet references from external_knowledge (format: [S1], [S2], etc.)
        snippet_refs = set(re.findall(r"\[S\d+\]", external_knowledge))
        
        # Validate Fact-Checker's claim_ledger
        fc_result = specialist_results.get("fact_checker", {})
        claim_ledger = fc_result.get("claim_ledger", [])
        if claim_ledger:
            validated_ledger, fc_stats = self._validate_claim_ledger(claim_ledger, snippet_refs, external_knowledge)
            fc_result["claim_ledger"] = validated_ledger
            fc_result["ledger_stats"] = fc_stats
            # Compute factual risk score in code
            image_assessment = fc_result.get("image_assessment", "")
            factual_risk = self._compute_factual_risk_score(validated_ledger, fc_stats, image_assessment)
            fc_result["factual_risk_score"] = factual_risk
        else:
            # Legacy format fallback
            fc_result["factual_risk_score"] = fc_result.get("factual_risk_score", 5)
        
        # Validate Logician's logic_ledger
        log_result = specialist_results.get("logician", {})
        logic_ledger = log_result.get("logic_ledger", [])
        if logic_ledger:
            validated_logic, log_stats = self._validate_logic_ledger(logic_ledger, snippet_refs)
            log_result["logic_ledger"] = validated_logic
            log_result["ledger_stats"] = log_stats
            logical_risk = self._compute_logical_risk_score(validated_logic, log_stats)
            log_result["logical_risk_score"] = logical_risk
        else:
            log_result["logical_risk_score"] = log_result.get("logical_risk_score", 5)

        chief_user_message = (
            "=== SPECIALIST JUDGE PANEL FINDINGS ===\n\n"
            f"--- 1. FACT-CHECKER ASSESSMENT (External Evidence, Reverse Image Search, Debate) ---\n"
            f"{json.dumps(specialist_results.get('fact_checker', {}), indent=2)}\n\n"
            f"--- 2. LINGUIST ASSESSMENT (Stylometrics, Emotion, Manipulation) ---\n"
            f"{json.dumps(specialist_results.get('linguist', {}), indent=2)}\n\n"
            f"--- 3. LOGICIAN ASSESSMENT (Internal Logic, Sourcing, Intent) ---\n"
            f"{json.dumps(specialist_results.get('logician', {}), indent=2)}\n\n"
            f"=== ORIGINAL CONTEXT & ARTICLE ===\n"
            f"{user_message}\n\n"
            "Now synthesize all specialist evaluations, resolve any conflicts, "
            "and deliver your final judicial verdict as JSON.\n"
            "CRITICAL AUDIENCE REMINDER: Write your reasoning, justification, and key facts in clear, user-friendly language. "
            "Do NOT use technical acronyms like MAD, RIS, KR, or EP in your output text. Instead, use natural phrases such as "
            "'Debate shows...', 'Reverse Image Search shows...', 'External knowledge retrieval shows...', and 'Early Predictor signal indicates...'."
        )

        print("  [Judge Panel] Chief Judicial Magistrate synthesizing panel findings...")
        log.info("[Judge][Panel] Querying Chief Judge for final synthesis…")

        try:
            result = self._call_and_parse(
                chief_user_message,
                temperature=0.2,
                system_prompt=_CHIEF_JUDGE_PROMPT,
                context="Panel Chief",
            )
            result = self._enforce_constitutional_guardrails(result, ris_summary, ep_result)
            result["ep_signal"] = ep_result
            result["panel_details"] = specialist_results
            self.last_user_message = chief_user_message
            log.info(f"[Judge][Panel] Final verdict: {result['verdict']} ({result['confidence']})")
            return result
        except Exception as exc:
            log.error(f"[Judge][Panel] Chief Judge synthesis failed: {exc}", exc_info=True)
            result = self._fallback_response(f"Chief Judge failed: {exc}")
            result["ep_signal"] = ep_result
            result["panel_details"] = specialist_results
            return result

    def save_judge_log(
        self,
        result: dict,
        mode: str,
        llm_text: str,
        ep_result: dict,
        external_knowledge: str = "",
        ris_summary: str = "",
        mad_summary: str = "",
        runtime: float | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Log complete judicial findings, mode, reasoning, specialist assessments,
        and verdict into JUDGE_output.txt inside the output/ folder.
        """
        if output_path is None:
            project_root = Path(__file__).resolve().parent.parent
            output_dir = project_root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "JUDGE_output.txt"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        lines = []
        sep_major = "=" * 80
        sep_minor = "-" * 60

        lines.append(sep_major)
        lines.append(f"  JUDGE MODULE (CLOUD LLM REASONING LAYER) FORENSIC LOG  —  {now_str}")
        lines.append(sep_major)
        lines.append("")

        lines.append("[1]  CONFIGURATION & CONTEXT AUDIT")
        lines.append(sep_minor)
        lines.append(f"    Active Architecture Pipeline : {mode.upper()}")
        lines.append(f"    Primary LLM Model            : {self.model_name}")
        lines.append(f"    Fallback Model Chain         : {', '.join(self.fallback_models)}")
        if runtime is not None:
            lines.append(f"    Execution Runtime            : {runtime:.2f}s")
        lines.append("")
        lines.append("    Early Predictor Multimodal Context:")
        lines.append(f"      Label       : {ep_result.get('label', 'N/A')}")
        lines.append(f"      Probability : {ep_result.get('probability', 'N/A')}")
        lines.append(f"      Confidence  : {ep_result.get('confidence', 'N/A')}")
        lines.append("")
        lines.append("    External Context Ingestion Status:")
        lines.append(f"      Knowledge Retrieval (KR) : {'YES' if external_knowledge else 'NO'}")
        lines.append(f"      Reverse Image Search (RIS): {'YES' if ris_summary else 'NO'}")
        lines.append(f"      Multi-Agent Debate (MAD) : {'YES' if mad_summary else 'NO'}")
        lines.append("")

        lines.append("[2]  ORIGINAL NEWS ARTICLE (ANALYZED TEXT)")
        lines.append(sep_minor)
        for a_line in llm_text.strip().splitlines():
            lines.append(f"    {a_line}")
        lines.append("")

        # Mode-specific sections
        if mode == "panel" and "panel_details" in result:
            lines.append("[3]  MULTI-JUDGE SPECIALIZED PANEL REPORTS")
            lines.append(sep_minor)
            details = result.get("panel_details", {})
            
            lines.append("  (A) SENIOR FACT-CHECKER ASSESSMENT (KR, RIS, MAD Provenance):")
            fact_data = details.get("fact_checker", {})
            if isinstance(fact_data, dict):
                lines.append(f"      Factual Risk Score : {fact_data.get('factual_risk_score', 'N/A')} / 10")
                lines.append(f"      Analysis           : {fact_data.get('factual_analysis', '')}")
                stats = fact_data.get("ledger_stats") or {}
                if stats:
                    lines.append(
                        "      Risk Basis         : "
                        f"{stats.get('supported', 0)} supported, "
                        f"{stats.get('contradicted', 0)} contradicted, "
                        f"{stats.get('not_covered', 0)} not covered"
                    )
                claim_ledger = fact_data.get("claim_ledger") or []
                if claim_ledger:
                    lines.append("      Claim Ledger       :")
                    for entry in claim_ledger:
                        if not isinstance(entry, dict):
                            continue
                        status = entry.get("status", "NOT_COVERED")
                        claim = str(entry.get("claim", "")).strip()
                        ref = str(entry.get("evidence_ref", "")).strip()
                        suffix = f"  {ref}" if ref else ""
                        lines.append(f"        - [{status}] {claim}{suffix}")
                if fact_data.get("corroborated_claims"):
                    lines.append(f"      Corroborated Claims: {', '.join(str(c) for c in fact_data['corroborated_claims'])}")
                if fact_data.get("refuted_or_unsubstantiated_claims"):
                    lines.append(f"      Refuted/Unsubst.   : {', '.join(str(c) for c in fact_data['refuted_or_unsubstantiated_claims'])}")
            else:
                lines.append(f"      {fact_data}")
            lines.append("")

            lines.append("  (B) SENIOR LINGUIST ASSESSMENT (Stylometrics & Emotional Framing):")
            ling_data = details.get("linguist", {})
            if isinstance(ling_data, dict):
                lines.append(f"      Linguistic Risk Score        : {ling_data.get('linguistic_risk_score', 'N/A')} / 10")
                lines.append(f"      Emotional Manipulation Level : {ling_data.get('emotional_manipulation_level', 'N/A')}")
                lines.append(f"      Stylistic Analysis           : {ling_data.get('stylistic_analysis', '')}")
                if ling_data.get("detected_devices"):
                    lines.append(f"      Detected Devices             : {', '.join(str(d) for d in ling_data['detected_devices'])}")
            else:
                lines.append(f"      {ling_data}")
            lines.append("")

            lines.append("  (C) SENIOR LOGICIAN ASSESSMENT (Internal Consistency & Intent):")
            logic_data = details.get("logician", {})
            if isinstance(logic_data, dict):
                lines.append(f"      Logical Risk Score : {logic_data.get('logical_risk_score', 'N/A')} / 10")
                lines.append(f"      Perceived Intent   : {logic_data.get('perceived_intent', 'N/A')}")
                lines.append(f"      Logical Analysis   : {logic_data.get('logical_analysis', '')}")
                logic_ledger = logic_data.get("logic_ledger") or []
                if logic_ledger:
                    lines.append("      Logic Ledger       :")
                    for entry in logic_ledger:
                        if not isinstance(entry, dict):
                            continue
                        status = entry.get("status", "UNVERIFIED")
                        issue = str(entry.get("issue", "")).strip()
                        ref = str(entry.get("evidence_ref", "")).strip()
                        suffix = f"  {ref}" if ref else ""
                        lines.append(f"        - [{status}] {issue}{suffix}")
                if logic_data.get("fallacies_or_contradictions"):
                    lines.append(f"      Fallacies/Contrad. : {', '.join(str(f) for f in logic_data['fallacies_or_contradictions'])}")
            else:
                lines.append(f"      {logic_data}")
            lines.append("")

            lines.append("[4]  CHIEF JUDICIAL MAGISTRATE SYNTHESIS DELIBERATION")
            lines.append(sep_minor)
            for r_line in result.get("reasoning", "").strip().splitlines():
                lines.append(f"    {r_line}")
            lines.append("")

        elif mode == "self_consistency" and "sc_votes" in result:
            lines.append("[3]  SELF-CONSISTENCY PARALLEL RUN BREAKDOWN")
            lines.append(sep_minor)
            lines.append(f"    Majority Vote Tally : {result.get('sc_votes', {})}")
            lines.append("")
            lines.append("[4]  SELECTED RUN REASONING PROTOCOL (CO-T)")
            lines.append(sep_minor)
            for r_line in result.get("reasoning", "").strip().splitlines():
                lines.append(f"    {r_line}")
            lines.append("")

        else:
            lines.append("[3]  7-STEP CONSTITUTIONAL CHAIN-OF-THOUGHT REASONING PROTOCOL")
            lines.append(sep_minor)
            for r_line in result.get("reasoning", "").strip().splitlines():
                lines.append(f"    {r_line}")
            lines.append("")

        lines.append("[FINAL JUDICIAL VERDICT]")
        lines.append(sep_minor)
        lines.append(f"    Verdict       : {result.get('verdict', 'INCONCLUSIVE')}")
        lines.append(f"    Confidence    : {result.get('confidence', 'N/A')}")
        lines.append(f"    Taxonomy Class: {result.get('color_class', 'N/A')}")
        lines.append("")
        lines.append("    Executive Justification:")
        for j_line in result.get("justification", "").strip().splitlines():
            lines.append(f"      {j_line}")
        lines.append("")
        lines.append("    Key Observations:")
        for fact in result.get("key_facts", []):
            lines.append(f"      • {fact}")
        lines.append("")
        if result.get("links"):
            lines.append("    Verified Supporting References:")
            for link in result.get("links", []):
                lines.append(f"      • [{link.get('source', 'EXTERNAL')}] {link.get('title', '')} ({link.get('url', '')})")
            lines.append("")

        lines.append(sep_major)
        lines.append("  END OF JUDICIAL VERDICT REPORT")
        lines.append(sep_major)
        lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        log.info(f"[Judge] Forensic log written -> {output_path}")
        print(f"  [Judge] Forensic log written -> {output_path}")
        return output_path


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
