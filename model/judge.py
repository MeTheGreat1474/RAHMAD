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
or wire photo licensing catalogs (e.g. Reuters, SCMP, Reuters Connect). This strongly \
supports CREDIBLE.
      - 'Out-of-Context (Misattributed / Recycled)': The image is demonstrably recycled from \
a completely different historical year, different event, or unrelated location. This \
strongly supports a FALSE CONNECTION or MISLEADING CONTENT classification.
      - 'Generic / Illustrative Imagery': Standard stock or symbolic photography without \
deceptive intent. Treat as neutral visual evidence.
      - 'Inconclusive / Insufficient Evidence': Ambiguous visual provenance; rely \
primarily on textual and debate evidence.

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
explicit 'Out-of-Context Imagery Detected' status and category. If marked as \
'Out-of-Context (Misattributed / Recycled)' (recycled from a different time/place/event), \
you MUST strongly consider a FALSE CONNECTION or MISLEADING CONTENT verdict. \
However, if marked as 'Authentic Context / Matching Event', do NOT penalize an article \
simply because syndicated outlets or photo agencies rephrase headlines for the same real-world event.
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

Note: This is verified external ground-truth evidence retrieved from reputable whitelisted news outlets via semantic similarity filtering.

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
1. Knowledge Retrieval Evidence (ground-truth external reports from reputable news sources)
2. Reverse Image Search Evidence (visual provenance, authenticity vs. out-of-context recycling)
3. Multi-Agent Debate Summary (critical arguments from Real News and Fake News domain debaters)

Evaluation Protocol:
- Identify the core verifiable factual claims (dates, events, people, actions, outcomes).
- Cross-reference claims against external Knowledge Retrieval: Are they corroborated, refuted, or unmentioned?
- Analyze Reverse Image Search findings: Is the image authentic to the event, or misattributed/recycled?
- Evaluate Multi-Agent Debate arguments: Which side presents stronger factual documentation?
- Assign a Factual Risk Score from 1 to 10 (1 = completely verified and corroborated, 10 = completely fabricated, contradictory, or debunked).

CRITICAL AUDIENCE & TERMINOLOGY RULE:
Your factual analysis will be presented to end users.
NEVER use internal system acronyms like "KR", "KR module", "RIS", "RIS module", "MAD", "MAD module", or "EP".
Instead, use user-friendly phrases:
- "Reverse Image Search shows..." or "Visual verification indicates..." (instead of RIS)
- "Debate shows..." or "The multi-agent debate indicates..." (instead of MAD)
- "External news reports show..." or "Knowledge retrieval indicates..." (instead of KR)
- "Initial screening indicates..." (instead of EP)

Output Format:
You must output a single, valid JSON object:
{
  "factual_analysis": "<concise 3-5 sentence analysis of claim corroboration, image authenticity, and external evidence alignment using user-friendly phrasing>",
  "corroborated_claims": ["<claim 1>", "<claim 2>"],
  "refuted_or_unsubstantiated_claims": ["<claim 1>", "<claim 2>"],
  "factual_risk_score": <integer from 1 to 10>
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
Your sole mission is to assess the INTERNAL CONSISTENCY, ARGUMENT COHERENCE, SOURCE ATTRIBUTION, and COMMUNICATIVE INTENT of the news article.

Evaluation Protocol:
- Internal consistency: Do the statements within the article contradict one another? Do reported numbers/statistics add up?
- Source attribution plausibility: Are quoted authorities, institutions, or witnesses credible and specific, or are they suspiciously anonymous, fabricated, or misattributed? Check for signs of imposter content (impersonating known media or officials).
- Communicative intent: What is the primary purpose of the piece? (To inform objectively, to satirize current events, to generate clickbait ad revenue, to deceive/defame, or to spread propaganda?)
- Logical fallacies: Detect false dichotomies, non-sequiturs, strawman arguments, or guilt-by-association.
- Assign a Logical Risk Score from 1 to 10 (1 = rigorous internal logic, transparent sourcing, authentic informational intent; 10 = severe contradictions, fabricated sources, imposter attribution, or deliberate deception).

Output Format:
You must output a single, valid JSON object:
{
  "logical_analysis": "<concise 3-5 sentence analysis of internal consistency, sourcing plausibility, and underlying intent>",
  "perceived_intent": "Informational" | "Satirical" | "Deceptive" | "Imposter" | "Sensationalist",
  "fallacies_or_contradictions": ["<specific issue 1>", "<specific issue 2>"],
  "logical_risk_score": <integer from 1 to 10>
}
"""

_CHIEF_JUDGE_PROMPT = """\
You are the Chief Judicial Magistrate and Synthesis Arbiter for RAHMAD (Reliable Automated Hybrid Multi-Agent Detection).
You preside over a panel of three specialized investigative judges who have analyzed the news article:
1. The Fact-Checker (External Evidence, Reverse Image Search provenance, and Multi-Agent Debate verification)
2. The Linguist (Writing style, emotional manipulation, sensationalism, and rhetoric)
3. The Logician (Internal consistency, source attribution plausibility, and communicative intent)

You also receive the Early Predictor multimodal signal and the original article text.

Judicial Synthesis Principles:
- Factual Priority: If the Fact-Checker proves core claims are refuted or the image is out-of-context (recycled), strong presumption toward FABRICATED CONTENT, FALSE CONNECTION, or MISLEADING CONTENT.
- Satire/Parody Recognition: If the Linguist and Logician detect satirical markers, irony, and humorous intent, classify as SATIRE OR PARODY even if the claims are factually untrue.
- Imposter Detection: If the Logician detects false impersonation of an established institution or journalist, classify as IMPOSTER CONTENT.
- Credibility Standard: When all three specialists report low risk scores (1-3) and confirm corroboration and internal consistency, classify as CREDIBLE.
- Inconclusive Standard: Use INCONCLUSIVE only when signals remain irreconcilably divided without sufficient evidence.

CRITICAL AUDIENCE & TERMINOLOGY RULE:
Your output ("reasoning", "justification", and "key_facts") is displayed directly to the end user.
- NEVER use technical jargon, internal acronyms, or system module names such as "MAD", "MAD module", "RIS", "RIS module", "KR", "KR module", or "EP".
- Instead, express findings in natural language that any reader can understand:
  • Instead of "MAD" or "MAD module", write "Debate shows...", "The debate analysis indicates...", or "Multi-Agent Debate arguments show...".
  • Instead of "RIS" or "RIS module", write "Reverse Image Search shows...", "Visual verification indicates...", or "Image provenance checks reveal...".
  • Instead of "KR" or "KR module", write "External knowledge retrieval indicates...", "Verified news reports show...", or "Independent fact-checking reveals...".
  • Instead of "EP" or "EP module", write "Initial screening indicates..." or "Early Predictor signal shows...".
Write with the objective, accessible tone of a senior investigative fact-checker.

Your output MUST be a single, valid JSON object formatted exactly as follows:
Important: Use markdown bolding (e.g. **key finding** or **critical claim**) to emphasize important phrases across "justification", "reasoning", and "key_facts".

{
  "reasoning": "<synthesized judicial deliberation detailing how the specialist findings were weighed, formatted as structured Markdown bullet points (e.g. '* **Factual Priority:** ...'). Do NOT write a wall of text. Use 1-2 detailed sentences per bullet point, strictly using user-friendly terms like 'Reverse Image Search shows...' or 'Debate shows...'. You MUST explicitly include dedicated bullet points for ALL pipeline components: Factual Assessment, Linguistic Assessment, Logical Assessment, Debate Analysis, Early Predictor (initial screening), External Knowledge Retrieval (verified news), and Reverse Image Search (visual provenance). Summarize their results transparently, even if they contradict your final verdict.>",
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
        Maximum tokens in the LLM response (default 1024).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
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
    ) -> str:
        """POST to OpenRouter and return the raw content string with multi-model fallback."""
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
        for model in models_to_try:
            print(f"  [Judge] Querying OpenRouter model '{model}'...")
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_message},
                ],
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens":  self.max_tokens,
            }

            log.info(f"[Judge] Sending request to OpenRouter — model: {model}")
            try:
                response = requests.post(
                    _OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                    timeout=45,
                )
                response.raise_for_status()

                resp_json = response.json()
                if "choices" in resp_json and len(resp_json["choices"]) > 0:
                    content = resp_json["choices"][0]["message"]["content"]
                    log.info(f"[Judge] Response received ({len(content)} chars) from model {model}.")
                    print(f"  [Judge] [OK] Verdict response received from '{model}' ({len(content)} chars).")
                    self.last_raw_response = content
                    return content
                else:
                    log.warning(f"[Judge] Model {model} response missing choices: {resp_json}")
            except Exception as exc:
                log.warning(f"[Judge] Model {model} failed: {exc}")
                print(f"  [Judge] [!] Model '{model}' failed or timed out. Trying fallback...")
                last_err = exc

        raise RuntimeError(f"All OpenRouter LLM models failed. Last error: {last_err}")

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """
        Extract and validate the JSON verdict from the LLM response.
        Handles cases where the model wraps JSON in markdown code fences.
        Now also extracts the 'reasoning' field added by the enhanced prompt.
        """
        # Remove <think>...</think> reasoning blocks common in modern LLMs
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        
        # Remove markdown code fences and whitespace
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip().rstrip("`").strip()

        # Try parsing directly
        data = None
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # First attempt: find first '{' and matching/last '}'
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                json_str = match.group()
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    pass

            if data is None:
                # Second attempt: handle potential truncated JSON (e.g. max_tokens reached or unclosed quotes/brackets)
                start_idx = cleaned.find("{")
                if start_idx != -1:
                    substring = cleaned[start_idx:]
                    # Attempt to salvage truncated JSON by removing unclosed strings and closing brackets/braces
                    # Try truncating up to last quote or comma and auto-closing braces
                    for cutoff in range(len(substring), 0, -1):
                        candidate = substring[:cutoff].rstrip(", \n\r\t")
                        # count braces
                        open_braces = candidate.count("{") - candidate.count("}")
                        if open_braces > 0:
                            # Close string if inside an unclosed string quote
                            quotes = candidate.count('"') - candidate.count(r'\"')
                            if quotes % 2 != 0:
                                candidate += '"'
                            candidate += "}" * open_braces
                        try:
                            data = json.loads(candidate)
                            break
                        except json.JSONDecodeError:
                            continue

            if data is None:
                raise ValueError(
                    f"[Judge] Could not extract valid JSON from LLM response:\n{raw}"
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

        return {
            "verdict":       verdict,
            "color_class":   color_class,
            "confidence":    str(data.get("confidence", "50.0%")),
            "justification": justification,
            "reasoning":     reasoning,   # CoT scratchpad
            "key_facts":     list(clean_facts),
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
                executor.submit(self._call_api, user_message, 0.2, prompt): (role, label)
                for role, (prompt, label) in specialist_tasks.items()
            }
            for future in as_completed(future_to_role):
                role, label = future_to_role[future]
                try:
                    raw_text = future.result()
                    try:
                        clean_raw = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                        clean_raw = re.sub(r"```(?:json)?\s*", "", clean_raw).strip().rstrip("`").strip()
                        parsed = json.loads(clean_raw)
                        specialist_results[role] = parsed
                    except Exception:
                        specialist_results[role] = {"summary": raw_text.strip()}
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
            chief_raw = self._call_api(
                chief_user_message,
                temperature=0.2,
                system_prompt=_CHIEF_JUDGE_PROMPT,
            )
            result = self._parse_response(chief_raw)
            result["ep_signal"] = ep_result
            result["panel_details"] = specialist_results
            self.last_user_message = chief_user_message
            self.last_raw_response = chief_raw
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
