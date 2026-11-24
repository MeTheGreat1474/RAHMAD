from .llm_client import LLMClient
from .mad_agent import DebaterAgent, strip_non_english
from .shared_memory import SharedMemoryBank
from datetime import datetime
from pathlib import Path
import logging
import json
import re

logger = logging.getLogger(__name__)

_SREF_RE = re.compile(r"\[S\d+\]")
# A sentence plus any citation tokens trailing it ("...claim. [S1], [S2]").
_SENT_RE = re.compile(r"([^.!?\n]+[.!?\u2026]+)((?:[ \t]*,?[ \t]*\[S\d+\])*)")

class OrchestratorAgent:
    def __init__(self):
        self.llm = LLMClient()
        
    def summarize_context(self, article, external_knowledge):
        prompt = (
            f"Analyze the following article for a fact-checking debate on news veracity.\n"
            f"Extract and analyze:\n"
            f"1. The specific core factual claims made in the article (who said what, what events allegedly happened).\n"
            f"2. Analysis of the current political/social climate and any potential biases, framing, or agendas the article might be propagating.\n\n"
            f"Focus strictly on the factual veracity of the claims (did the reported events/statements actually occur in reality?) and how the political framing affects this authenticity.\n\n"
            f"Article:\n{article}"
        )
        system_prompt = "You are an objective fact-checking orchestrator analyzing article claims and political/media context."
        article_analysis = strip_non_english(self.llm.generate(prompt=prompt, system_prompt=system_prompt))

        ext_section = external_knowledge.strip() if external_knowledge else "No external knowledge provided."
        combined_context = (
            f"{article_analysis.strip()}\n\n"
            f"**Verified External Knowledge (Direct from Knowledge Retrieval):**\n"
            f"{ext_section}"
        )
        return combined_context
        
    def assign_profiles(self, topic_summary):
        prompt = f"""Given the following summary of news claims and evidence, generate 6 diverse domain-specific fact-checking and investigative profiles relevant to verifying the factual veracity and authenticity of these claims.
Examples of suitable roles: Investigative Journalist, Forensic Data Analyst, Political Fact-Checker, Media Context Analyst, Public Records Specialist, Source Verification Expert, Disinformation Researcher, Archival Historian.
IMPORTANT: Return ONLY concise professional role titles (2 to 4 words each). Do NOT include descriptions, opinions, or arguments in the titles.
Return EXACTLY a JSON list of 6 strings. No other text.
Topic Summary: {topic_summary}"""
        
        fallback_profiles = [
            "Investigative Journalist", "Forensic Data Analyst", "Political Fact-Checker", 
            "Media Context Analyst", "Public Records Specialist", "Source Verification Expert"
        ]

        try:
            response = self.llm.generate(prompt=prompt, system_prompt="You output strictly valid JSON arrays of strings.")
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                raw_profiles = json.loads(match.group(0))
                if isinstance(raw_profiles, list) and len(raw_profiles) >= 6:
                    cleaned_profiles = []
                    for p in raw_profiles[:6]:
                        clean_title = strip_non_english(str(p).split(":")[0].split(" - ")[0]).strip()
                        cleaned_profiles.append(clean_title if clean_title else "Fact-Checker")
                    return cleaned_profiles
        except Exception as e:
            logger.error(f"Failed to parse profiles: {e}")
            
        logger.warning("Using fallback profiles for MAD agents.")
        return fallback_profiles

    def render_evidence_table(self, consolidated) -> str:
        """
        Render the consolidated evidence-table output as a plain-text block.

        consolidate_discourse() returns a structured dict; every downstream
        consumer (Judge prompt, MAD_output.txt, pipeline log) expects text.
        Handling the conversion here keeps one canonical rendering.
        """
        if isinstance(consolidated, str):
            return consolidated
        if not isinstance(consolidated, dict):
            return str(consolidated)

        lines = []

        def _emit_side(title, items):
            lines.append(title)
            if not items:
                lines.append("  (no evidence-backed claims recorded)")
                lines.append("")
                return
            for item in items:
                if isinstance(item, dict):
                    claim = str(item.get("claim", "")).strip()
                    ref = str(item.get("evidence_ref", "")).strip()
                    agent = str(item.get("agent", "")).strip()
                    suffix = f" {ref}" if ref else " (no citation)"
                    who = f" — {agent}" if agent else ""
                    lines.append(f"  - {claim}{suffix}{who}")
                else:
                    lines.append(f"  - {item}")
            lines.append("")

        _emit_side("EVIDENCE TABLE — REAL NEWS SIDE", consolidated.get("real_news_evidence") or [])
        _emit_side("EVIDENCE TABLE — FAKE NEWS SIDE", consolidated.get("fake_news_evidence") or [])

        lines.append("CLASH POINTS")
        clash_points = consolidated.get("clash_points") or []
        if clash_points:
            for point in clash_points:
                lines.append(f"  - {point}")
        else:
            lines.append("  (none recorded)")
        lines.append("")
        lines.append(
            "EVIDENTIAL WEIGHT: "
            + str(consolidated.get("evidential_weight", "Insufficient evidence"))
        )

        raw = consolidated.get("raw_fallback")
        if raw:
            lines.append("")
            lines.append("RAW CONSOLIDATION FALLBACK:")
            lines.append(str(raw))

        return "\n".join(lines)

    def save_debate_log(self, shared_memory, consolidated, agents, context_summary, output_path=None):
        """
        Log all agent debate arguments, profiles, context, and synthesis
        into MAD_output.txt inside the output/ folder.
        """
        consolidated = self.render_evidence_table(consolidated)
        if output_path is None:
            project_root = Path(__file__).resolve().parent.parent
            output_dir = project_root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "MAD_output.txt"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        lines = []
        sep_major = "=" * 80
        sep_minor = "-" * 60

        lines.append(sep_major)
        lines.append(f"  MULTI-AGENT DEBATE (MAD) DISCOURSE LOG  —  {now_str}")
        lines.append(sep_major)
        lines.append("")

        lines.append("[1]  ORIGINAL NEWS ARTICLE")
        lines.append(sep_minor)
        for a_line in shared_memory.article.strip().splitlines():
            lines.append(f"    {a_line}")
        lines.append("")

        lines.append("[2]  TOPIC & CONTEXT SUMMARY")
        lines.append(sep_minor)
        for c_line in context_summary.strip().splitlines():
            lines.append(f"    {c_line}")
        lines.append("")

        lines.append("[3]  DEBATE TEAMS & PROFILES")
        lines.append(sep_minor)
        lines.append("    REAL NEWS SIDE (Argues news claim is authentic, factually true, and accurate):")
        for a in agents:
            if a.side == "Real News":
                lines.append(f"      • {a.name} ({a.profile})")
        lines.append("")
        lines.append("    FAKE NEWS SIDE (Argues news claim is fabricated, distorted, or misinformation):")
        for a in agents:
            if a.side == "Fake News":
                lines.append(f"      • {a.name} ({a.profile})")
        lines.append("")

        lines.append("[4]  FULL DEBATE ARGUMENTS BY STAGE")
        lines.append(sep_minor)
        stages = ["Round 1: First Speakers", "Round 2: Second Speakers", "Round 3: Third Speakers"]
        for stage in stages:
            args = shared_memory.argument_history.get(stage, [])
            if not args:
                continue
            lines.append("")
            lines.append(f"  --- STAGE: {stage.upper()} ---")
            lines.append("")
            for entry in args:
                round_str = f" [Round {entry['round']}]" if entry.get("round") else ""
                lines.append(f"  [{entry['side']}]{round_str} {entry['agent_name']}:")
                for arg_line in str(entry["argument"]).strip().splitlines():
                    lines.append(f"      {arg_line}")
                lines.append("")

        lines.append("[5]  CONSOLIDATED DISCOURSE SYNTHESIS")
        lines.append(sep_minor)
        for s_line in consolidated.strip().splitlines():
            lines.append(f"    {s_line}")
        lines.append("")

        lines.append(sep_major)
        lines.append("  END OF MULTI-AGENT DEBATE LOG")
        lines.append(sep_major)
        lines.append("")

        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")
        print(f"  [MAD] Debate argument log written -> {output_path}")
        return output_path

    def run_debate(self, article, external_knowledge):
        logger.info("Initializing MAD Module...")
        print("\n" + "=" * 50)
        print("  MULTI-AGENT DEBATE (MAD) MODULE")
        print("=" * 50)
        print("  [MAD] Summarizing context and assigning profiles...")
        
        context_summary = self.summarize_context(article, external_knowledge)
        shared_memory = SharedMemoryBank(article, context_summary)
        
        profiles = self.assign_profiles(context_summary)
        
        agents = []
        speaker_roles = ["First Speaker", "Second Speaker", "Third Speaker"]
        for i in range(3): # 3 speakers per side
            agents.append(DebaterAgent(f"Agent_{i+1}", "Real News", profiles[i], speaker_roles[i]))
            agents.append(DebaterAgent(f"Agent_{i+4}", "Fake News", profiles[i+3], speaker_roles[i]))
            
        stages = ["Round 1: First Speakers", "Round 2: Second Speakers", "Round 3: Third Speakers"]
        
        for i, stage in enumerate(stages):
            logger.info(f"--- Starting Stage: {stage} ---")
            print(f"\n  [MAD] --- Stage: {stage} ---")
            
            # The agents are in order:
            # Index 0: First Speaker Real News
            # Index 1: First Speaker Fake News
            # Index 2: Second Speaker Real News
            # Index 3: Second Speaker Fake News
            # Index 4: Third Speaker Real News
            # Index 5: Third Speaker Fake News
            
            # In round i, we run the agents at index 2*i and 2*i + 1
            current_agents = [agents[2*i], agents[2*i + 1]]
            
            for agent in current_agents:
                print(f"    -> {agent.name} ({agent.side} - {agent.profile} - {agent.speaker_role}) is generating argument...")
                argument = agent.generate_argument(stage, shared_memory)
                display_name = f"{agent.name} ({agent.profile} - {agent.speaker_role})"
                shared_memory.add_argument(
                    stage, 
                    display_name, 
                    agent.side, 
                    argument
                )
                    
        print("\n  [MAD] Consolidating discourse...")
        consolidated = self.render_evidence_table(self.consolidate_discourse(shared_memory))
        print("  [MAD] Debate consolidated successfully.")
        print("=" * 50 + "\n")
        
        # Save complete debate arguments to output/MAD_output.txt
        try:
            self.save_debate_log(shared_memory, consolidated, agents, context_summary)
        except Exception as e:
            logger.error(f"Failed to save MAD debate log: {e}")
            print(f"  [!] Failed to save MAD debate log: {e}")

        agents_info = [{"name": a.name, "side": a.side, "profile": a.profile} for a in agents]
        return consolidated, agents_info
        
    @staticmethod
    def _claim_key(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(text).lower())[:120]

    def _normalize_evidence_items(self, raw) -> list[dict]:
        """Coerce LLM evidence items (dicts or strings) into the ledger shape."""
        if isinstance(raw, dict):
            raw = [raw]
        elif isinstance(raw, str):
            raw = [raw]
        elif not isinstance(raw, (list, tuple)):
            return []

        items, seen = [], set()
        for entry in raw:
            if isinstance(entry, str):
                claim_raw, ref_value, agent = entry, "", ""
            elif isinstance(entry, dict):
                claim_raw = (
                    entry.get("claim")
                    or entry.get("point")
                    or entry.get("statement")
                    or entry.get("evidence")
                    or entry.get("argument")
                    or ""
                )
                ref_value = (
                    entry.get("evidence_ref")
                    or entry.get("ref")
                    or entry.get("citation")
                    or entry.get("source_ref")
                    or ""
                )
                agent = entry.get("agent") or entry.get("speaker") or ""
            else:
                continue

            if isinstance(ref_value, (list, tuple)):
                ref_value = ", ".join(str(r) for r in ref_value)

            claim = strip_non_english(str(claim_raw))
            claim = re.sub(r"\*\*|`", "", claim)
            refs = _SREF_RE.findall(str(ref_value)) or _SREF_RE.findall(claim)
            claim = _SREF_RE.sub("", claim).strip(" .,;:-")
            if len(claim) < 15:
                continue

            key = self._claim_key(claim)
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "claim": claim[:500],
                "evidence_ref": ", ".join(dict.fromkeys(refs)),
                "agent": strip_non_english(str(agent))[:120],
            })
        return items[:10]

    @staticmethod
    def _extract_consolidation_json(text: str):
        """Parse JSON from a model reply that may include think tags, fences or prose."""
        if not text:
            return None
        clean = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", text, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"```(?:json)?", "", clean, flags=re.IGNORECASE).replace("```", "").strip()

        candidates = []
        start, end = clean.find("{"), clean.rfind("}")
        if start != -1 and end > start:
            candidates.append(clean[start:end + 1])
        candidates.append(clean)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _normalize_consolidation(self, parsed) -> dict:
        parsed = parsed if isinstance(parsed, dict) else {}

        def pick(*keys):
            for key in keys:
                value = parsed.get(key)
                if value:
                    return value
            return []

        real = self._normalize_evidence_items(
            pick("real_news_evidence", "real_news", "real_side", "authentic_evidence", "real_news_claims")
        )
        fake = self._normalize_evidence_items(
            pick("fake_news_evidence", "fake_news", "fake_side", "questioned_evidence", "fake_news_claims")
        )

        clashes = pick("clash_points", "clashes", "key_clash_points")
        if isinstance(clashes, str):
            clashes = [clashes]
        clash_points = []
        for point in clashes if isinstance(clashes, (list, tuple)) else []:
            text = re.sub(r"\*\*|`", "", strip_non_english(str(point))).strip()
            if text:
                clash_points.append(text[:300])

        weight_raw = (
            parsed.get("evidential_weight")
            or parsed.get("evidence_weight")
            or parsed.get("weight")
            or ""
        )

        return {
            "real_news_evidence": real,
            "fake_news_evidence": fake,
            "clash_points": clash_points[:6],
            "evidential_weight": strip_non_english(str(weight_raw)).strip(),
        }

    @staticmethod
    def _evidence_sentences(text: str) -> list[str]:
        """Split text into sentences while keeping citation tokens attached."""
        units = []
        for line in re.split(r"\n+", text):
            line = line.strip()
            if not line:
                continue
            consumed_end = 0
            for match in _SENT_RE.finditer(line):
                sentence = (match.group(1) + match.group(2)).strip()
                consumed_end = match.end()
                if _SREF_RE.search(sentence):
                    units.append(sentence)
            remainder = line[consumed_end:].strip()
            if remainder and _SREF_RE.search(remainder):
                units.append(remainder)
        return units

    def _collect_cited_claims(self, shared_memory) -> dict:
        """Deterministically harvest claims that carry [S#] refs from the transcript."""
        collected = {"Real News": [], "Fake News": []}
        seen = {"Real News": set(), "Fake News": set()}
        history = getattr(shared_memory, "argument_history", {}) or {}

        for entries in history.values():
            for entry in entries:
                side = entry.get("side")
                if side not in collected:
                    continue
                agent = strip_non_english(str(entry.get("agent_name", "")))
                text = strip_non_english(str(entry.get("argument", "")))
                for chunk in self._evidence_sentences(text):
                    claim = re.sub(r"^[\s>*#\-\u2022\d.)]+", "", chunk).strip()
                    claim = re.sub(r"\*\*|`", "", claim)
                    refs = list(dict.fromkeys(_SREF_RE.findall(claim)))
                    claim = _SREF_RE.sub("", claim).strip(" .,;:-")
                    if len(claim) < 15:
                        continue
                    key = self._claim_key(claim)
                    if key in seen[side]:
                        continue
                    seen[side].add(key)
                    collected[side].append({
                        "claim": claim[:500],
                        "evidence_ref": ", ".join(refs),
                        "agent": agent[:120],
                    })
                    if len(collected[side]) >= 10:
                        break
        return collected

    def _derive_clash_points(self, consolidated) -> list[str]:
        def refs(items):
            found = set()
            for item in items:
                found.update(_SREF_RE.findall(str(item.get("evidence_ref", ""))))
            return found

        real_refs = refs(consolidated.get("real_news_evidence") or [])
        fake_refs = refs(consolidated.get("fake_news_evidence") or [])
        shared = sorted(real_refs & fake_refs)

        if shared:
            return [f"Both sides cite {', '.join(shared)} but draw opposite conclusions from the same evidence."]
        if real_refs and fake_refs:
            return [
                f"Real News cites {', '.join(sorted(real_refs))} while Fake News cites "
                f"{', '.join(sorted(fake_refs))}; each side weighs different evidence."
            ]
        if real_refs or fake_refs:
            side = "Real News" if real_refs else "Fake News"
            cited = ", ".join(sorted(real_refs | fake_refs))
            return [f"Only {side} anchors claims to cited evidence ({cited}); the opposing side offers no citation."]
        return []

    def _derive_evidential_weight(self, consolidated) -> str:
        real_count = len(consolidated.get("real_news_evidence") or [])
        fake_count = len(consolidated.get("fake_news_evidence") or [])
        if real_count and not fake_count:
            return "Real News stronger"
        if fake_count and not real_count:
            return "Fake News stronger"
        if real_count and fake_count:
            return "Balanced"
        return "Insufficient evidence"

    @staticmethod
    def _canonical_weight(weight: str, consolidated) -> str:
        text = (weight or "").lower()
        if "real" in text:
            return "Real News stronger"
        if "fake" in text:
            return "Fake News stronger"
        if "balanc" in text or "equal" in text:
            return "Balanced"
        if "insufficient" in text or "no evidence" in text:
            return "Insufficient evidence"
        return ""

    def consolidate_discourse(self, shared_memory):
        history = shared_memory.get_formatted_history()
        prompt = (
            f"You are an objective academic research analyst reviewing a completed academic debate on media verification.\n"
            f"Provide a neutral, structured analytical synthesis of the debate arguments recorded below.\n"
            f"CRITICAL: Your output MUST be a JSON object with the following structure:\n"
            f'{{\n'
            f'  "real_news_evidence": [\n'
            f'    {{"claim": "<specific claim from Real News side>", "evidence_ref": "[S1] or similar", "agent": "<agent name>"}},\n'
            f'    ...\n'
            f'  ],\n'
            f'  "fake_news_evidence": [\n'
            f'    {{"claim": "<specific claim from Fake News side>", "evidence_ref": "[S1] or similar", "agent": "<agent name>"}},\n'
            f'    ...\n'
            f'  ],\n'
            f'  "clash_points": [\n'
            f'    "<description of key evidentiary clash points>"\n'
            f'  ],\n'
            f'  "evidential_weight": "Real News stronger | Fake News stronger | Balanced | Insufficient evidence"\n'
            f'}}\n'
            f"Rules:\n"
            f"- Return ONLY the JSON object. Do not repeat or summarise the transcript outside it.\n"
            f"- Every claim in real_news_evidence and fake_news_evidence MUST include an evidence_ref that appears in the debate history (e.g., [S1], [S2]).\n"
            f"- If a side makes a claim without citing evidence, do NOT include it.\n"
            f"- Do NOT include a moderator verdict or conclusion about truth/falsity.\n"
            f"- Extract only the strongest, best-evidenced points from each side (maximum 10 per side).\n"
            f"- Write exclusively in English. Never output Chinese or any non-English script characters.\n\n"
            f"Debate Transcript:\n{history}"
        )
        system_prompt = (
            "You are an objective scholarly evaluator synthesizing arguments from a media analysis debate. "
            "Output ONLY valid JSON. Respond in English only."
        )

        parsed = None
        try:
            raw = self.llm.generate(prompt=prompt, system_prompt=system_prompt)
            parsed = self._extract_consolidation_json(raw)
            if parsed is None:
                logger.warning(
                    "Consolidate discourse returned non-JSON; building the evidence table from the transcript"
                )
        except Exception as exc:
            logger.error(f"Consolidation LLM call failed: {exc}")

        consolidated = self._normalize_consolidation(parsed)

        # Guarantee both sides are populated whenever the debate cited evidence:
        # the local model does not always honour the JSON instruction.
        cited = self._collect_cited_claims(shared_memory)
        for key, side in (("real_news_evidence", "Real News"), ("fake_news_evidence", "Fake News")):
            items = consolidated[key]
            seen_keys = {self._claim_key(item["claim"]) for item in items}
            covered_refs = set()
            for item in items:
                covered_refs.update(_SREF_RE.findall(str(item.get("evidence_ref", ""))))
            for item in cited.get(side, []):
                claim_key = self._claim_key(item["claim"])
                item_refs = set(_SREF_RE.findall(str(item.get("evidence_ref", ""))))
                # Skip transcript duplicates of a claim the model already extracted.
                if claim_key in seen_keys or (item_refs and item_refs <= covered_refs):
                    continue
                items.append(item)
                seen_keys.add(claim_key)
                covered_refs.update(item_refs)
            consolidated[key] = items[:10]

        if not consolidated["clash_points"]:
            consolidated["clash_points"] = self._derive_clash_points(consolidated)

        weight = self._canonical_weight(consolidated.get("evidential_weight", ""), consolidated)
        consolidated["evidential_weight"] = weight or self._derive_evidential_weight(consolidated)

        return consolidated

