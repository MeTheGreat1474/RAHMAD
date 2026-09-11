from .llm_client import LLMClient
from .mad_agent import DebaterAgent
from .shared_memory import SharedMemoryBank
from datetime import datetime
from pathlib import Path
import logging
import json
import re

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    def __init__(self):
        self.llm = LLMClient()
        
    def summarize_context(self, article, external_knowledge):
        prompt = (
            f"Analyze the following article and external verified knowledge for a fact-checking debate on news veracity.\n"
            f"Extract and summarize:\n"
            f"1. The specific core factual claims made in the article (who said what, what events allegedly happened).\n"
            f"2. The verified external facts that corroborate or contradict these claims.\n"
            f"3. Analysis of the current political/social climate and any potential biases or agendas the article might be propagating.\n\n"
            f"Focus strictly on the factual veracity of the claims (did the reported events/statements actually occur in reality?) and how the political framing affects this authenticity.\n\n"
            f"Article:\n{article}\n\n"
            f"External Knowledge:\n{external_knowledge}"
        )
        system_prompt = "You are an objective fact-checking orchestrator summarizing claims and evidence for a news veracity debate."
        return self.llm.generate(prompt=prompt, system_prompt=system_prompt)
        
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
                        clean_title = str(p).split(":")[0].split(" - ")[0].strip()
                        cleaned_profiles.append(clean_title if clean_title else "Fact-Checker")
                    return cleaned_profiles
        except Exception as e:
            logger.error(f"Failed to parse profiles: {e}")
            
        logger.warning("Using fallback profiles for MAD agents.")
        return fallback_profiles

    def save_debate_log(self, shared_memory, consolidated, agents, context_summary, output_path=None):
        """
        Log all agent debate arguments, profiles, context, and synthesis
        into MAD_output.txt inside the output/ folder.
        """
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
        consolidated = self.consolidate_discourse(shared_memory)
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
        
    def consolidate_discourse(self, shared_memory):
        history = shared_memory.get_formatted_history()
        prompt = (
            f"You are an objective academic research analyst reviewing a completed academic debate on media verification.\n"
            f"Provide a neutral, structured analytical synthesis of the debate arguments recorded below:\n"
            f"1. Key arguments and evidence presented in favor of factual authenticity (Real News side).\n"
            f"2. Key arguments and evidence presented questioning authenticity or highlighting discrepancies/bias (Fake News side).\n"
            f"3. Assessment of clash points and evidential weight between both sides.\n\n"
            f"Debate Transcript:\n{history}"
        )
        system_prompt = "You are an objective scholarly evaluator synthesizing arguments from a media analysis debate."
        
        consolidated = self.llm.generate(prompt=prompt, system_prompt=system_prompt)
        return consolidated

