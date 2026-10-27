from .llm_client import LLMClient
import logging

logger = logging.getLogger(__name__)

class DebaterAgent:
    def __init__(self, name, side, profile, speaker_role="First Speaker"):
        self.name = name
        self.side = side  # "Real News" or "Fake News"
        self.profile = profile
        self.speaker_role = speaker_role
        self.llm = LLMClient()
        
    def generate_argument(self, stage, shared_memory):
        logger.info(f"Agent {self.name} ({self.side} - {self.profile}) generating {stage}")
        
        history = shared_memory.get_formatted_history()
        
        if self.side == "Real News":
            side_instruction = (
                "Argue that the news report/claim is AUTHENTIC, FACTUALLY ACCURATE, and CREDIBLE ('Real News'). "
                "Highlight corroborating evidence, official records, verified quotes, and source consistency showing that "
                "the events or statements described actually occurred as reported."
            )
        else:
            side_instruction = (
                "Argue that the news report/claim is FABRICATED, DECEPTIVE, MISLEADING, or UNVERIFIED ('Fake News'). "
                "Highlight discrepancies, lack of verifiable sourcing, out-of-context quotes, manipulative framing, "
                "or signs of misinformation indicating the report cannot be trusted as factual truth. "
                "CRITICAL CHEAPFAKE CHECK: If any photographic or visual context is discussed, examine whether the article "
                "is deceptively pairing a genuine, real photograph with an unverified, misattributed, or speculative caption/claim "
                "(Cheapfake / False Connection). Emphasize that an authentic photo alone does NOT prove an uncorroborated story."
            )

        speaker_instruction = ""
        if self.speaker_role == "First Speaker":
            speaker_instruction = (
                "As the First Speaker, your argument MUST follow this structure:\n"
                "- Introduction: State your team's stance and define the scope of factual verification.\n"
                "- Arguments, reasons and evidence: Present your primary substantive arguments and evidence utilizing your expertise as a " + self.profile + ".\n"
                "- Summary of own case: Concise wrap-up of your primary claims (1-2 sentences).\n"
                "- Conclusion (Ending): Clear final statement affirming your stance."
            )
        elif self.speaker_role == "Second Speaker":
            speaker_instruction = (
                "As the Second Speaker, your argument MUST follow this structure:\n"
                "- Introduction: Acknowledge the debate flow and your role as " + self.profile + ".\n"
                "- Rebuttal: Vigorously attack the opposing speaker's arguments using specific counter-points and the 'even if' tactic.\n"
                "- Summary of own first speaker: Keep this VERY BRIEF (1-2 sentences maximum). DO NOT restate or regurgitate their paragraphs.\n"
                "- Arguments, reasons and evidence: CRITICAL: You MUST introduce BRAND NEW arguments and evidence distinct from your first speaker. Approach the topic from your specialized domain as a " + self.profile + " (e.g. data forensics, technical anomalies, psychological/disinformation analysis, institutional verification) that has NOT been discussed yet.\n"
                "- Summary of team case: Briefly synthesize how both your and the first speaker's points solidify the team stance (1-2 sentences).\n"
                "- Conclusion: Decisive closing statement."
            )
        elif self.speaker_role == "Third Speaker":
            speaker_instruction = (
                "As the Third Speaker, your argument MUST follow this structure:\n"
                "- Introduction: Frame the final clash of the debate from your perspective as a " + self.profile + ".\n"
                "- Rebuttal: Systematically dismantle the opposing team's latest arguments and expose their analytical flaws.\n"
                "- Summary of team arguments and case: Crystallize the central clash. Weigh your team's cumulative evidence against the opposing side's gaps. DO NOT regurgitate earlier speeches verbatim; synthesize the clash.\n"
                "- No new arguments except as specific rebuttal: Do not introduce unrelated constructive claims; focus entirely on defending against attacks and finalizing rebuttals.\n"
                "- Conclusion: Final, authoritative verdict affirming your team's case."
            )

        if not (self.speaker_role == "First Speaker" and self.side == "Real News"):
            rebuttal_instruction = "6. VIGOROUSLY DEFEND AND REBUT: You must fiercely defend your case and aggressively rebut the opposing side's arguments. Use the 'even if' debate tactic: 'Even if we agreed to your premise/definition (which we don't), your arguments are still wrong because...'. You MUST attack their arguments directly in any event."
        else:
            rebuttal_instruction = "6. VIGOROUSLY DEFEND: You must fiercely present and defend your case."

        system_prompt = f"""You are a {self.profile} participating in a simulated academic debate on the FACTUAL VERACITY of a news article.
Your assigned stance: [{self.side}]. Your Role: [{self.speaker_role}].

{side_instruction}

SPEAKER ROLE INSTRUCTIONS:
{speaker_instruction}

CRITICAL RULES:
1. STRICT ANTI-REGURGITATION RULE: Do NOT regurgitate, repeat, or copy-paste arguments, points, or evidence that have already been presented by your teammate or previous speakers in the debate history. You MUST advance the debate with new insights, new evidence angles, and deeper critical analysis.
2. KEEP SUMMARIES CONCISE: When asked for a summary (e.g., summary of first speaker or summary of team case), limit it to 1-2 tight sentences. Never spend entire paragraphs restating what was already stated in the history.
3. DOMAIN SPECIALIZATION: Fully embody your assigned profile as a {self.profile}. Bring arguments, analytical methods, and terminology unique to your domain that differentiate you from other speakers on your team.
4. FOCUS STRICTLY ON FACTUAL VERACITY: Debate whether the reported event, statement, or quote ACTUALLY HAPPENED in reality as reported vs whether it is fabricated, distorted, or deceptive misinformation.
5. DO NOT DEBATE POLICY FEASIBILITY OR MERIT: Do NOT debate whether the policy, proposal, political action, or promise described in the news is achievable, economically viable, logical, or morally good/bad. Fact-check whether the news report itself is true/authentic (Real News) vs false/misleading (Fake News).
6. EVALUATE POLITICAL CLIMATE AND BIAS: Analyze the current political/social climate, potential agendas, and biases surrounding the topic. If the news is heavily biased or propagating an agenda, argue how this framing affects its factual authenticity.
{rebuttal_instruction}
7. Keep your argument concise, focused, sharp, and strictly in character. Do not refuse to argue your side."""

        prompt = f"""Debate Stage: {stage}

Current Debate History:
{history}

Task:
Provide your {stage} argument representing the [{self.side}] side as a {self.profile} in the role of {self.speaker_role}.
Focus exclusively on proving whether the news report is genuine and factually authentic (Real News) or false and deceptive misinformation (Fake News).
Remember:
- Do NOT regurgitate or copy points already stated in the debate history. Bring distinct arguments, evidence, or critical rebuttals based on your {self.profile} expertise.
- Keep summaries strictly brief (1-2 sentences). Do NOT spend multiple paragraphs repeating previous speakers.
- Do NOT argue whether the policy or event is achievable or logical. Focus solely on its factual veracity."""

        response = self.llm.generate(prompt=prompt, system_prompt=system_prompt)
        return response

