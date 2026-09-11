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
                "or signs of misinformation indicating the report cannot be trusted as factual truth."
            )

        speaker_instruction = ""
        if self.speaker_role == "First Speaker":
            speaker_instruction = (
                "As the First Speaker, your argument MUST follow this structure:\n"
                "- Introduction\n"
                "- Arguments, reasons and evidence\n"
                "- Summary of own case\n"
                "- Conclusion (Ending)"
            )
        elif self.speaker_role == "Second Speaker":
            speaker_instruction = (
                "As the Second Speaker, your argument MUST follow this structure:\n"
                "- Introduction\n"
                "- Rebuttal (address the previous speaker's points)\n"
                "- Summary of own first speaker\n"
                "- Arguments, reasons and evidence\n"
                "- Summary of team case\n"
                "- Conclusion"
            )
        elif self.speaker_role == "Third Speaker":
            speaker_instruction = (
                "As the Third Speaker, your argument MUST follow this structure:\n"
                "- Introduction\n"
                "- Rebuttal (address the previous speaker's points)\n"
                "- Summary of team arguments and case\n"
                "- No new arguments except as specific rebuttal\n"
                "- Conclusion"
            )

        system_prompt = f"""You are a {self.profile} participating in a simulated academic debate on the FACTUAL VERACITY of a news article.
Your assigned stance: [{self.side}]. Your Role: [{self.speaker_role}].

{side_instruction}

SPEAKER ROLE INSTRUCTIONS:
{speaker_instruction}

CRITICAL RULES:
1. FOCUS STRICTLY ON FACTUAL VERACITY: Debate whether the reported event, statement, or quote ACTUALLY HAPPENED in reality as reported vs whether it is fabricated, distorted, or deceptive misinformation.
2. DO NOT DEBATE POLICY FEASIBILITY OR MERIT: Do NOT debate whether the policy, proposal, political action, or promise described in the news is achievable, economically viable, logical, or morally good/bad. That is completely irrelevant to fact-checking. You are fact-checking whether the news report itself is true/authentic (Real News) vs false/misleading (Fake News).
3. Use your domain expertise as a {self.profile} to evaluate claims, evidence, and context from the debate history.
4. Keep your argument concise, focused, sharp, and strictly in character. Do not refuse to argue your side."""

        prompt = f"""Debate Stage: {stage}

Current Debate History:
{history}

Task:
Provide your {stage} argument representing the [{self.side}] side as a {self.profile} in the role of {self.speaker_role}.
Focus exclusively on proving whether the news report is genuine and factually authentic (Real News) or false and deceptive misinformation (Fake News).
Remember: Do NOT argue whether the policy or event is achievable or logical. Focus solely on its factual veracity."""

        response = self.llm.generate(prompt=prompt, system_prompt=system_prompt)
        return response

