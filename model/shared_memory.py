class SharedMemoryBank:
    def __init__(self, context_summary):
        self.context_summary = context_summary
        self.argument_history = {
            "Round 1: First Speakers": [],
            "Round 2: Second Speakers": [],
            "Round 3: Third Speakers": []
        }

    def add_argument(self, stage, agent_name, side, argument, round_num=None):
        if stage in self.argument_history:
            self.argument_history[stage].append({
                "agent_name": agent_name,
                "side": side,
                "argument": argument,
                "round": round_num
            })

    def get_formatted_history(self, up_to_stage=None):
        formatted = f"Context & Fact-Checking Summary:\n{self.context_summary}\n\n"
        
        stages_to_include = ["Round 1: First Speakers", "Round 2: Second Speakers", "Round 3: Third Speakers"]
        
        for stage in stages_to_include:
            arguments = self.argument_history.get(stage, [])
            if arguments:
                formatted += f"--- {stage} ---\n"
                for arg in arguments:
                    round_str = f" [Round {arg['round']}]" if arg.get("round") else ""
                    formatted += f"[{arg['side']}]{round_str} {arg['agent_name']}: {arg['argument']}\n\n"
            
            if up_to_stage and stage == up_to_stage:
                break
                
        return formatted

