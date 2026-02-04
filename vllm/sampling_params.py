from dataclasses import dataclass

@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    seed: int = 0
    max_new_tokens: int = 512
    ignore_eos: bool = False
