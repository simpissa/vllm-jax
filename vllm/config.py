from dataclasses import dataclass

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int
    max_num_seqs: int

    num_blocks: int
    block_size: int