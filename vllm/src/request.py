from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from copy import copy

from vllm.sampling_params import SamplingParams

class Status:
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Request:
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams, eos_token_id: int):
    
        self.id = next(Request.counter)
        self.status = Status.WAITING
        self.ignore_eos = sampling_params.ignore_eos
        self.max_new_tokens = sampling_params.max_new_tokens
        self.sampling_params = sampling_params
        self.eos_token_id = eos_token_id
        self.token_ids = copy(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_completion_tokens = 0
    
    def is_finished(self) -> bool:
        return self.status == Status.FINISHED

    def get_last_token_id(self) -> int:
        return self.token_ids[-1]

    def __len__(self) -> int:
        return self.num_completion_tokens + self.num_prompt_tokens
    
    
