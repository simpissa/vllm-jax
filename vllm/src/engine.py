from dataclasses import fields
from transformers import AutoTokenizer
import jax
from typing import Optional, Any

from vllm.config import Config
from vllm.sampling_params import SamplingParams
from .scheduler import Scheduler
from .kv_cache_manager import KVCacheManager
from .request import Request, Status
from .model_runner import ModelRunner

class Engine:

    def __init__(self, config: Config, device: Optional[Any] = None):
        self.config = config
        self.device = device or jax.devices()[0]
        self.rng_key = jax.random.PRNGKey(0)

        kv_cache_manager = KVCacheManager(self.config.num_blocks, self.config.block_size)

        self.scheduler = Scheduler(self.config, kv_cache_manager)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.model_runner = ModelRunner(self.config, kv_cache_manager, device=self.device)
    
    def preprocess(self, prompt: str):
        token_ids = self.tokenizer.encode(prompt)
        return {"token_ids" : token_ids}

    def add_request(self, request: Request):
        self.scheduler.add(request)

    def step(self):
        decode_requests, prefill_requests = self.scheduler.schedule()
        token_ids, self.rng_key = self.model_runner.run(
            decode_requests,
            prefill_requests,
            self.rng_key,
        ) # single token produced by each request
        self.scheduler.postprocess(decode_requests, token_ids)        
        outputs = [(request.id, request.num_completion_tokens) for request in decode_requests if request.is_finished()]
        return outputs


    def generate(self, prompts: list[str], **kwargs):
        sampling_fields = {f.name for f in fields(SamplingParams)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in sampling_fields}
        sampling_params = SamplingParams(**config_kwargs)
        self.rng_key = jax.random.PRNGKey(sampling_params.seed)



        requests = []
        request_id_to_index = {}
        request_by_id = {}
        for index, prompt in enumerate(prompts):
            inputs = self.preprocess(prompt)
            request = Request(sampling_params=sampling_params, eos_token_id=self.tokenizer.eos_token_id, **inputs)
            requests.append(request)
            request_id_to_index[request.id] = index
            request_by_id[request.id] = request
            self.add_request(request)
        
        results = [""] * len(prompts)
        while not self.scheduler.is_finished():
            outputs = self.step()

            for request_id, _ in outputs:
                request = request_by_id[request_id]
                completion_ids = request.token_ids[request.num_prompt_tokens:]
                results[request_id_to_index[request.id]] = self.tokenizer.decode(
                    completion_ids
                )

        return results
