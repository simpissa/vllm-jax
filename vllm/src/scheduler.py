from collections import deque

from vllm.src.request import Request, Status
from vllm.src.kv_cache_manager import KVCacheManager
from vllm.config import Config

class Scheduler:

    def __init__(self, config: Config, kv_cache_manager: KVCacheManager):
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_num_seqs = config.max_num_seqs
        self.kv_cache_manager = kv_cache_manager
        self.waiting : list[Request] = deque()
        self.running : list[Request] = deque() # front is left

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, request: Request):
        self.waiting.append(request)

    def preempt(self, request):
        request.status = Status.WAITING
        self.waiting.appendleft(request)
        self.kv_cache_manager.deallocate(request)

    def schedule(self) -> tuple[list[Request], list[Request]]:
        decode_requests = []
        prefill_requests = []
        num_batched_tokens = num_seqs = 0

        # decode
        while self.running and num_seqs < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
            request = self.running.popleft()
            
            while not self.kv_cache_manager.can_allocate(request, False):
                assert self.running, "Request cant fit in memory"
                self.preempt(self.running.pop()) # can choose better policy
            
            self.kv_cache_manager.allocate_slots(request, False) 
            decode_requests.append(request)
            num_seqs += 1
            num_batched_tokens += 1

        # self.running.extend(decode_requests) # round-robin
        self.running.extendleft(reversed(decode_requests))

        # prefill with remaining budget
        while self.waiting and num_seqs < self.max_num_seqs:
            request = self.waiting[0]

            if num_batched_tokens + len(request) >= self.max_num_batched_tokens or not self.kv_cache_manager.can_allocate(request, True):
                break

            self.waiting.popleft()
            self.running.append(request)
            request.status = Status.RUNNING
            self.kv_cache_manager.allocate_slots(request, True)
            prefill_requests.append(request)
            num_seqs += 1
            num_batched_tokens += len(request)

        return decode_requests, prefill_requests

    def postprocess(self, requests: list[Request], token_ids: list[int]):
        for request, token_id in zip(requests, token_ids):
            if (not request.ignore_eos and token_id == request.eos_token_id) \
                    or request.num_completion_tokens == request.max_new_tokens:
                request.status = Status.FINISHED
                self.running.remove(request)
