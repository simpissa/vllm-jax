from dataclasses import dataclass
from collections import deque
from typing import Dict

from vllm.src.request import Request

ceil_div = lambda a,b : (a + b - 1) // b

@dataclass
class Block:
    block_id : int
    ref_count : int

class KVCacheManager:
    
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.blocks: Dict[int, Block] = {i: Block(i, 0) for i in range(num_blocks)}
        self.free_block_queue: deque[Block] = deque(self.blocks.values())
        self.req_to_blocks: Dict[int, list[Block]] = {}
        
    def get_block_table(self, request: Request) -> list[Block]:
        if request.id not in self.req_to_blocks:
            self.req_to_blocks[request.id] = []
        return self.req_to_blocks[request.id]

    def can_allocate(self, request: Request, is_prefill: bool) -> bool:
        block_table = self.get_block_table(request)
        total_tokens = len(request)

        if is_prefill:
            needed_blocks = ceil_div(total_tokens, self.block_size)
            return needed_blocks <= len(self.free_block_queue)

        # decode
        assert len(block_table) > 0
        if total_tokens % self.block_size != 0:
            return True
        
        return len(self.free_block_queue) > 0

    def allocate_slots(self, request: Request, is_prefill: bool): 
        block_table = self.get_block_table(request)
        total_tokens = len(request)

        if is_prefill:
            needed_blocks = ceil_div(total_tokens, self.block_size)
            while len(block_table) < needed_blocks:
                block = self.free_block_queue.popleft()
                block.ref_count += 1
                block_table.append(block)
            return

        # decode
        assert len(block_table) > 0
        if total_tokens % self.block_size == 0:
            block = self.free_block_queue.popleft()
            block.ref_count += 1
            block_table.append(block)

    def deallocate(self, request: Request):
        block_table = self.get_block_table(request)
        for block in block_table:
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free_block_queue.append(block)
        
        block_table.clear()
    
