from dataclasses import dataclass
from typing import Any
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, set_mesh
from flax import nnx, struct
from transformers import Qwen3Config

from vllm.config import Config
from vllm.model.qwen3 import (DecodeAttentionMetadata, PrefillAttentionMetadata,
                              Qwen3ForCausalLM)
from vllm.sampler import Sampler
from vllm.utils.hf_convert import load_hf_weights_into_model
from .request import Request
from .kv_cache_manager import KVCacheManager

@dataclass
class DecodeBatch:
    input_ids: jnp.ndarray
    positions: jnp.ndarray
    slot_mapping: jnp.ndarray
    block_tables: jnp.ndarray

    context_lens: jnp.ndarray

@dataclass
class PrefillBatch:
    input_ids: jnp.ndarray
    positions: jnp.ndarray
    slot_mapping: jnp.ndarray

    block_tables: jnp.ndarray

    cu_seqlens_q: jnp.ndarray
    cu_seqlens_k: jnp.ndarray
    max_seqlen_q: jnp.ndarray
    max_seqlen_k: jnp.ndarray

class ModelRunner:

    def __init__(self, config: Config, kv_cache_manager: KVCacheManager, device: Any):
        self.block_size = config.block_size
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_blocks_per_seq = (self.max_num_batched_tokens + self.block_size - 1) // self.block_size
        self.kv_cache_manager = kv_cache_manager
        self.device = device
        self.model_config = Qwen3Config.from_pretrained(config.model)

        # create model on cpu first
        cpu_device = jax.devices("cpu")[0]
        cpu_mesh = Mesh(
            mesh_utils.create_device_mesh((1, ), devices=[cpu_device]),
            ("model", ),
        )
        self.mesh = Mesh(
            mesh_utils.create_device_mesh((1, ), devices=[self.device]),
            ("model", ),
        )
        set_mesh(cpu_mesh)
        self.model = Qwen3ForCausalLM(
            self.model_config,
            jax.random.PRNGKey(0),
            mesh=cpu_mesh,
        )
        load_hf_weights_into_model(self.model, config.model, device=cpu_device)

        # move to device
        if self.device != cpu_device:
            state = nnx.filter_state(nnx.state(self.model), nnx.Param)
            moved_state = nnx.map_state(
                lambda _, value: jax.device_put(value, device=self.device),
                state,
            )
            nnx.update(self.model, moved_state)
        set_mesh(self.mesh)
        
        print(f"Loaded HF weights for {config.model}")
        self.kv_caches = self._init_kv_cache(config.num_blocks)
        self.sampler = Sampler()
        self._warmup()

    def _warmup(self):
        input_ids = self._to_device(jnp.asarray([0], dtype=jnp.int32))
        positions = self._to_device(jnp.asarray([0], dtype=jnp.int32))
        slot_mapping = self._to_device(jnp.asarray([0], dtype=jnp.int32))
        block_tables = self._to_device(
            jnp.asarray([[0] + [-1] * (self.max_blocks_per_seq - 1)], dtype=jnp.int32)
        )

        prefill_batch = PrefillBatch(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            cu_seqlens_q=self._to_device(jnp.asarray([0, 1], dtype=jnp.int32)),
            cu_seqlens_k=self._to_device(jnp.asarray([0, 1], dtype=jnp.int32)),
            max_seqlen_q=self._to_device(jnp.asarray(1, dtype=jnp.int32)),
            max_seqlen_k=self._to_device(jnp.asarray(1, dtype=jnp.int32)),
        )
        self.forward_prefill(prefill_batch)

        decode_batch = DecodeBatch(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=self._to_device(jnp.asarray([1], dtype=jnp.int32)),
        )
        self.forward_decode(decode_batch)

    def _init_kv_cache(self, num_blocks: int) -> list[jax.Array]:
        num_layers = self.model_config.num_hidden_layers
        num_kv_heads = self.model_config.num_key_value_heads
        head_dim = getattr(self.model_config, "head_dim",
                           self.model_config.hidden_size // self.model_config.num_attention_heads)
        cache_shape = (2, num_blocks, self.block_size, num_kv_heads, head_dim)
        return [
            self._to_device(jnp.zeros(cache_shape, dtype=jnp.float32))
            for _ in range(num_layers)
        ]

    def _to_device(self, value):
        return jax.device_put(value, device=self.device)

    def pad_block_tables(self, block_tables: list[list[int]]) -> jnp.ndarray:
        max_blocks = self.max_blocks_per_seq
        if not block_tables:
            return jnp.empty((0, max_blocks), dtype=jnp.int32)
        padded_tables = [
            block_table + [-1] * (max_blocks - len(block_table))
            for block_table in block_tables
        ]
        return jnp.asarray(padded_tables, dtype=jnp.int32)


    def prepare_prefill(self, requests: list[Request]) -> PrefillBatch:
        input_ids = []
        positions = []
        slot_mapping = []
        block_tables = []
        seqlens = []

        for request in requests:
            request_len = len(request)
            seqlens.append(request_len)
            input_ids.extend(request.token_ids)
            positions.extend(range(request_len))

            block_table = self.kv_cache_manager.get_block_table(request)
            block_tables.append([block.block_id for block in block_table])
            for token_idx in range(request_len):
                block_idx = token_idx // self.block_size
                block_id = block_table[block_idx].block_id
                block_offset = token_idx % self.block_size
                slot_mapping.append(block_id * self.block_size + block_offset)

        cu_seqlens = [0]
        for seqlen in seqlens:
            cu_seqlens.append(cu_seqlens[-1] + seqlen)

        max_seqlen = max(seqlens)
        block_tables = self.pad_block_tables(block_tables)
        return PrefillBatch(
            input_ids=self._to_device(jnp.asarray(input_ids, dtype=jnp.int32)),
            positions=self._to_device(jnp.asarray(positions, dtype=jnp.int32)),
            slot_mapping=self._to_device(jnp.asarray(slot_mapping, dtype=jnp.int32)),
            block_tables=self._to_device(block_tables),
            cu_seqlens_q=self._to_device(jnp.asarray(cu_seqlens, dtype=jnp.int32)),
            cu_seqlens_k=self._to_device(jnp.asarray(cu_seqlens, dtype=jnp.int32)),
            max_seqlen_q=self._to_device(jnp.asarray(max_seqlen, dtype=jnp.int32)),
            max_seqlen_k=self._to_device(jnp.asarray(max_seqlen, dtype=jnp.int32)),
        )

    def prepare_decode(self, requests: list[Request]) -> DecodeBatch:
        input_ids = []
        positions = []
        context_lens = [] 
        slot_mapping = []
        block_tables = []

        for request in requests:
            input_ids.append(request.get_last_token_id())
            positions.append(len(request)-1)
            context_lens.append(len(request))

            block_table = self.kv_cache_manager.get_block_table(request)
            block_tables.append([block.block_id for block in block_table])
            last_block = block_table[-1]
            block_offset = (len(request) - 1) % self.block_size
            slot_mapping.append(last_block.block_id * self.block_size + block_offset)

        block_tables = self.pad_block_tables(block_tables)
        return DecodeBatch(
            input_ids=self._to_device(jnp.asarray(input_ids, dtype=jnp.int32)),
            positions=self._to_device(jnp.asarray(positions, dtype=jnp.int32)),
            slot_mapping=self._to_device(jnp.asarray(slot_mapping, dtype=jnp.int32)),
            block_tables=self._to_device(block_tables),
            context_lens=self._to_device(jnp.asarray(context_lens, dtype=jnp.int32)),
        )

    def forward_decode(self, batch: DecodeBatch) -> jax.Array:
        metadata = DecodeAttentionMetadata(
            input_positions=batch.positions,
            slot_mapping=batch.slot_mapping,
            block_tables=batch.block_tables,
            context_lens=batch.context_lens,
        )
        self.kv_caches, hidden_states, _ = self.model(
            self.kv_caches,
            batch.input_ids,
            metadata,
        )
        return self.model.compute_logits(hidden_states)

    def forward_prefill(self, batch: PrefillBatch):
        metadata = PrefillAttentionMetadata(
            input_positions=batch.positions,
            slot_mapping=batch.slot_mapping,
            block_tables=batch.block_tables,
            cu_seqlens_q=batch.cu_seqlens_q,
            cu_seqlens_k=batch.cu_seqlens_k,
            max_seqlen_q=batch.max_seqlen_q,
            max_seqlen_k=batch.max_seqlen_k,
        )
        self.kv_caches, _, _ = self.model(
            self.kv_caches,
            batch.input_ids,
            metadata,
        )

    def run(self,
            decode_requests: list[Request],
            prefill_requests: list[Request],
            rng_key: jax.Array) -> tuple[list[int], jax.Array]:
        decode_outputs = []
        if decode_requests:
            decode_batch = self.prepare_decode(decode_requests)
            logits = self.forward_decode(decode_batch)
            for request, request_logits in zip(decode_requests, logits):
                token_ids, rng_key = self.sampler.sample(
                    request_logits,
                    request.sampling_params,
                    rng_key,
                )
                token_id = int(jax.device_get(token_ids)[0])
                request.token_ids.append(token_id)
                request.num_completion_tokens += 1
                decode_outputs.append(token_id)

        if prefill_requests:
            prefill_batch = self.prepare_prefill(prefill_requests)
            self.forward_prefill(prefill_batch)

        return decode_outputs, rng_key
