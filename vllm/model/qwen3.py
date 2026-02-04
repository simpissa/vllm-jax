from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx, struct
from transformers import Qwen3Config, modeling_flax_utils

init_fn = nnx.initializers.uniform()


@struct.dataclass
class DecodeAttentionMetadata:
    input_positions: jax.Array
    slot_mapping: jax.Array
    block_tables: jax.Array
    context_lens: jax.Array


@struct.dataclass
class PrefillAttentionMetadata:
    input_positions: jax.Array
    slot_mapping: jax.Array
    block_tables: jax.Array
    cu_seqlens_q: jax.Array
    cu_seqlens_k: jax.Array
    max_seqlen_q: jax.Array
    max_seqlen_k: jax.Array


AttentionMetadata = Union[DecodeAttentionMetadata, PrefillAttentionMetadata]


def get_padded_head_dim(head_dim: int) -> int:
    return head_dim


def get_padded_num_heads(num_heads: int, _: int) -> int:
    return num_heads


def _maybe_scale_positions(positions: jax.Array,
                           rope_scaling: Optional[dict]) -> jax.Array:
    if rope_scaling is None:
        return positions
    scaling_type = rope_scaling.get("type")
    factor = rope_scaling.get("factor", 1.0)
    if scaling_type in {"linear", "dynamic"}:
        return positions / factor
    return positions


def apply_rope(x: jax.Array, positions: jax.Array, head_dim: int,
               rope_theta: float, rope_scaling: Optional[dict]) -> jax.Array:
    positions = _maybe_scale_positions(positions, rope_scaling)
    positions = positions.astype(jnp.float32)
    inv_freq = 1.0 / (rope_theta**(jnp.arange(0, head_dim, 2,
                                             dtype=jnp.float32) / head_dim))
    freqs = jnp.einsum("t,d->td", positions, inv_freq)
    cos = jnp.cos(freqs)[:, None, :]
    sin = jnp.sin(freqs)[:, None, :]

    x_rope = x[..., :head_dim]
    x_pass = x[..., head_dim:]
    x1 = x_rope[..., 0::2]
    x2 = x_rope[..., 1::2]
    x_rope = jnp.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos),
                       axis=-1)
    x_rope = x_rope.reshape(x_rope.shape[:-2] + (head_dim, ))
    if x_pass.shape[-1] == 0:
        return x_rope
    return jnp.concatenate((x_rope, x_pass), axis=-1)


def _repeat_kv(x: jax.Array, num_heads: int) -> jax.Array:
    if x.shape[1] == num_heads:
        return x
    repeat = num_heads // x.shape[1]
    return jnp.repeat(x, repeat, axis=1)


def _scaled_dot_product_attention(q: jax.Array,
                                  k: jax.Array,
                                  v: jax.Array,
                                  causal: bool) -> jax.Array:
    q_t = jnp.transpose(q, (1, 0, 2))
    k_t = jnp.transpose(k, (1, 0, 2))
    scores = jnp.einsum("hqd,hkd->hqk", q_t, k_t) / jnp.sqrt(q.shape[-1])
    if causal:
        mask = jnp.tril(jnp.ones((q.shape[0], k.shape[0]), dtype=bool))
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores, axis=-1)
    v_t = jnp.transpose(v, (1, 0, 2))
    out = jnp.einsum("hqk,hkd->hqd", weights, v_t)
    return jnp.transpose(out, (1, 0, 2))


def _update_kv_cache(kv_cache: jax.Array, slot_mapping: jax.Array,
                     k: jax.Array, v: jax.Array) -> jax.Array:
    flat_cache = kv_cache.reshape(kv_cache.shape[0], -1, kv_cache.shape[-2],
                                  kv_cache.shape[-1])
    flat_cache = flat_cache.at[0, slot_mapping].set(k)
    flat_cache = flat_cache.at[1, slot_mapping].set(v)
    return flat_cache.reshape(kv_cache.shape)


def _build_slot_indices(block_table: jax.Array, context_len: int,
                        block_size: int) -> jax.Array:
    num_blocks = (context_len + block_size - 1) // block_size
    slots = []
    for block_id in block_table[:num_blocks]:
        block_id = int(block_id)
        slots.append(
            jnp.arange(block_size, dtype=jnp.int32) + block_id * block_size)
    if not slots:
        return jnp.empty((0, ), dtype=jnp.int32)
    return jnp.concatenate(slots, axis=0)[:context_len]


def _decode_attention(q: jax.Array, kv_cache: jax.Array,
                      metadata: DecodeAttentionMetadata) -> jax.Array:
    num_heads = q.shape[1]
    num_kv_heads = kv_cache.shape[-2]
    head_dim = kv_cache.shape[-1]
    block_size = kv_cache.shape[2]
    flat_cache = kv_cache.reshape(2, -1, num_kv_heads, head_dim)

    outputs = []
    for i in range(q.shape[0]):
        context_len = int(metadata.context_lens[i])
        block_table = metadata.block_tables[i]
        slots = _build_slot_indices(block_table, context_len, block_size)
        k = flat_cache[0, slots]
        v = flat_cache[1, slots]
        k = _repeat_kv(k, num_heads)
        v = _repeat_kv(v, num_heads)
        outputs.append(_scaled_dot_product_attention(q[i:i + 1], k, v, False))
    if not outputs:
        return jnp.empty((0, num_heads, head_dim), dtype=q.dtype)
    return jnp.concatenate(outputs, axis=0)


def _prefill_attention(q: jax.Array, k: jax.Array, v: jax.Array,
                       metadata: PrefillAttentionMetadata) -> jax.Array:
    num_heads = q.shape[1]
    k = _repeat_kv(k, num_heads)
    v = _repeat_kv(v, num_heads)
    cu_seqlens = metadata.cu_seqlens_q

    outputs = []
    start = 0
    for idx in range(cu_seqlens.shape[0] - 1):
        end = int(cu_seqlens[idx + 1])
        seq_q = q[start:end]
        seq_k = k[start:end]
        seq_v = v[start:end]
        outputs.append(_scaled_dot_product_attention(seq_q, seq_k, seq_v, True))
        start = end
    if not outputs:
        return jnp.empty((0, num_heads, q.shape[-1]), dtype=q.dtype)
    return jnp.concatenate(outputs, axis=0)


def attention(kv_cache: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array,
              metadata: AttentionMetadata) -> Tuple[jax.Array, jax.Array]:
    kv_cache = _update_kv_cache(kv_cache, metadata.slot_mapping, k, v)
    if isinstance(metadata, PrefillAttentionMetadata):
        outputs = _prefill_attention(q, k, v, metadata)
    else:
        outputs = _decode_attention(q, kv_cache, metadata)
    return kv_cache, outputs


class Qwen3Attention(nnx.Module):

    def __init__(self, config: Qwen3Config, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh=None):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.rms_norm_eps = config.rms_norm_eps

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = get_padded_head_dim(self.head_dim_original)

        sharding_size = mesh.shape["model"] if mesh is not None else 1
        self.num_heads = get_padded_num_heads(self.num_heads, sharding_size)
        self.num_kv_heads = get_padded_num_heads(self.num_kv_heads,
                                                 sharding_size)

        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
        )
        self.q_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
        )
        self.k_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
        )
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
        )

    def __call__(self, kv_cache: Optional[jax.Array], x: jax.Array,
                 attention_metadata: AttentionMetadata) -> Tuple[jax.Array, jax.Array]:
        q = self.q_proj(x)
        q = self.q_norm(q)
        q = apply_rope(q, attention_metadata.input_positions,
                       self.head_dim_original, self.rope_theta,
                       self.rope_scaling)

        k = self.k_proj(x)
        k = self.k_norm(k)
        k = apply_rope(k, attention_metadata.input_positions,
                       self.head_dim_original, self.rope_theta,
                       self.rope_scaling)

        v = self.v_proj(x)
        kv_cache, outputs = attention(kv_cache, q, k, v, attention_metadata)
        o = self.o_proj(outputs)
        return kv_cache, o


class Qwen3MLP(nnx.Module):

    def __init__(self, config: Qwen3Config, dtype: jnp.dtype, rng: nnx.Rngs):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        act = config.hidden_act

        self.gate_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
        )
        self.up_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
        )
        self.down_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )
        self.act_fn = modeling_flax_utils.ACT2FN[act]

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fused = gate * up
        return self.down_proj(fused)


class Qwen3DecoderLayer(nnx.Module):

    def __init__(self, config: Qwen3Config, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh=None):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.input_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.self_attn = Qwen3Attention(config=config,
                                        dtype=dtype,
                                        rng=rng,
                                        mesh=mesh)
        self.post_attention_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.mlp = Qwen3MLP(
            config=config,
            dtype=dtype,
            rng=rng,
        )

    def __call__(self, kv_cache: jax.Array, x: jax.Array,
                 attention_metadata: AttentionMetadata) -> Tuple[jax.Array, jax.Array]:
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(
            kv_cache,
            hidden_states,
            attention_metadata,
        )
        attn_output += x

        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        outputs = self.mlp(attn_output)
        outputs = residual + outputs
        return kv_cache, outputs


class Qwen3Model(nnx.Module):

    def __init__(self, config: Qwen3Config, rng: nnx.Rngs, mesh=None,
                 dtype: Optional[jnp.dtype] = None,
                 vocab_size: Optional[int] = None) -> None:
        dtype = dtype or jnp.float32
        vocab_size = vocab_size or config.vocab_size
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.embed = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )
        self.layers = nnx.List([
            Qwen3DecoderLayer(
                config=config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
            ) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        if config.tie_word_embeddings:
            self.lm_head = self.embed.embedding
        else:
            self.lm_head = nnx.Param(
                init_fn(rng.params(), (hidden_size, vocab_size), dtype),
                sharding=(None, "model"),
            )

    def __call__(self, kv_caches: List[jax.Array], input_ids: jax.Array,
                 attention_metadata: AttentionMetadata) -> Tuple[List[jax.Array], jax.Array]:
        x = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(
                kv_cache,
                x,
                attention_metadata,
            )
            kv_caches[i] = kv_cache
        x = self.norm(x)
        return kv_caches, x


class Qwen3ForCausalLM(nnx.Module):

    def __init__(self, config: Qwen3Config, rng_key: jax.Array, mesh=None,
                 dtype: Optional[jnp.dtype] = None,
                 vocab_size: Optional[int] = None) -> None:
        self.config = config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Qwen3Model(
            config=config,
            rng=self.rng,
            mesh=mesh,
            dtype=dtype,
            vocab_size=vocab_size,
        )

    def __call__(self, kv_caches: List[jax.Array], input_ids: jax.Array,
                 attention_metadata: AttentionMetadata, *args
                 ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
        )
        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if self.config.tie_word_embeddings:
            return jnp.dot(hidden_states, self.model.lm_head.value.T)
        return jnp.dot(hidden_states, self.model.lm_head.value)
