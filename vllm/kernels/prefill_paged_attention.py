"""Module containing prefill paged attention."""
from __future__ import annotations

import functools
import math
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
import numpy as np

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


def prefill_paged_attention_kernel(
        # inputs
        q_ref,  # [block_m, block_h, head_dim]
        q_positions_ref,  # [block_m]
        k_pages_ref,  # [total_num_pages, page_size, head_dim]
        k_scales_pages_ref,  # [total_num_pages, page_size]
        v_pages_ref,  # [total_num_pages, page_size, head_dim]
        v_scales_pages_ref,  # [total_num_pages, page_size]
        block_tables_ref,  # [pages_per_partition]
        seq_len_ref,  # [1]
        seq_start_pos_ref,  # [1]
        # outputs
        o_ref: Any,  # [block_m, block_h, head_dim]
        *residual_refs: Any,  # [block_m, block_h], [block_m, block_h]
        num_heads: int,
        block_m: int,
        q_blocks_per_seq: int,
        k_splits: int,
        pages_per_compute_block: int,
        mask_value: float,
        attn_logits_soft_cap: float | None,
):
    del num_heads
    pid2 = pl.program_id(2)

    block_h, head_dim = q_ref.shape[-2:]
    page_size = k_pages_ref.shape[-2]
    pages_per_partition = block_tables_ref.shape[0]
    block_k = pages_per_compute_block * page_size

    tmp = pid2 // q_blocks_per_seq
    q_block_id = pid2 % q_blocks_per_seq
    split_id = tmp % k_splits

    seq_len = seq_len_ref[0]
    seq_start_pos = seq_start_pos_ref[0]
    q_rel = q_block_id * block_m + jnp.arange(block_m, dtype=jnp.int32)
    q_in_range = q_rel < seq_len
    q_m = pl.ds(0, block_m)
    q_h = pl.ds(0, block_h)
    q = q_ref[q_m, q_h, :]
    q_pos = q_positions_ref[q_m]
    q = q * (1.0 / math.sqrt(head_dim))

    m_i = jnp.zeros((block_m, block_h), dtype=jnp.float32) + jnp.finfo(jnp.float32).min
    l_i = jnp.zeros((block_m, block_h), dtype=jnp.float32)
    o = jnp.zeros((block_m, block_h, head_dim), dtype=jnp.float32)

    start_page_idx = split_id * pages_per_partition
    end_page_idx = start_page_idx + pages_per_partition
    used_pages = pl.cdiv(seq_len, page_size)
    end_page_idx = jnp.minimum(end_page_idx, used_pages)

    log2e = math.log2(math.e)

    def _compute(start_page_idx, end_page_idx, o, m_i, l_i):
        def body(start_k, carry):
            o_prev, m_prev, l_prev = carry
            block_tables_slice = pl.ds(
                    start_k * pages_per_compute_block, pages_per_compute_block
            )
            block_tables = block_tables_ref[block_tables_slice]
            safe_block_tables = jnp.where(block_tables >= 0, block_tables, 0)
            k = k_pages_ref[safe_block_tables].reshape(block_k, head_dim)
            v = v_pages_ref[safe_block_tables].reshape(block_k, head_dim)

            if k_scales_pages_ref is not None:
                k = k.astype(q.dtype)

            q_2d = q.reshape(block_m * block_h, head_dim)
            uncapped_logits = pl.dot(q_2d, k.T).reshape(block_m, block_h, block_k)

            if k_scales_pages_ref is not None:
                k_scale = k_scales_pages_ref[safe_block_tables].reshape((1, 1, block_k))
                uncapped_logits *= k_scale.astype(uncapped_logits.dtype)

            if attn_logits_soft_cap is not None:
                logits = jnp.tanh(uncapped_logits / attn_logits_soft_cap)
                logits = logits * attn_logits_soft_cap
            else:
                logits = uncapped_logits

            curr_start_page_idx = start_page_idx + start_k * pages_per_compute_block
            curr_start_token_idx = curr_start_page_idx * page_size

            kv_rel = jnp.arange(block_k, dtype=jnp.int32) + curr_start_token_idx
            kv_pos = kv_rel + seq_start_pos
            kv_valid = kv_rel < seq_len
            causal = kv_pos[None, :] <= q_pos[:, None]
            mask = q_in_range[:, None] & kv_valid[None, :] & causal
            mask = lax.broadcast_in_dim(mask, (block_m, block_h, block_k), (0, 2))
            logits = jnp.where(mask, logits, mask_value)

            m_curr = logits.max(axis=-1)
            m_next = jnp.maximum(m_prev, m_curr)
            correction = jnp.exp2((m_prev - m_next) * log2e)
            l_prev_corr = correction * l_prev
            s_curr = jnp.exp2((logits - m_next[:, :, None]) * log2e)
            l_curr = s_curr.sum(axis=-1)
            l_next = l_prev_corr + l_curr
            o_prev_corr = correction[:, :, None] * o_prev

            if v_scales_pages_ref is not None:
                v_scale = v_scales_pages_ref[safe_block_tables].reshape((1, 1, block_k))
                s_curr *= v_scale.astype(s_curr.dtype)
                v = v.astype(s_curr.dtype)

            s_2d = s_curr.reshape(block_m * block_h, block_k)
            o_curr = pl.dot(s_2d.astype(v.dtype), v).reshape(block_m, block_h, head_dim)
            o_curr = o_curr.astype(jnp.float32)
            o_next = o_prev_corr + o_curr

            return o_next, m_next, l_next

        max_it = pl.cdiv(end_page_idx - start_page_idx, pages_per_compute_block)
        return lax.fori_loop(0, max_it, body, (o, m_i, l_i))

    o, m_i, l_i = lax.cond(
            start_page_idx >= end_page_idx,
            lambda: (o, m_i, l_i),
            lambda: _compute(start_page_idx, end_page_idx, o, m_i, l_i),
    )

    valid_q = lax.broadcast_in_dim(q_in_range, (block_m, block_h), (0,))
    o = jnp.where(valid_q[:, :, None], o, 0)
    l_i = jnp.where(valid_q, l_i, 0)
    m_i = jnp.where(valid_q, m_i, jnp.finfo(jnp.float32).min)

    o_ref[...] = o.astype(o_ref.dtype)
    if residual_refs is not None:
        l_ref, m_ref = residual_refs
        l_ref[...] = l_i
        m_ref[...] = m_i


def _largest_divisor_at_most(value: int, upper: int) -> int:
    for candidate in (16, 8, 4, 2, 1):
        if candidate <= upper and value % candidate == 0:
            return candidate
    return 1


def prefill_paged_attention_unbatched(
        q: jax.Array,
        k_pages: jax.Array,
        v_pages: jax.Array,
        block_tables: jax.Array,
        cu_seqlens: jax.Array,
        slot_mapping: jax.Array,
        input_positions: jax.Array,
        max_seqlen_bucket: int,
        k_scales_pages: jax.Array | None = None,
        v_scales_pages: jax.Array | None = None,
        *,
        block_m: int,
        block_h: int,
        pages_per_compute_block: int,
        k_splits: int,
        num_warps: int,
        num_stages: int,
        interpret: bool,
        debug: bool,
        mask_value: float,
        attn_logits_soft_cap: float | None,
) -> jax.Array:
    del slot_mapping

    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads, total_num_pages, page_size, head_dim_k = k_pages.shape
    batch, pages_per_sequence = block_tables.shape

    if k_pages.shape != v_pages.shape:
        raise ValueError(
                f"k_pages and v_pages must have the same shape. Got {k_pages.shape} and"
                f" {v_pages.shape}"
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(
                "Number of Q heads must be divisible by number of KV heads. Got"
                f" {num_heads} and {num_kv_heads}."
        )
    if head_dim_k != head_dim:
        raise ValueError(
                "head_dim of Q must be the same as that of K/V. Got"
                f" {head_dim} and {head_dim_k}."
        )
    if cu_seqlens.shape != (batch + 1,):
        raise ValueError("cu_seqlens must have shape [batch + 1]")
    if block_tables.dtype != jnp.int32:
        raise ValueError("block_tables must be int32")

    q_heads_per_kv_head = num_heads // num_kv_heads
    q_heads_per_kv_head_orig = q_heads_per_kv_head
    q_reshaped = q.reshape(total_tokens, num_kv_heads, q_heads_per_kv_head, head_dim)

    if q_heads_per_kv_head % block_h:
        pad_heads = -q_heads_per_kv_head % block_h
        q_reshaped = jnp.pad(q_reshaped, ((0, 0), (0, 0), (0, pad_heads), (0, 0)))
        q_heads_per_kv_head += pad_heads

    head_splits = pl.cdiv(q_heads_per_kv_head, block_h)
    q_blocks_per_seq = pl.cdiv(max_seqlen_bucket, block_m)
    k_splits_eff = _largest_divisor_at_most(pages_per_sequence, k_splits)
    pages_per_partition = pages_per_sequence // k_splits_eff
    pages_per_compute_block = min(pages_per_partition, pages_per_compute_block)
    pages_per_compute_block = _largest_divisor_at_most(
            pages_per_partition, pages_per_compute_block
    )

    seq_starts = cu_seqlens[:-1].astype(jnp.int32)
    seq_ends = cu_seqlens[1:].astype(jnp.int32)
    seq_lens = seq_ends - seq_starts
    seq_start_positions = input_positions[seq_starts]

    q_offsets = jnp.arange(max_seqlen_bucket, dtype=jnp.int32)[None, :]
    q_idx = seq_starts[:, None] + q_offsets
    q_valid = q_idx < seq_ends[:, None]
    safe_q_idx = jnp.where(q_valid, q_idx, 0)
    q_seq = q_reshaped[safe_q_idx]
    q_pos_seq = input_positions[safe_q_idx]

    q_seq = jnp.transpose(q_seq, (0, 1, 2, 3, 4))

    block_tables_split = block_tables.reshape(batch, k_splits_eff, pages_per_partition)

    grid = (num_kv_heads, head_splits, batch * k_splits_eff * q_blocks_per_seq)
    kernel = functools.partial(
            prefill_paged_attention_kernel,
            num_heads=q_heads_per_kv_head,
            block_m=block_m,
            q_blocks_per_seq=q_blocks_per_seq,
            k_splits=k_splits_eff,
            pages_per_compute_block=pages_per_compute_block,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
    )

    if k_scales_pages is not None:
        assert k_scales_pages.shape == (num_kv_heads, total_num_pages, page_size)
        k_scales_spec = pl.BlockSpec((None, total_num_pages, page_size),
                                 lambda h, hs, p: (h, 0, 0))
    else:
        k_scales_spec = None
    if v_scales_pages is not None:
        assert v_scales_pages.shape == (num_kv_heads, total_num_pages, page_size)
        v_scales_spec = pl.BlockSpec((None, total_num_pages, page_size),
                                 lambda h, hs, p: (h, 0, 0))
    else:
        v_scales_spec = None

    def _seq_from_pid(pid):
        return pid // (k_splits_eff * q_blocks_per_seq)

    def _split_from_pid(pid):
        return (pid // q_blocks_per_seq) % k_splits_eff

    def _qblock_from_pid(pid):
        return pid % q_blocks_per_seq

    o, l, m = pl.pallas_call(
            kernel,
            grid=grid,
            in_specs=[
                    pl.BlockSpec(
                            (None, block_m, None, block_h, head_dim),
                            lambda h, hs, p: (
                                    _seq_from_pid(p),
                                    _qblock_from_pid(p) * block_m,
                                    h,
                                    hs * block_h,
                                    0,
                            ),
                    ),
                    pl.BlockSpec(
                            (None, block_m),
                            lambda h, hs, p: (
                                    _seq_from_pid(p),
                                    _qblock_from_pid(p) * block_m,
                            ),
                    ),
                    pl.BlockSpec(
                            (None, total_num_pages, page_size, head_dim),
                            lambda h, hs, p: (h, 0, 0, 0),
                    ),
                    k_scales_spec,
                    pl.BlockSpec(
                            (None, total_num_pages, page_size, head_dim),
                            lambda h, hs, p: (h, 0, 0, 0),
                    ),
                    v_scales_spec,
                    pl.BlockSpec(
                            (None, None, pages_per_partition),
                            lambda h, hs, p: (_seq_from_pid(p), _split_from_pid(p), 0),
                    ),
                    pl.BlockSpec((1,), lambda h, hs, p: (_seq_from_pid(p),)),
                    pl.BlockSpec((1,), lambda h, hs, p: (_seq_from_pid(p),)),
            ],
            out_specs=[
                    pl.BlockSpec(
                            (None, None, None, block_m, block_h, head_dim),
                            lambda h, hs, p: (p, h, hs, 0, 0, 0),
                    ),
                    pl.BlockSpec(
                            (None, None, None, block_m, block_h),
                            lambda h, hs, p: (p, h, hs, 0, 0),
                    ),
                    pl.BlockSpec(
                            (None, None, None, block_m, block_h),
                            lambda h, hs, p: (p, h, hs, 0, 0),
                    ),
            ],
            out_shape=[
                    jax.ShapeDtypeStruct(
                            (batch * k_splits_eff * q_blocks_per_seq, num_kv_heads, head_splits,
               block_m, block_h, head_dim),
                            dtype=q.dtype,
                    ),
                    jax.ShapeDtypeStruct(
                            (batch * k_splits_eff * q_blocks_per_seq, num_kv_heads, head_splits,
               block_m, block_h),
                            dtype=jnp.float32,
                    ),
                    jax.ShapeDtypeStruct(
                            (batch * k_splits_eff * q_blocks_per_seq, num_kv_heads, head_splits,
               block_m, block_h),
                            dtype=jnp.float32,
                    ),
            ],
            debug=debug,
            interpret=interpret,
            compiler_params=plgpu.CompilerParams(
                    num_warps=num_warps, num_stages=num_stages
            ),
            name=f"prefill_paged_attention_{block_m=}_{block_h=}_{pages_per_compute_block=}",
    )(
            q_seq,
            q_pos_seq,
            k_pages,
            k_scales_pages,
            v_pages,
            v_scales_pages,
            block_tables_split,
            seq_lens,
            seq_start_positions,
    )

    o = o.reshape(batch, k_splits_eff, q_blocks_per_seq, num_kv_heads, head_splits,
                                block_m, block_h, head_dim)
    l = l.reshape(batch, k_splits_eff, q_blocks_per_seq, num_kv_heads, head_splits,
                                block_m, block_h)
    m = m.reshape(batch, k_splits_eff, q_blocks_per_seq, num_kv_heads, head_splits,
                                block_m, block_h)

    m_next = m.max(axis=1)
    correction = jnp.exp(m - m_next[:, None])
    o = o * correction[..., None].astype(o.dtype)
    l_next = (l * correction).sum(axis=1)
    eps = jnp.finfo(l_next.dtype).eps
    o = o.sum(axis=1) / ((l_next[..., None] + eps).astype(o.dtype))

    o = jnp.transpose(o, (0, 1, 4, 2, 3, 5, 6))
    o = o.reshape(batch, q_blocks_per_seq * block_m, num_kv_heads,
                                q_heads_per_kv_head, head_dim)
    o = o[:, :max_seqlen_bucket, :, :q_heads_per_kv_head_orig, :]
    o = o.reshape(batch, max_seqlen_bucket, num_heads, head_dim)

    flat_positions = jnp.where(q_valid, q_idx, total_tokens).reshape(-1)
    flat_values = o.reshape(-1, num_heads, head_dim)
    output = jnp.zeros((total_tokens + 1, num_heads, head_dim), dtype=q.dtype)
    output = output.at[flat_positions].set(flat_values, mode="drop")
    return output[:total_tokens]


@functools.partial(
        jax.jit,
        static_argnames=[
                "max_seqlen_bucket",
                "block_m",
                "block_h",
                "pages_per_compute_block",
                "k_splits",
                "num_warps",
                "num_stages",
                "interpret",
                "debug",
                "mask_value",
                "attn_logits_soft_cap",
        ],
)
def prefill_paged_attention(
        q: jax.Array,
        k_pages: jax.Array,
        v_pages: jax.Array,
        block_tables: jax.Array,
        cu_seqlens: jax.Array,
        slot_mapping: jax.Array,
        input_positions: jax.Array,
        *,
        max_seqlen_bucket: int,
        k_scales_pages: jax.Array | None = None,
        v_scales_pages: jax.Array | None = None,
        block_m: int = 16,
        block_h: int = 16,
        pages_per_compute_block: int = 8,
        k_splits: int = 8,
        num_warps: int = 8,
        num_stages: int = 2,
        interpret: bool = False,
        debug: bool = False,
        mask_value: float = DEFAULT_MASK_VALUE,
        attn_logits_soft_cap: float | None = None,
) -> jax.Array:
    if block_h % 16:
        raise ValueError(f"block_h must divisible by 16, but is {block_h}.")
    return prefill_paged_attention_unbatched(
            q,
            k_pages,
            v_pages,
            block_tables,
            cu_seqlens,
            slot_mapping,
            input_positions,
            max_seqlen_bucket,
            k_scales_pages,
            v_scales_pages,
            block_m=block_m,
            block_h=block_h,
            pages_per_compute_block=pages_per_compute_block,
            k_splits=k_splits,
            num_warps=num_warps,
            num_stages=num_stages,
            interpret=interpret,
            debug=debug,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
    )


def prefill_paged_attention_reference(
        q: jax.Array,
        k_pages: jax.Array,
        v_pages: jax.Array,
        block_tables: jax.Array,
        cu_seqlens: jax.Array,
        slot_mapping: jax.Array,
        input_positions: jax.Array,
        *,
        max_seqlen_bucket: int,
        mask_value: float = DEFAULT_MASK_VALUE,
        attn_logits_soft_cap: float | None = None,
) -> jax.Array:
    del block_tables
    del max_seqlen_bucket

    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads, _, _, _ = k_pages.shape
    flat_k = k_pages.reshape(num_kv_heads, -1, head_dim)
    flat_v = v_pages.reshape(num_kv_heads, -1, head_dim)

    safe_slots = jnp.where(slot_mapping >= 0, slot_mapping, 0).astype(jnp.int32)
    k = jnp.take(flat_k, safe_slots, axis=1)
    v = jnp.take(flat_v, safe_slots, axis=1)
    k = jnp.transpose(k, (1, 0, 2))
    v = jnp.transpose(v, (1, 0, 2))
    if num_heads != num_kv_heads:
        repeat = num_heads // num_kv_heads
        k = jnp.repeat(k, repeat, axis=1)
        v = jnp.repeat(v, repeat, axis=1)

    token_idx = jnp.arange(total_tokens, dtype=jnp.int32)
    seq_ids = jnp.sum(token_idx[:, None] >= cu_seqlens[None, 1:], axis=1)
    token_valid = slot_mapping >= 0
    causal = input_positions[:, None] >= input_positions[None, :]
    same_seq = seq_ids[:, None] == seq_ids[None, :]
    mask = same_seq & causal & token_valid[:, None] & token_valid[None, :]

    q_t = jnp.transpose(q, (1, 0, 2))
    k_t = jnp.transpose(k, (1, 0, 2))
    scores = jnp.einsum("hqd,hkd->hqk", q_t, k_t) / jnp.sqrt(q.shape[-1])
    if attn_logits_soft_cap is not None:
        scores = jnp.tanh(scores / attn_logits_soft_cap)
        scores = scores * attn_logits_soft_cap
    scores = jnp.where(mask[None, :, :], scores, mask_value)
    weights = jax.nn.softmax(scores, axis=-1)
    weights = jnp.where(token_valid[None, :, None], weights, 0)
    v_t = jnp.transpose(v, (1, 0, 2))
    out = jnp.einsum("hqk,hkd->hqd", weights, v_t)
    return jnp.transpose(out, (1, 0, 2))
