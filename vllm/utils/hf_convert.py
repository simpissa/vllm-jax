import json
import os
import re
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, set_mesh
from safetensors.numpy import load_file
from transformers import Qwen3Config
from transformers.utils.hub import cached_file

from vllm.model.qwen3 import Qwen3ForCausalLM as JaxQwen3ForCausalLM

FlatKey = Tuple[object, ...]


def _flat_key_from_str(path: str) -> FlatKey:
    parts = []
    for part in path.split("."):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part)
    return tuple(parts)


def _convert_weight(key: str, value, config: Qwen3Config) -> jax.Array:
    value = jnp.asarray(value)
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim",
                       config.hidden_size // config.num_attention_heads)

    if re.search(r"q_proj\.weight", key):
        return value.T.reshape(config.hidden_size, num_heads, head_dim)
    if re.search(r"k_proj\.weight", key) or re.search(r"v_proj\.weight", key):
        return value.T.reshape(config.hidden_size, num_kv_heads, head_dim)
    if re.search(r"o_proj\.weight", key):
        return value.T.reshape(num_heads, head_dim, config.hidden_size)
    if re.search(r"mlp\.down_proj\.weight", key):
        return value.T
    if re.search(r"mlp\.(gate|up)_proj\.weight", key):
        return value.T
    if re.search(r"embed_tokens\.weight", key):
        return value
    if re.search(r"lm_head\.weight", key):
        return value.T
    if re.search(r"(input_layernorm|post_attention_layernorm|norm)\.weight", key):
        return value
    if re.search(r"self_attn\.(q_norm|k_norm)\.weight", key):
        return value

    raise ValueError(f"Unsupported weight key: {key}")


_HF_TO_NNX = {
    r"model\.embed_tokens\.weight": "model.embed.embedding",
    r"model\.layers\.([0-9]+)\.input_layernorm\.weight":
    r"model.layers.\1.input_layernorm.scale",
    r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight":
    r"model.layers.\1.post_attention_layernorm.scale",
    r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight":
    r"model.layers.\1.self_attn.q_norm.scale",
    r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight":
    r"model.layers.\1.self_attn.k_norm.scale",
    r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight":
    r"model.layers.\1.self_attn.q_proj.kernel",
    r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight":
    r"model.layers.\1.self_attn.k_proj.kernel",
    r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight":
    r"model.layers.\1.self_attn.v_proj.kernel",
    r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight":
    r"model.layers.\1.self_attn.o_proj.kernel",
    r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight":
    r"model.layers.\1.mlp.gate_proj.kernel",
    r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight":
    r"model.layers.\1.mlp.up_proj.kernel",
    r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight":
    r"model.layers.\1.mlp.down_proj.kernel",
    r"model\.norm\.weight": "model.norm.scale",
    r"lm_head\.weight": "model.lm_head",
}


def _map_key(hf_key: str) -> str | None:
    for pattern, repl in _HF_TO_NNX.items():
        if re.match(pattern, hf_key):
            return re.sub(pattern, repl, hf_key)
    return None


def _build_template_from_model(model: JaxQwen3ForCausalLM) -> Dict[FlatKey, jax.Array]:
    state = nnx.filter_state(nnx.state(model), nnx.Param)
    flat_state = nnx.to_flat_state(state)
    return dict(flat_state)


def _resolve_safetensors_path(model_name_or_path: str) -> str:
    if os.path.isfile(model_name_or_path):
        return model_name_or_path
    if os.path.isdir(model_name_or_path):
        return os.path.join(model_name_or_path, "model.safetensors")
    return cached_file(model_name_or_path, "model.safetensors")


def _load_safetensors_state_dict(model_name_or_path: str) -> Dict[str, jax.Array]:
    if os.path.isfile(model_name_or_path) or os.path.isdir(model_name_or_path):
        safetensors_path = _resolve_safetensors_path(model_name_or_path)
        if os.path.exists(safetensors_path):
            return load_file(safetensors_path)
        index_path = os.path.join(model_name_or_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as handle:
                index = json.load(handle)
            return _load_sharded_state_dict(model_name_or_path, index["weight_map"])

    try:
        safetensors_path = _resolve_safetensors_path(model_name_or_path)
        return load_file(safetensors_path)
    except OSError:
        index_path = cached_file(model_name_or_path, "model.safetensors.index.json")
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        return _load_sharded_state_dict(model_name_or_path, index["weight_map"])


def _load_sharded_state_dict(model_name_or_path: str,
                             weight_map: Dict[str, str]) -> Dict[str, jax.Array]:
    state_dict: Dict[str, jax.Array] = {}
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        shard_path = (
            shard if os.path.isfile(shard) else cached_file(model_name_or_path, shard)
        )
        state_dict.update(load_file(shard_path))
    return state_dict


def convert_hf_model_to_pytree(model_name_or_path: str,
                               model: JaxQwen3ForCausalLM,
                               device: jax.Device | None = None,
                               verify: bool = True) -> nnx.State:
    device = jax.devices("cpu")[0]
    config = model.config
    state_dict = _load_safetensors_state_dict(model_name_or_path)

    template = _build_template_from_model(model)
    converted: Dict[FlatKey, jax.Array] = {key: None for key in template}

    for key, value in state_dict.items():
        mapped_key = _map_key(key)
        if mapped_key is None:
            continue
        if config.tie_word_embeddings and mapped_key == "model.lm_head":
            continue
        flat_key = _flat_key_from_str(mapped_key)
        if flat_key not in converted:
            raise ValueError(f"Unexpected key in model state: {mapped_key}")
        with jax.default_device(device):
            converted[flat_key] = _convert_weight(key, value, config)

    missing = [key for key, value in converted.items() if value is None]
    if missing:
        raise ValueError(f"Missing weights for {missing[:5]} (total {len(missing)})")

    pytree = nnx.from_flat_state(converted)
    if verify:
        verify_pytree_matches_model(pytree, model)
    return pytree


def verify_pytree_matches_model(pytree: nnx.State,
                                model: JaxQwen3ForCausalLM) -> None:
    template = _build_template_from_model(model)
    flat_state = nnx.to_flat_state(nnx.filter_state(pytree, nnx.Param))
    for key, value in dict(flat_state).items():
        if key not in template:
            raise ValueError(f"Unexpected key in pytree: {key}")
        if template[key].shape != value.shape:
            raise ValueError(
                f"Shape mismatch for {key}: expected {template[key].shape}, got {value.shape}"
            )


def load_hf_weights_into_model(model: JaxQwen3ForCausalLM,
                               model_name_or_path: str,
                               device: jax.Device | None = None,
                               verify: bool = True) -> None:
    pytree = convert_hf_model_to_pytree(
        model_name_or_path,
        model,
        device=device,
        verify=verify,
    )
    nnx.update(model, pytree)
