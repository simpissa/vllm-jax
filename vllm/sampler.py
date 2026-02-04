import jax
import jax.numpy as jnp

from vllm.sampling_params import SamplingParams


class Sampler:

    def sample(self, logits: jax.Array, sampling_params: SamplingParams,
               rng_key: jax.Array) -> tuple[jax.Array, jax.Array]:
        if logits.ndim == 1:
            logits = logits[None, :]
        if sampling_params.temperature <= 0:
            token_ids = jnp.argmax(logits, axis=-1)
            return token_ids, rng_key

        logits = logits / sampling_params.temperature
        if sampling_params.top_k > 0:
            top_k = min(sampling_params.top_k, logits.shape[-1])
            top_values, top_indices = jax.lax.top_k(logits, top_k)
            masked_logits = jnp.full_like(logits, jnp.finfo(logits.dtype).min)
            batch_idx = jnp.arange(logits.shape[0])[:, None]
            masked_logits = masked_logits.at[batch_idx, top_indices].set(top_values)
            logits = masked_logits

        if sampling_params.top_p < 1.0:
            sorted_indices = jnp.argsort(-logits, axis=-1)
            sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
            sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
            cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
            keep_mask = cumulative_probs <= sampling_params.top_p
            keep_mask = keep_mask.at[:, 0].set(True)
            sorted_logits = jnp.where(keep_mask, sorted_logits,
                                      jnp.finfo(sorted_logits.dtype).min)
            masked_logits = jnp.full_like(logits, jnp.finfo(logits.dtype).min)
            batch_idx = jnp.arange(logits.shape[0])[:, None]
            masked_logits = masked_logits.at[batch_idx, sorted_indices].set(sorted_logits)
            logits = masked_logits

        rng_key, sample_key = jax.random.split(rng_key)
        token_ids = jax.random.categorical(sample_key, logits, axis=-1)
        return token_ids, rng_key
