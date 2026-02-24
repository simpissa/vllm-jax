import random
import time
from transformers import AutoTokenizer


MODEL = "Qwen/Qwen3-0.6B"
NUM_REQUESTS = 8
PROMPT_WORDS = 200
MAX_NEW_TOKENS = 32
SEED = 42


def build_prompts(tokenizer) -> list[str]:
    rng = random.Random(SEED)
    vocab_size = len(tokenizer)
    prompts = []
    for _ in range(NUM_REQUESTS):
        token_ids = [rng.randrange(vocab_size) for _ in range(PROMPT_WORDS)]
        prompt = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        prompts.append(prompt)
    return prompts


def run_local(prompts):
    from vllm.config import Config
    from vllm.model import qwen3
    from vllm.src.engine import Engine

    qwen3.USE_DECODE_PAGED_ATTENTION_KERNEL = True
    qwen3.USE_PREFILL_PAGED_ATTENTION_KERNEL = False

    config = Config(
        model=MODEL,
        max_num_batched_tokens=2048,
        max_num_seqs=min(NUM_REQUESTS, 8),
        num_blocks=160,
        block_size=16,
    )
    engine = Engine(config)

    warmup_count = min(NUM_REQUESTS, config.max_num_seqs)
    _ = engine.generate(
        prompts[:warmup_count],
        max_new_tokens=1,
        temperature=0.0,
        ignore_eos=True,
        seed=SEED,
    )

    start = time.perf_counter()
    _ = engine.generate(
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        ignore_eos=True,
        seed=SEED,
    )
    elapsed = time.perf_counter() - start

    generated_tokens = NUM_REQUESTS * MAX_NEW_TOKENS
    return generated_tokens / elapsed


def run_hf(prompts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    model.eval()

    warmup_inputs = tokenizer(prompts[:1], return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        _ = model.generate(
            **warmup_inputs,
            max_new_tokens=8,
            min_new_tokens=8,
            do_sample=False,
        )

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_lens = inputs["attention_mask"].sum(dim=1)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated_tokens = NUM_REQUESTS * MAX_NEW_TOKENS
    return generated_tokens / elapsed


tokenizer = AutoTokenizer.from_pretrained(MODEL)
prompts = build_prompts(tokenizer)

print("running hf")
hf_toks = run_hf(prompts) # requires torch
print("running vllm")
local_toks = run_local(prompts)

print(f"local_vllm_jax: {local_toks:.2f} tok/s")
print(f"huggingface: {hf_toks:.2f} tok/s")