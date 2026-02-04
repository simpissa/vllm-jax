import jax

from vllm.config import Config
from vllm.src.engine import Engine


def main() -> None:
    config = Config(
        model="Qwen/Qwen3-0.6B",
        max_num_batched_tokens=128,
        max_num_seqs=4,
        num_blocks=128,
        block_size=16,
    )
    engine = Engine(config)
    print(engine.generate(
        ["Hello from vLLM!"],
        max_new_tokens=16,
        temperature=0.0,
        seed=0,
    ))


if __name__ == "__main__":
    main()
