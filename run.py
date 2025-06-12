import os
from functools import partial
from core.bench import bench
from argparse import ArgumentParser
from core.dataset import DATASET_NAMES
from core.utils import load_config
from core.registry import ENGINE_TO_CLASS, ENGINE_TO_CONFIG
from core.messages import FEW_SHOTS_MESSAGES_FORMATTER


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--engine", type=str, required=True, choices=ENGINE_TO_CLASS.keys()
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--tasks", type=str, required=True, choices=DATASET_NAMES, nargs="+"
    )
    parser.add_argument(
        "--split", type=str, choices=['train', 'val', 'test'], default='test',
    )
    parser.add_argument("--model", type=str)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--base-url", type=str, default=os.environ.get("BASE_URL"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--run-timeout", type=int, default=10)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--limit", type=int, required=False)
    parser.add_argument("--num-shots", type=int, required=False)
    parser.add_argument("--save-outputs", action="store_true")
    args = parser.parse_args()

    tasks = args.tasks
    if not all(task in DATASET_NAMES for task in tasks):
        raise ValueError(f"Invalid task names: {tasks}, available: {DATASET_NAMES}")

    if args.config is None:
        args.config = os.path.join("tests/configs", f"{args.engine}.yaml")

    config_kwargs = {
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "run_timeout": args.run_timeout,
        "max_tokens": args.max_tokens,
        "strict": args.strict,
    }
    engine_config = load_config(ENGINE_TO_CONFIG[args.engine], args.config, config_kwargs)        
    engine = ENGINE_TO_CLASS[args.engine](engine_config)

    messages_formatter = partial(FEW_SHOTS_MESSAGES_FORMATTER, num_shots=args.num_shots)

    bench(
        engine=engine,
        tasks=tasks,
        limit=args.limit,
        messages_formatter=messages_formatter,
        save_outputs=args.save_outputs,
        close_engine=True,
        split=args.split,
    )
