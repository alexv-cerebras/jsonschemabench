import os
from functools import partial
from core.bench import bench
from argparse import ArgumentParser
from core.dataset import DATASET_NAMES
from core.utils import load_config, disable_print
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
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=False, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--limit", type=int, required=False)
    parser.add_argument("--num_shots", type=int, required=False)
    parser.add_argument("--save_outputs", action="store_true")
    args = parser.parse_args()

    tasks = args.tasks
    if not all(task in DATASET_NAMES for task in tasks):
        raise ValueError(f"Invalid task names: {tasks}, available: {DATASET_NAMES}")
    
    if not args.hf_token:
        raise ValueError("Hugging Face token is required. Set it with --hf_token or HF_TOKEN environment variable.")

    if args.config is None:
        args.config = os.path.join("tests/configs", f"{args.engine}.yaml")

    engine_config = load_config(ENGINE_TO_CONFIG[args.engine], args.config)        
    engine = ENGINE_TO_CLASS[args.engine](engine_config)

    messages_formatter = partial(FEW_SHOTS_MESSAGES_FORMATTER, num_shots=args.num_shots)

    bench(
        engine=engine,
        tasks=tasks,
        limit=args.limit,
        messages_formatter=messages_formatter,
        save_outputs=args.save_outputs,
        hf_token=args.hf_token,
        close_engine=True,
    )
