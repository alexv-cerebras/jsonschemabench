import os
import sys
from tqdm import tqdm
from json import dumps
from dataclasses import asdict
from typing import List, Optional, Union
from glob import glob

from core.engine import Engine
from core.evaluator import evaluate
from core.types import GenerationOutput
from core.dataset import Dataset, DatasetConfig
from core.utils import disable_print, nanoid, safe_min, print_scores
from core.messages import MessagesFormatter, FEW_SHOTS_MESSAGES_FORMATTER


def bench(
    engine: Engine,
    tasks: List[str],
    limit: Optional[int] = None,
    messages_formatter: Union[
        MessagesFormatter, List[MessagesFormatter]
    ] = FEW_SHOTS_MESSAGES_FORMATTER,
    hf_token: Optional[str] = None,
    close_engine: bool = True,
    save_outputs: bool = False,
) -> List[List[GenerationOutput]]:
    """Benchmarks an engine with specified tasks and datasets.

    :param engine: Engine
        The engine to benchmark.
    :param tasks: List[str]
        The tasks to benchmark.
    :param limit: Optional[int]
        The limit on the number of samples to benchmark.
    :param messages_formatter: Union[MessagesFormatter, List[MessagesFormatter]]
        The function(s) to format the schema into a list of messages. If a single
        function is provided, it will be used for all tasks. If a list of
        functions is provided, each function will be used for the corresponding
        task.
    :param hf_token: Optional[str]
        The Hugging Face token to use for loading datasets.
    :param close_engine: bool
        Whether to close the engine after the benchmark.
    :param save_outputs: bool
        Whether to save the generation outputs after the benchmark.

    :return: List[List[GenerationOutput]]
        The generation outputs for each sample for each task.
    """
    if not isinstance(messages_formatter, list):
        messages_formatter = [messages_formatter] * len(tasks)

    all_outputs = []
    for task, mf in zip(tasks, messages_formatter):
        task_outputs = []
        dataset = Dataset(DatasetConfig(task, limit=limit))
        for messages, schema in tqdm(
            dataset.iter(mf),
            total=safe_min(len(dataset), limit),
            desc=task,
            file=sys.stdout,
        ):
            schema = engine.adapt_schema(schema)
            result = engine.generate(task, messages, schema)
            task_outputs.append(result)
        all_outputs.append(task_outputs)

    if save_outputs:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate two levels up to the "jsonschemabench" directory
        jsonschemabench_path = os.path.abspath(os.path.join(current_dir, ".."))
        outputs_path = os.path.join(jsonschemabench_path, "outputs")
        engine_path = os.path.join(outputs_path, engine.name)
        
        if not os.path.exists(outputs_path):
            os.makedirs(outputs_path)

        if not os.path.exists(engine_path):
            os.makedirs(engine_path)
            
        id: int = len(glob(f"{engine_path}/*.jsonl"))
        with open(f"{engine_path}/{id}.jsonl", "w") as f:
            f.write(
                f"{dumps({'engine': engine.name, 'engine_config': asdict(engine.config)})}\n"
            )

            for outputs in all_outputs:
                for output in outputs:
                    f.write(f"{dumps(asdict(output))}\n")

        print(f"Outputs saved to {engine_path}/{id}.jsonl")

    if close_engine:
        engine.close()

    return all_outputs
