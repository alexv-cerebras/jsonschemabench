from json import loads
from typing import Dict, List
from argparse import ArgumentParser
from dacite import from_dict, Config

from core.evaluator import evaluate
from core.types import GenerationOutput
from core.utils import print_scores, plot_perf_metrics, save_scores
import json
import os


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--outputs", type=str, required=True)
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--output-path", type=str, required=False, default=None)
    parser.add_argument("--failed-schemas-output-path", type=str, required=False, default=None)
    args = parser.parse_args()

    dacite_config = Config(check_types=False)
    with open(args.outputs, "r") as f:
        engine_config = loads(f.readline())
        outputs = [
            from_dict(GenerationOutput, loads(line), config=dacite_config)
            for line in f.readlines()[1:]
        ]

    task_outputs: Dict[str, List[GenerationOutput]] = {}
    for output in outputs:
        if output.task not in task_outputs:
            task_outputs[output.task] = []
        task_outputs[output.task].append(output)

    compliance = []
    perf_metrics = []
    output_tokens = []
    declared_coverage = []
    empirical_coverage = []
    for task_name, outputs in task_outputs.items():
        dc, ec, cl, pm, ot, categories_cov, failed_schemas = evaluate(outputs)
        
        if args.output_path:
            os.makedirs(args.output_path, exist_ok=True)
            path = os.path.join(args.output_path, f"{task_name}_category_coverage.json")
            with open(path, "w") as f:
                json.dump(categories_cov, f)
                
        if args.failed_schemas_output_path:
            os.makedirs(args.failed_schemas_output_path, exist_ok=True)
            for cat in categories_cov.keys():
                path = os.path.join(args.failed_schemas_output_path, cat)
                os.makedirs(path, exist_ok=True)
                
                path = os.path.join(path, f"{task_name}_failed_schemas.json")
                with open(path, "w") as f:
                    json.dump(failed_schemas, f)
                    
        compliance.append(cl)
        perf_metrics.append(pm)
        declared_coverage.append(dc)
        empirical_coverage.append(ec)
        output_tokens.append(ot)

    if args.output_path:
        save_scores(
            declared_coverage,
            empirical_coverage,
            compliance,
            perf_metrics,
            output_tokens,
            list(task_outputs.keys()),
            args.output_path,
        )
    
    print(engine_config)
    print_scores(
        declared_coverage,
        empirical_coverage,
        compliance,
        perf_metrics,
        output_tokens,
        list(task_outputs.keys()),
        args.details,
    )

    if args.details:
        plot_perf_metrics(
            perf_metrics,
            list(task_outputs.keys()),
            f"{args.outputs.split('.')[0]}.png",
            engine_config["engine"],
        )
