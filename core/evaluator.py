import numpy as np
from uuid import UUID
from json import loads
from typing import List, Tuple
from ipaddress import IPv4Address, IPv6Address
from jsonschema import Draft202012Validator, FormatChecker, SchemaError

from core.utils import bootstrap
from core.types import (
    Schema,
    CompileStatusCode,
    GenerationOutput,
    AggregatedPerfMetrics,
    Metric,
)


categories = [
    "additionalProperties",
    "allOf",
    "anchor",
    "anyOf",
    "boolean_schema",
    "const",
    "contains",
    "content",
    "default",
    "defs",
    "dependentRequired",
    "dependentSchemas",
    "dynamicRef",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "if-then-else",
    "infinite-loop-detection",
    "items",
    "maxContains",
    "maxItems",
    "maxLength",
    "maxProperties",
    "maximum",
    "minContains",
    "minItems",
    "minLength",
    "minProperties",
    "minimum",
    "multipleOf",
    "not",
    "oneOf",
    "pattern",
    "patternProperties",
    "prefixItems",
    "properties",
    "propertyNames",
    "ref",
    "required",
    "type",
    "unevaluatedItems",
    "unevaluatedProperties",
    "uniqueItems",
]


def is_json_schema_valid(schema: Schema):
    try:
        Draft202012Validator.check_schema(schema)
        return True
    except SchemaError:
        return False


format_checker = FormatChecker()


@format_checker.checks("ipv4")
def ipv4_check(value):
    IPv4Address(value)


@format_checker.checks("ipv6")
def ipv6_check(value):
    IPv6Address(value)


@format_checker.checks("uuid")
def uuid_check(value):
    UUID(value)


def validate_json_schema(instance: Schema, schema: Schema) -> bool:
    if not is_json_schema_valid(schema):
        return False
    validator = Draft202012Validator(schema, format_checker=format_checker)
    try:
        validator.validate(instance)

    # we catch all exceptions include ValidationError and Error from extension validators
    except Exception:
        return False
    return True


def evaluate(
    outputs: List[GenerationOutput],
) -> Tuple[Metric, Metric, Metric, AggregatedPerfMetrics, Metric]:
    output_tokens_list = []
    declared_coverage_list = []
    empirical_coverage_list = []
    categories_coverage_list, categories_coverage_dict = [], {}
    failed_schemas = {}

    for generation_output in outputs:
        generation = generation_output.generation
        schema = generation_output.schema 
        cur_categories = [category for category in categories if category in generation]
        categories_coverage_list.append(cur_categories)

        if schema is None or generation is None:
            continue

        if generation_output.metadata.compile_status.code == CompileStatusCode.OK:
            declared_coverage_list.append(1)
        else:
            declared_coverage_list.append(0)

        try:
            json_object = loads(generation)
        except Exception:
            empirical_coverage_list.append(0)
            continue

        if not validate_json_schema(json_object, schema):
            empirical_coverage_list.append(0)
            continue

        empirical_coverage_list.append(1)
        output_tokens_list.append(generation_output.token_usage.output_tokens)
        
    for i, label in enumerate(empirical_coverage_list):
        for cat in categories_coverage_list[i]:
            categories_coverage_dict[cat] = categories_coverage_dict.get(cat, []) + [label]
            if not label:
                failed_schemas[cat] = failed_schemas.get(cat, []) + [outputs[i].schema]
            
    for cat, values in categories_coverage_dict.items():
        categories_coverage_dict[cat] = sum(values) / len(values) if values else 0
        
    ttft_list = [
        generation_output.perf_metrics.ttft
        for generation_output in outputs
        if generation_output.perf_metrics.ttft is not None
    ]
    tpot_list = [
        generation_output.perf_metrics.tpot
        for generation_output in outputs
        if generation_output.perf_metrics.tpot is not None
    ]
    tgt_list = [
        generation_output.perf_metrics.tgt
        for generation_output in outputs
        if generation_output.perf_metrics.tgt is not None
    ]
    gct_list = [
        generation_output.perf_metrics.gct
        for generation_output in outputs
        if generation_output.perf_metrics.gct is not None
    ]

    compliance_list = [
        ec for ec, dc in zip(empirical_coverage_list, declared_coverage_list) if dc == 1
    ]

    dc_mean_list = bootstrap(declared_coverage_list, np.mean)
    ec_mean_list = bootstrap(empirical_coverage_list, np.mean)
    c_mean_list = bootstrap(compliance_list, np.mean)

    return (
        Metric.from_values(dc_mean_list),
        Metric.from_values(ec_mean_list),
        Metric.from_values(c_mean_list),
        AggregatedPerfMetrics(
            ttft=Metric.from_values(ttft_list),
            tpot=Metric.from_values(tpot_list),
            tgt=Metric.from_values(tgt_list),
            gct=Metric.from_values(gct_list),
        ),
        Metric.from_values(output_tokens_list),
        categories_coverage_dict,
        failed_schemas
    )
