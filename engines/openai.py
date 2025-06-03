import os
from time import time
from dataclasses import dataclass
import logging
import stopit
from typing import Dict, Any, List, Optional

from core.registry import register_engine
from core.engine import Engine, EngineConfig
from core.evaluator import is_json_schema_valid
from core.types import (
    Token,
    CompileStatus,
    DecodingStatus,
    GenerationOutput,
    CompileStatusCode,
    DecodingStatusCode,
)


@dataclass
class OpenAIConfig(EngineConfig):
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    base_url: Optional[str] = None


class OpenAIEngine(Engine[OpenAIConfig]):
    name = "openai"

    def __init__(
        self,
        config: OpenAIConfig,
        api_key_variable_name: Optional[str] = "OPENAI_API_KEY",
    ):
        super().__init__(config)

        from openai import OpenAI
        # from cerebras.cloud.sdk import Cerebras

        # TODO: clients don't support include_internal,
        # so I need to use requests ;(
        self.client = OpenAI(
            api_key=os.getenv(api_key_variable_name),
            base_url=config.base_url,
        )
        logging.info(self.client)
        self.tokenizer = None

    def _generate(self, output: GenerationOutput) -> None:
        def collect_streamed_tokens():
            tokens_str: List[str] = []
            first_token_arrival_time = None

            with stopit.ThreadingTimeout(10) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    stream = self.client.chat.completions.create(
                        model=self.config.model,
                        messages=output.messages,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"schema": output.schema, "name": "json_schema"},
                        },
                        stream=True,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        # include_internal=True,
                        stream_options={"include_usage": True},
                    )
                    
            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                logging.error("Grammar compilation timed out")
                output.metadata.compile_status = CompileStatus(
                    code=CompileStatusCode.COMPILE_TIMEOUT,
                    message="Grammar compilation timed out",
                )
                return

            with stopit.ThreadingTimeout(10) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    for i, chunk in enumerate(stream):
                        if i == 0:
                            first_token_arrival_time = time()

                        if len(chunk.choices) == 0 or chunk.choices[0].finish_reason is not None:
                            continue

                        content = chunk.choices[0].delta.content
                        if not content:
                            continue

                        tokens_str.append(content)

                    return tokens_str, chunk, first_token_arrival_time
                
            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                logging.error("Grammar runtime timed out")
                output.metadata.compile_status = CompileStatus(
                    code=CompileStatusCode.RUNTIME_TIMEOUT,
                    message="Grammar runtime timed out",
                )
                return

        try:
            res = collect_streamed_tokens()
            if not res:
                return
            tokens_str, chunk, first_token_arrival_time = res
            # tokens_str, chunk, first_token_arrival_time = run_with_timeout(
            #     collect_streamed_tokens, timeout=5
            # )
        # except TimeoutError as e:
        #     output.metadata.compile_status = CompileStatus(
        #         code=CompileStatusCode.RUNTIME_TIMEOUT, message=str(e)
        #     )
        #     logging.error(f"Timeout during generation: {e}")
        #     return
        except Exception as e:
            output.metadata.compile_status = CompileStatus(
                code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=str(e)
            )
            logging.error(f"Unexpected error during generation: {e}")
            return

        # print(chunk)
        output.token_usage.output_tokens = chunk.usage.completion_tokens
        output.metadata.first_token_arrival_time = first_token_arrival_time
        output.metadata.compile_status = CompileStatus(code=CompileStatusCode.OK)
        output.metadata.decoding_status = DecodingStatus(code=DecodingStatusCode.OK)

        output.generation = "".join(tokens_str)
        return

    def adapt_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        recursively_set_additional_properties_false(schema)
        add_root_type_if_missing(schema)
        # schema = set_all_properties_required(schema)
        if not is_json_schema_valid(schema):
            print("The JSON schema after adaptation is no longer valid.")
        return schema

    @property
    def max_context_length(self):
        max_context_length_dict = {
            "gpt-4o": 128 * 1000,
            "gpt-4o-mini": 128 * 1000,
            "llama3.1-8b": 128 * 1000,
        }
        return max_context_length_dict[self.config.model]


def add_root_type_if_missing(schema: dict):
    if "type" not in schema:
        schema["type"] = "object"


def recursively_set_additional_properties_false(schema: dict):
    if not isinstance(schema, dict):
        return
    if (
        "additionalProperties" not in schema or schema["additionalProperties"]
    ) and schema.get("properties"):
        schema["additionalProperties"] = False
    if "properties" in schema:
        for prop in schema["properties"]:
            recursively_set_additional_properties_false(schema["properties"][prop])
    if "items" in schema:
        recursively_set_additional_properties_false(schema["items"])


def set_all_properties_required(schema: object) -> object:
    if not isinstance(schema, dict):
        return schema
    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
    for value in schema.values():
        if isinstance(value, dict):
            set_all_properties_required(value)
        elif isinstance(value, list):
            for item in value:
                set_all_properties_required(item)
    return schema


register_engine(OpenAIEngine, OpenAIConfig)
