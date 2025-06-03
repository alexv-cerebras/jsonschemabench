from time import time
from dataclasses import dataclass
import logging
import stopit
from functools import cached_property
import requests
from typing import Dict, Any, List, Optional
import json

from core.registry import register_engine
from core.engine import Engine, EngineConfig
from core.evaluator import is_json_schema_valid
from core.types import (
    CompileStatus,
    DecodingStatus,
    GenerationOutput,
    CompileStatusCode,
    DecodingStatusCode,
)


class LengthExceedError(Exception):
    pass


def _send_request(url, method, params=None, headers=None, req_timeout=5):
    try:
        if method.lower() == "post":
            # Socket‐level timeout is now 5 s for connect AND read
            response = requests.post(
                url,
                json=params,
                headers=headers,
                timeout=req_timeout
            )
        else:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=req_timeout
            )
            
    except requests.exceptions.Timeout as te:
        # This triggers if either the connect or read phase took longer than req_timeout
        logging.error("HTTP request timed out: %r", te)
        raise
    
    if response.status_code != 200:
        logging.error(
            "Request failed (status %s): %r",
            response.status_code,
            response.text
        )
        
        code = json.loads(response.text)["code"]
        if code == "context_length_exceeded":
            raise LengthExceedError(
                "The request exceeded the maximum context length."
            )
        raise Exception(f"HTTP {response.status_code} → {response.text}")

    return response.json()


@dataclass
class CerebrasConfig(EngineConfig):
    model: str
    base_url: str
    max_tokens: int
    temperature: Optional[float] = None
    run_timeout: Optional[int] = None


class CerebrasEngine(Engine[CerebrasConfig]):
    name = "cerebras"

    def __init__(
        self,
        config: CerebrasConfig
    ):
        super().__init__(config)
        self.url = (
            f"{self.config.base_url}/chat/completions" if self.config.base_url.endswith("/v1")
            else f"{self.config.base_url}/v1/chat/completions"
        )
        self.run_timeout = self.config.run_timeout or 5
    
    @property
    def headers(self):
        headers = {
            "Content-Type": "application/json"
        }
        return headers
    
    def _get_payload(self, output: GenerationOutput):
        payload = {
            "model": self.config.model,
            "messages": output.messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": output.schema,
                    "name": "json_schema"
                }
            },
            "include_internal": True,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        return payload

    def _generate(self, output: GenerationOutput) -> None:
        try:
            response = _send_request(
                self.url, 
                "post", 
                params=self._get_payload(output), 
                headers=self.headers,
                req_timeout=self.run_timeout
            )
        except requests.exceptions.Timeout as e:
            output.metadata.compile_status = CompileStatus(
                code=CompileStatusCode.RUNTIME_TIMEOUT, message=str(e)
            )
            logging.error(f"Error during generation: {e}")
            return
        except LengthExceedError as e:
            output.metadata.compile_status = CompileStatus(
                code=CompileStatusCode.PROMPT_TOO_LONG, message=str(e)
            )
            logging.error(f"Error during generation: {e}")
            return
        except Exception as e:
            output.metadata.compile_status = CompileStatus(
                code=CompileStatusCode.UNKOWN_ERROR, message=str(e)
            )
            logging.error(f"Error during generation: {e}")
            return
        
        # Time to first token in s
        output.perf_metrics.ttft = response["time_info"]["prompt_time"]
        # Time per output token in ms
        output.perf_metrics.tpot = (response["time_info"]["completion_time"]/response["usage"]["completion_tokens"]) * 1000
        # Total generation time in s
        output.perf_metrics.tgt = response["time_info"]["completion_time"]
        # Grammar compilation time in s
        output.perf_metrics.gct = response["usage"]["constraint_manager_compile_time"]
        # Prefilling time in s
        output.perf_metrics.prft = response["usage"]["grammar_overhead_time"]

        output.token_usage.output_tokens = response["usage"]["completion_tokens"]
        output.token_usage.input_tokens = response["usage"]["prompt_tokens"]
        output.metadata.compile_status = CompileStatus(code=CompileStatusCode.OK)
        output.metadata.decoding_status = DecodingStatus(code=DecodingStatusCode.OK)

        output.generation = response["choices"][0]["message"]["content"]
        return

    def adapt_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        recursively_set_additional_properties_false(schema)
        add_root_type_if_missing(schema)
        # schema = set_all_properties_required(schema)
        if not is_json_schema_valid(schema):
            print("The JSON schema after adaptation is no longer valid.")
        return schema

    @cached_property
    def max_context_length(self):
        return self.config.max_tokens

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


register_engine(CerebrasEngine, CerebrasConfig)
