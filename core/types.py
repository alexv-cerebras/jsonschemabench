import numpy as np
from enum import Enum
from uuid import uuid4
from dataclasses import dataclass, field, fields
from typing import List, Dict, Any, Optional

from core.messages import Message
from core.utils import safe_divide, safe_subtract, safe_reduce


Schema = Dict[str, Any]


class CompileStatusCode(str, Enum):
    TBD = "tbd" #-1
    OK = "ok" #0
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    RUNTIME_GRAMMAR_ERROR = "runtime_grammar_error"
    API_BAD_RESPONSE = "api_bad_response"
    PROMPT_TOO_LONG = "prompt_too_long"
    COMPILE_TIMEOUT = "compile_timeout"
    RUNTIME_TIMEOUT = "runtime_timeout"
    UNKOWN_ERROR = "unknown_error"


class DecodingStatusCode(str, Enum):
    TBD = "tbd"
    OK = "ok"
    EXCEEDING_MAX_CTX = "exceeding_max_ctx"
    DECODING_TIMEOUT = "decoding_timeout"
    BAD_API_RESPONSE = "bad_api_response"
    UNKOWN_ERROR = "unknown_error"


@dataclass
class CompileStatus:
    code: CompileStatusCode = CompileStatusCode.TBD
    message: Optional[str] = None


@dataclass
class DecodingStatus:
    code: DecodingStatusCode = DecodingStatusCode.TBD
    message: Optional[str] = None


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def __str__(self) -> str:
        return (
            f"token usage: {self.input_tokens:,} input, {self.output_tokens:,} output."
        )


@dataclass
class Token:
    id: Optional[int] = None
    text: Optional[str] = None
    logprob: Optional[float] = None


@dataclass
class GenerationMetadata:
    first_token_arrival_time: Optional[float] = None
    grammar_compilation_end_time: Optional[float] = None
    compile_status: Optional[CompileStatus] = field(default_factory=CompileStatus)
    decoding_status: Optional[DecodingStatus] = field(default_factory=DecodingStatus)


@dataclass
class PerfMetrics:
    """Performance metrics for generation processes."""

    # Time to first token in s
    ttft: Optional[float] = None
    # Time per output token in ms
    tpot: Optional[float] = None
    # Total generation time in s
    tgt: Optional[float] = None
    # Grammar compilation time in s
    gct: Optional[float] = None
    # Prefilling time in s
    prft: Optional[float] = None
    # Peak memory in MB
    peak_memory: Optional[float] = None

    @classmethod
    def from_timestamps(
        cls,
        start_time: float,
        grammar_compilation_end_time: Optional[float],
        first_token_arrival_time: Optional[float],
        end_time: float,
        num_output_tokens: int,
    ):
        ttft = safe_subtract(first_token_arrival_time, start_time)
        tpot = (
            safe_divide(
                safe_subtract(end_time, first_token_arrival_time),
                safe_subtract(num_output_tokens, 1),
            )
            if num_output_tokens > 0
            else None
        )
        tgt = safe_subtract(end_time, start_time)
        gct = safe_subtract(grammar_compilation_end_time, start_time)
        # prft = safe_subtract(first_token_arrival_time, grammar_compilation_end_time)
        return cls(
            time_to_first_token=ttft,
            time_per_output_token=tpot * 1000 if tpot is not None else None,
            total_generation_time=tgt,
            grammar_compilation_time=gct,
            # prft=prft,
        )


@dataclass
class Metric:
    values: List[float] = field(default_factory=list)
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None

    @classmethod
    def from_values(cls, values: List[float]) -> "Metric":
        return cls(
            values=values,
            min=safe_reduce(values, min),
            max=safe_reduce(values, max),
            median=safe_reduce(values, np.median),
            std=safe_reduce(values, np.std),
        )


@dataclass
class AggregatedPerfMetrics:
    time_to_first_token: Metric = field(default_factory=Metric)
    time_per_output_token: Metric = field(default_factory=Metric)
    total_generation_time: Metric = field(default_factory=Metric)
    grammar_compilation_time: Metric = field(default_factory=Metric)
    grammar_overhead_time: Metric = field(default_factory=Metric)
    # prft: Metric = field(default_factory=Metric)


@dataclass
class GenerationOutput:
    """Output of a generation run."""

    task: str
    messages: List[Message]
    generation: str
    schema: Schema
    id: str = field(default_factory=lambda: str(uuid4()))
    generated_tokens: List[Token] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    perf_metrics: PerfMetrics = field(default_factory=PerfMetrics)
    metadata: GenerationMetadata = field(default_factory=GenerationMetadata)
