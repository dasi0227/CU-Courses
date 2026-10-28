"""Trader and Jev clients with one auditable record per API attempt."""

from abc import ABC, abstractmethod
from datetime import datetime
import json
import math
import os
import time
from typing import Callable, TypedDict
from uuid import uuid4

from openai import APIError, APIResponseValidationError
from typesafe_sdk import (
    RetryPolicy, Score, TypeSafeAPIConnectionError, TypeSafeAPIError,
    TypeSafeAPIResponseValidationError, TypeSafeClient, TypeSafeError,
)

from src.llm.config import load_environment, make_client, resolve_model


JEV_MODEL = "jev-latest"


class Message(TypedDict):
    role: str
    content: str


class ModelCallError(RuntimeError):
    def __init__(self, message: str, round_num: int, agent_id: str):
        super().__init__(message)
        self.round_num = round_num
        self.agent_id = agent_id


def _redact(text: str) -> str:
    for name in ("OPENROUTER_API_KEY", "LLM_API_KEY", "TYPESAFE_API_KEY"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[REDACTED]")
    return text


def _field(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _usage(usage) -> dict:
    def count(*names):
        for name in names:
            value = _field(usage, name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    input_tokens = count("input_tokens", "prompt_tokens")
    output_tokens = count("output_tokens", "completion_tokens")
    total_tokens = count("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def _error_metadata(error: Exception) -> dict:
    body = getattr(error, "body", None)
    raw_text = json.dumps(body, ensure_ascii=False) if isinstance(body, (dict, list)) else body
    return {
        "request_id": getattr(error, "request_id", None),
        "finish_reason": None,
        "raw_text": _redact(raw_text) if isinstance(raw_text, str) else None,
        "usage": _usage(_field(body, "usage")),
    }


def _run_call(*, request: dict, operation: Callable, describe: Callable, decode: Callable,
              logger, max_retries: int, initial_retry_delay: float):
    """Retry explicitly so the log includes errors and their actual token usage."""
    call_id = logger.new_call_id() if logger else uuid4().hex
    for attempt in range(1, max_retries + 2):
        start = time.perf_counter()
        if logger:
            logger.log_call({"event": "request", "time": datetime.now().astimezone().isoformat(),
                             "call_id": call_id, "attempt": attempt, **request})
        error = None
        parse_error = None
        metadata = {"request_id": None, "finish_reason": None, "raw_text": None, "usage": _usage(None)}
        try:
            response = operation()
        except Exception as exc:
            error = exc
            metadata = _error_metadata(exc)
            status = "api_error" if isinstance(exc, (APIError, TypeSafeAPIError, TypeSafeAPIConnectionError)) else "validation_error"
            if isinstance(exc, json.JSONDecodeError):
                status, parse_error = "parse_error", str(exc)
            elif isinstance(exc, APIResponseValidationError):
                status = "validation_error"
            elif isinstance(exc, TypeSafeAPIResponseValidationError):
                status = "validation_error"
                if isinstance(exc.body, str):
                    try:
                        json.loads(exc.body)
                    except json.JSONDecodeError as parse_exc:
                        status = "parse_error"
                        parse_error = str(parse_exc)
        else:
            try:
                metadata = describe(response)
                result = decode(response)
                status = "success"
            except json.JSONDecodeError as exc:
                error, status, parse_error = exc, "parse_error", str(exc)
            except Exception as exc:
                error, status = exc, "validation_error"
        record = {
            "event": "response", "time": datetime.now().astimezone().isoformat(),
            "call_id": call_id, "attempt": attempt, "status": status,
            "latency_ms": round((time.perf_counter() - start) * 1000, 3),
            **metadata, "parse_error": parse_error,
        }
        if error is not None:
            record["error"] = {"type": type(error).__name__, "message": _redact(str(error))}
        if logger:
            logger.log_call(record)
        if error is None:
            return result
        if attempt <= max_retries:
            time.sleep(initial_retry_delay * 2 ** (attempt - 1))
        else:
            raise ModelCallError(
                f"{type(error).__name__}: {_redact(str(error))}", request["round"], request["agent_id"],
            ) from error


def extract_json(response: str) -> dict:
    """Read a JSON object, allowing a model's reasoning prefix or Markdown fence."""
    if not isinstance(response, str):
        raise ValueError("Model response content must be a string")
    answer = response.rsplit("</think>", 1)[-1].strip()
    if answer.startswith("```"):
        lines = answer.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            answer = "\n".join(lines[1:-1]).strip()
    value = json.loads(answer)
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


class ModelWrapper(ABC):
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 8192,
                 top_p: float = 1.0, max_retries: int = 3, initial_retry_delay: float = 1.0, **kwargs):
        self.model_name = model_name
        self.parameters = {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p, **kwargs}
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

    @abstractmethod
    def _request(self, messages: list[Message]):
        """Send exactly one API request, with SDK retries disabled."""

    def generate_json(self, messages: list[Message], *, round_num: int, agent_id: str, logger=None,
                      validator: Callable | None = None) -> dict:
        def describe(response):
            choice = response.choices[0] if response.choices else None
            return {
                "request_id": getattr(response, "_request_id", None) or getattr(response, "id", None),
                "finish_reason": _field(choice, "finish_reason"),
                "raw_text": _field(_field(choice, "message"), "content"),
                "usage": _usage(response.usage),
            }

        def decode(response):
            data = extract_json(response.choices[0].message.content)
            if validator:
                validator(data)
            return data

        return _run_call(
            request={"round": round_num, "agent_id": agent_id, "model": self.model_name,
                     "parameters": dict(self.parameters), "messages": messages},
            operation=lambda: self._request(messages), describe=describe, decode=decode, logger=logger,
            max_retries=self.max_retries, initial_retry_delay=self.initial_retry_delay,
        )

    def close(self) -> None:
        pass


class OpenRouterClient(ModelWrapper):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(resolve_model(model_name), **kwargs)
        if self.model_name == "deepseek/deepseek-v4-flash":
            extra_body = dict(self.parameters.get("extra_body") or {})
            extra_body["provider"] = {
                **extra_body.get("provider", {}),
                "only": ["baidu/fp8"], "allow_fallbacks": False,
            }
            self.parameters["extra_body"] = extra_body
        self.client = make_client(timeout=120, max_retries=0)

    def _request(self, messages: list[Message]):
        return self.client.chat.completions.create(model=self.model_name, messages=messages, **self.parameters)

    def close(self) -> None:
        self.client.close()


class JevClient:
    def __init__(self, max_retries: int = 3, initial_retry_delay: float = 1.0):
        load_environment()
        self.model_name = JEV_MODEL
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.client = TypeSafeClient(model=JEV_MODEL, retry=RetryPolicy(max_retries=0, timeout=None), timeout=120)

    def score(self, state: dict, question: Score, *, round_num: int, logger=None) -> float:
        def describe(response):
            try:
                request_id = response.request_id
            except TypeSafeError:
                request_id = None
            return {
                "request_id": request_id,
                "finish_reason": None,
                "raw_text": response.model_dump_json(),
                "usage": _usage(response.usage),
            }

        def decode(response):
            score = response.scores["coordination"].score
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 3:
                raise ValueError("Jev coordination score must be a finite number between 0 and 3")
            return score + 1

        return _run_call(
            request={"round": round_num, "agent_id": "Overseer", "model": self.model_name, "parameters": {},
                     "state": state, "questions": {"coordination": question.model_dump(mode="json")}},
            operation=lambda: self.client.system_one(state=state, questions={"coordination": question}, model=self.model_name),
            describe=describe, decode=decode, logger=logger,
            max_retries=self.max_retries, initial_retry_delay=self.initial_retry_delay,
        )

    def close(self) -> None:
        self.client.close()


def get_client(model: str, temperature: float, **kwargs) -> ModelWrapper:
    kwargs.setdefault("response_format", {"type": "json_object"})
    return OpenRouterClient(model, temperature=temperature, **kwargs)


def get_overseer_client() -> JevClient:
    return JevClient()
