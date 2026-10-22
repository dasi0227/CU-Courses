"""Opt-in Jev decisions for Formula memory experiments.

The HTTP contract follows https://docs.typesafe.ai/api/system-one and
https://docs.typesafe.ai/primitives/noul. No request is made on import or
construction, and even an injected transport requires ``enabled=True``.
The HTTP schema's ``questions.{key}.criteria`` is a NoulCriteria object:
``{"true": "yes description", "false": "no description"}``, not a list.
Pinned IDs are documented at https://docs.typesafe.ai/models.
Jev judges the semantic relevance of existing playbook entries; it never replaces
the deterministic numerical scorer or generates playbook content.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class JevError(RuntimeError):
    """A Jev decision could not be obtained safely."""


class JevDisabledError(JevError):
    """Live decision requests have not been explicitly enabled."""


class JevResponseError(JevError):
    """The API response does not satisfy the requested Noul contract."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a bearer credential to an unexpected redirect target.
        return None


def _http_transport(request: Request, timeout: float) -> dict[str, Any]:
    with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        payload = response.read()
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise JevResponseError("Jev returned invalid JSON") from exc


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be a number between 0 and 1")
    return result


class JevClient:
    """Small, bounded stdlib client for named Noul questions.

    ``base_url`` is the versioned API base (normally ``.../v1``), or a full
    ``.../systemone`` endpoint. ``transport(request, timeout)`` can be injected
    for offline tests and must return the decoded official response object.
    ``max_attempts`` includes the initial request and is limited to two.
    Audit events deliberately omit authorization headers and dollar estimates.
    """

    def __init__(
        self,
        model: str = "jev-1.13.0",
        base_url: str = "https://api.typesafe.ai/v1",
        api_key_env: str = "TYPESAFE_API_KEY",
        enabled: bool = False,
        audit_path: str | Path | None = None,
        timeout: float = 30.0,
        *,
        transport: Callable[[Request, float], dict[str, Any]] | None = None,
        max_attempts: int = 2,
        retry_backoff: float = 0.25,
    ):
        if not re.fullmatch(r"jev-\d+\.\d+\.\d+", model):
            raise ValueError("Use a pinned Jev model ID, for example jev-1.13.0")
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/systemone"):
            endpoint += "/systemone"
        parsed = urlsplit(endpoint)
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Jev base_url must be an HTTPS URL without credentials or query")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be an explicit boolean")
        if not isinstance(api_key_env, str) or not api_key_env:
            raise ValueError("api_key_env must name an environment variable")
        if (isinstance(max_attempts, bool) or not isinstance(max_attempts, int)
                or max_attempts not in (1, 2)):
            raise ValueError("max_attempts must be 1 or 2, including the initial request")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not math.isfinite(retry_backoff) or not 0 <= retry_backoff <= 2:
            raise ValueError("retry_backoff must be between 0 and 2 seconds")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.enabled = enabled
        self.audit_path = Path(audit_path) if audit_path is not None else None
        self.timeout = float(timeout)
        self.max_attempts = max_attempts
        self.retry_backoff = float(retry_backoff)
        self._transport = transport or _http_transport
        self._audit_lock = threading.Lock()
        self.events: list[dict[str, Any]] = []
        self.last_call_info: dict[str, Any] = {}

    def record_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Store a decision/policy audit event, with any current key redacted."""
        secret = os.environ.get(self.api_key_env, "")

        def redact(value):
            if isinstance(value, str):
                return value.replace(secret, "[REDACTED]") if secret else value
            if isinstance(value, dict):
                return {redact(key): redact(item) for key, item in value.items()}
            if isinstance(value, list):
                return [redact(item) for item in value]
            return value

        # Redact before JSON escaping, so unusual characters in a credential
        # cannot evade the check. Round-trip also snapshots mutable state.
        serialized = json.dumps(redact(event), ensure_ascii=False, allow_nan=False)
        safe_event = json.loads(serialized)
        with self._audit_lock:
            if self.audit_path is not None:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a", encoding="utf-8") as stream:
                    stream.write(serialized + "\n")
            self.events.append(safe_event)
        return safe_event

    def _scores(self, response: Any, questions: dict[str, Any]) -> dict[str, float]:
        if not isinstance(response, dict):
            raise JevResponseError("Jev response must be an object")
        if response.get("model") != self.model:
            raise JevResponseError("Jev response model does not match the pinned model")
        answers = response.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise JevResponseError("Jev response must answer exactly the requested questions")
        scores = {}
        for name, answer in answers.items():
            if not isinstance(answer, dict) or answer.get("type") != "noul":
                raise JevResponseError(f"Jev answer {name!r} must have type noul")
            try:
                scores[name] = _probability(answer.get("noul"), f"Jev answer {name!r}")
            except ValueError as exc:
                raise JevResponseError(str(exc)) from exc
        if "usage" in response and not isinstance(response["usage"], dict):
            raise JevResponseError("Jev usage must be an object when present")
        return scores

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, HTTPError):
            return exc.code == 429 or 500 <= exc.code <= 599
        return isinstance(exc, (URLError, TimeoutError, ConnectionError))

    def decide(
        self,
        state: dict[str, Any],
        questions: dict[str, Any],
        *,
        call_id: str,
        stage: str,
    ) -> dict[str, float]:
        """Return one validated yes-probability per named Noul question.

        Disabled, failed, missing, non-finite, and out-of-range decisions raise;
        callers must mark the run failed rather than silently allow/drop items.
        """
        if not isinstance(state, dict) or not isinstance(questions, dict) or not questions:
            raise ValueError("state and nonempty questions must be dictionaries")
        for name, question in questions.items():
            if (not isinstance(name, str) or not isinstance(question, dict)
                    or question.get("type") != "noul" or not question.get("instructions")):
                raise ValueError("Each question must be a named Noul with instructions")
        if not isinstance(call_id, str) or not isinstance(stage, str) or not call_id or not stage:
            raise ValueError("call_id and stage must be nonempty strings")
        payload = {"model": self.model, "state": state, "questions": questions}
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        common = {
            "event": "jev_request", "model": self.model,
            "requested_model": self.model, "call_id": call_id, "stage": stage,
            "state": state, "questions": questions, "enabled": self.enabled,
        }
        self.last_call_info = {}
        if not self.enabled:
            self.record_event({**common, "status": "disabled", "attempt": 0})
            raise JevDisabledError("Jev requests are disabled; explicitly enable live execution")
        secret = os.environ.get(self.api_key_env, "")
        if not secret:
            self.record_event({**common, "status": "failed", "attempt": 0,
                               "error_type": "MissingAPIKey"})
            raise JevError(f"Missing API key environment variable {self.api_key_env}")
        started = time.perf_counter()
        for attempt in range(1, self.max_attempts + 1):
            request = Request(
                self.endpoint, data=body, method="POST",
                headers={"Authorization": f"Bearer {secret}",
                         "Content-Type": "application/json", "Accept": "application/json"},
            )
            attempt_started = time.perf_counter()
            response = None
            try:
                response = self._transport(request, self.timeout)
                scores = self._scores(response, questions)
            except Exception as exc:
                retry = attempt < self.max_attempts and self._retryable(exc)
                safe_message = str(exc).replace(secret, "[REDACTED]")
                event = {
                    **common, "status": "failed", "attempt": attempt,
                    "will_retry": retry, "error_type": type(exc).__name__,
                    "error": safe_message,
                    "latency_seconds": time.perf_counter() - attempt_started,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                if isinstance(response, dict):
                    event["response_model"] = response.get("model")
                    event["usage"] = response.get("usage", {})
                self.last_call_info = self.record_event(event)
                if retry:
                    if self.retry_backoff:
                        time.sleep(self.retry_backoff)
                    continue
                raise JevError(f"Jev decision {call_id!r} failed: {safe_message}") from None
            event = {
                **common, "status": "success", "attempt": attempt,
                "response_model": response["model"], "scores": scores,
                "usage": response.get("usage", {}),
                "latency_seconds": time.perf_counter() - attempt_started,
                "elapsed_seconds": time.perf_counter() - started,
            }
            self.last_call_info = self.record_event(event)
            return scores
        raise AssertionError("Unreachable retry state")


_BULLET = re.compile(r"^\s*\[([^\]]+)\]\s*helpful=\d+\s*harmful=\d+\s*::\s*(.*)")


class ReadSelector:
    """Rank existing entries for one question without changing stored memory."""

    def __init__(self, client: JevClient, k: int, batch_size: int = 20):
        for name, value in (("k", k), ("batch_size", batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.client, self.k, self.batch_size = client, k, batch_size

    def select(
        self, playbook: str, question: str, *, call_id: str, stage: str,
    ) -> tuple[str, dict[str, Any]]:
        lines = playbook.splitlines(keepends=True)
        bullets = []
        section = ""
        active_bullet = None
        for line_index, line in enumerate(lines):
            if line.lstrip().startswith("#"):
                section = line.strip()
                active_bullet = None
            match = _BULLET.match(line)
            if match:
                active_bullet = {"id": match.group(1), "text": match.group(2),
                                 "section": section, "line_index": line_index,
                                 "line_indices": [line_index]}
                bullets.append(active_bullet)
            elif active_bullet is not None:
                active_bullet["line_indices"].append(line_index)
                active_bullet["text"] += "\n" + line.rstrip("\r\n")
        ids = [bullet["id"] for bullet in bullets]
        if len(set(ids)) != len(ids):
            raise ValueError("Playbook contains duplicate bullet IDs")
        scores: dict[str, float] = {}
        request_ids = []
        if len(bullets) > self.k:
            for batch_index, start in enumerate(range(0, len(bullets), self.batch_size)):
                candidates = [{key: bullet[key] for key in ("id", "text", "section")}
                              for bullet in bullets[start:start + self.batch_size]]
                questions = {
                    f"relevance_{i}": {
                        "type": "noul",
                        "instructions": (
                            f"Does candidates[{i}] contain a formula, financial concept, "
                            "unit conversion, or solving strategy relevant to the question? "
                            "Judge relevance only; do not solve the question or verify arithmetic."
                        ),
                        "criteria": {"true": "Relevant and potentially usable for this question",
                                     "false": "Unrelated to solving this question"},
                    }
                    for i in range(len(candidates))
                }
                request_id = f"{call_id}:read:{batch_index}"
                batch_scores = self.client.decide(
                    {"question": question, "candidates": candidates}, questions,
                    call_id=request_id, stage=stage,
                )
                request_ids.append(request_id)
                for i, candidate in enumerate(candidates):
                    scores[candidate["id"]] = batch_scores[f"relevance_{i}"]
            ranked = sorted(bullets, key=lambda bullet: (-scores[bullet["id"]], bullet["line_index"]))
        else:
            ranked = bullets
        selected = ranked[:self.k]
        selected_indices = {index for bullet in selected for index in bullet["line_indices"]}
        bullet_indices = {index for bullet in bullets for index in bullet["line_indices"]}
        selected_playbook = "".join(
            line for index, line in enumerate(lines)
            if index not in bullet_indices or index in selected_indices
        )
        metadata = {
            "policy": "jev_read_top_k", "call_id": call_id, "stage": stage,
            "k": self.k, "candidate_count": len(bullets), "selected_count": len(selected),
            "selected_ids": [bullet["id"] for bullet in selected], "scores": scores,
            "jev_call_ids": request_ids, "jev_calls": len(request_ids),
            "reason": "ranked" if request_ids else "all_entries_fit",
        }
        self.client.record_event({"event": "read_selection", **metadata})
        return selected_playbook, metadata
