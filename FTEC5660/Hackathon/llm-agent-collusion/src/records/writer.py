"""Write one run's configuration, calls, rounds, and summary."""

import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any
from zoneinfo import ZoneInfo

from src.evaluation.evaluate import build_summary
from src.records.reader import read_jsonl


def timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="milliseconds")


class ExperimentLogger:
    def __init__(self, config: dict[str, Any], base_dir: str | Path = "results"):
        self.config = config
        self.log_dir = Path(base_dir) / config["run_id"]
        # Never overwrite another run, including a repeated seed in the same second.
        self.log_dir.mkdir(parents=True, exist_ok=False)
        self._lock = Lock()
        self._call_count = 0
        self._started = monotonic()
        self._secrets = [os.environ[key] for key in (
            "OPENROUTER_API_KEY", "LLM_API_KEY", "TYPESAFE_API_KEY"
        ) if os.environ.get(key)]
        self._write_json("config.json", config)
        for name in ("calls.jsonl", "market.jsonl"):
            (self.log_dir / name).touch()
        self.save_summary("running")

    def _write_json(self, name: str, record: dict):
        destination = self.log_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def redact(self, message: str) -> str:
        for secret in self._secrets:
            message = message.replace(secret, "[REDACTED]")
        return message

    def new_call_id(self) -> str:
        with self._lock:
            self._call_count += 1
            return f"c{self._call_count}"

    def _append(self, name: str, record: dict):
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        with self._lock:
            with (self.log_dir / name).open("a", encoding="utf-8") as file:
                file.write(line)

    def log_call(self, record: dict):
        self._append("calls.jsonl", record)

    def log_round(self, record: dict):
        self._append("market.jsonl", record)

    def save_summary(self, status: str, error: dict | None = None) -> dict:
        if error:
            error = dict(error, message=self.redact(error["message"]))
        with self._lock:
            summary = build_summary(
                self.config,
                read_jsonl(self.log_dir / "market.jsonl"),
                read_jsonl(self.log_dir / "calls.jsonl"),
                status=status,
                started_at=self.config["created_at"],
                ended_at=None if status == "running" else timestamp(),
                duration_seconds=monotonic() - self._started,
                error=error,
            )
            self._write_json("summary.json", summary)
        return summary
