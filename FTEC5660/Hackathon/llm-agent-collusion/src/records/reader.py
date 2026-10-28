import json
from pathlib import Path
from typing import Any, Dict, List


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def read_json(path: Path) -> Dict[str, Any]:
    """Read one result object; malformed records must not silently disappear."""
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        data = json.load(file, parse_constant=_reject_constant)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=_reject_constant)
                if not isinstance(record, dict):
                    raise ValueError("Expected a JSON object")
            except ValueError as exc:
                raise ValueError(f"Invalid record in {path}, line {line_number}: {exc}") from exc
            records.append(record)
    return records


def find_run_dirs(root: Path) -> List[Path]:
    """Find a single run or runs below an output root, using the new result files."""
    root = Path(root)
    if (root / "config.json").is_file():
        candidates = [root]
    else:
        candidates = sorted(path.parent for path in root.rglob("config.json"))
    return [
        path for path in candidates
        if all((path / name).is_file() for name in ("calls.jsonl", "market.jsonl", "summary.json"))
    ]
