#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an accurate Hong Kong supermarket receipt accountant. "
                "Read only the attached receipt and return valid JSON, with no "
                "markdown or explanation. Use decimal HKD amounts as strings.",
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": (
                            "Extract three things from this one receipt:\n"
                            "1. subtotal_hkd: the printed SUBTOTAL before rounding.\n"
                            "2. final_payment_hkd: the final amount actually paid, "
                            "after applying ROUNDING (for example the OCTOPUS, CASH, "
                            "or CARD payment amount).\n"
                            "3. discounts_hkd: an array containing the absolute "
                            "amount of every discount, promotion, coupon, or other "
                            "price reduction that must be added back to get the "
                            "pre-discount bill. Do not include ROUNDING, payments, "
                            "taxes, or item prices in this array. Include each "
                            "discount line exactly once.\n\n"
                            "Return exactly this JSON shape: "
                            '{"subtotal_hkd":"102.31",'
                            '"final_payment_hkd":"102.30",'
                            '"discounts_hkd":["5.39"]}. '
                            "If there is no discount, use an empty array. Do not "
                            "confuse the subtotal with the final payment."
                        ),
                    },
                    {"type": "image_url", "image_url": "{image_url}"},
                ],
            ),
        ]
    )
    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        timeout=60,
        max_retries=2,
    )
    return prompt | model


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    import sys
    import time

    def parse_amount(value: Any) -> Decimal:
        if isinstance(value, bool) or value is None:
            raise ValueError("missing amount")
        text = str(value).strip().replace(",", "")
        match = re.search(r"-?\d+(?:\.\d{1,2})?", text)
        if not match:
            raise ValueError(f"invalid amount: {value!r}")
        return Decimal(match.group(0)).quantize(Decimal("0.01"))

    def parse_receipt(result: Any) -> tuple[Decimal, Decimal]:
        content = getattr(result, "content", result)
        if isinstance(content, list):
            content = "".join(
                block if isinstance(block, str) else block.get("text", "")
                for block in content
                if isinstance(block, str) or isinstance(block, dict)
            )
        text = str(content).strip()
        # Accept a fenced JSON object or a short preamble if the model ignores
        # the JSON-only instruction, while still validating every required field.
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("model response did not contain a JSON object")
        data = json.loads(text[start : end + 1])
        subtotal = parse_amount(data["subtotal_hkd"])
        paid = parse_amount(data["final_payment_hkd"])
        discounts = data["discounts_hkd"]
        if not isinstance(discounts, list):
            raise ValueError("discounts_hkd must be an array")
        without_discount = subtotal + sum(
            (abs(parse_amount(amount)) for amount in discounts), Decimal("0.00")
        )
        return paid, without_discount.quantize(Decimal("0.01"))

    total_paid = Decimal("0.00")
    total_without_discount = Decimal("0.00")
    for image in images:
        request = {"image_url": image_data_url(image)}
        parsed = None
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                parsed = parse_receipt(chain.invoke(request))
                break
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(1)
        if parsed is None:
            # Keep the provided runner alive so it can still write results.csv;
            # the warning makes a failed receipt extraction visible to the user.
            print(
                f"Warning: could not extract {image.name}: {last_error}",
                file=sys.stderr,
            )
            continue
        paid, without_discount = parsed
        total_paid += paid
        total_without_discount += without_discount

    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_without_discount:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
