#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path
from functools import lru_cache
import openai
import tiktoken
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Always load the project-root .env, independent of the working directory.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

def initialize_clients(api_provider):
    """Initialize separate clients for generator, reflector, and curator"""
    if api_provider == "dashscope":
        base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for the DashScope provider")
    elif api_provider == "sambanova":
        # Use SambaNova API
        base_url = "https://api.sambanova.ai/v1"
        api_key = os.getenv('SAMBANOVA_API_KEY', '')
        if not api_key:
            raise ValueError("SambaNova api key not found in environment variables")
    elif api_provider == "together":
        # Use Together API
        base_url = "https://api.together.xyz/v1"
        api_key = os.getenv('TOGETHER_API_KEY', '')
        if not api_key:
            raise ValueError("Together api key not found in environment variables")
    elif api_provider == "openai":
        # Use OpenAI API
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            raise ValueError("OpenAI api key not found in environment variables")
    elif api_provider == "commonstack":
        # Use Commonstack API
        base_url = "https://api.commonstack.ai/v1"
        api_key = os.getenv('COMMONSTACK_API_KEY', '')
        if not api_key:
            raise ValueError("Commonstack api key not found in environment variables")
    else:
        raise ValueError(
            f"Invalid api_provider name: {api_provider}. Must be 'dashscope', 'sambanova', 'together', 'openai', or 'commonstack'"
        )
        
    generator_client = openai.OpenAI(api_key=api_key, base_url=base_url)
    reflector_client = openai.OpenAI(api_key=api_key, base_url=base_url)
    curator_client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    print(f"Using {api_provider} API for all models")
    return generator_client, reflector_client, curator_client

def get_section_slug(section_name):
    """Convert section name to slug format (3-5 chars)"""
    # Common section mappings - updated to match original sections
    slug_map = {
        "financial_strategies_and_insights": "fin",
        "formulas_and_calculations": "calc",
        "code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics": "prob",
        "context_clues_and_indicators": "ctx",
        "others": "misc",
        "meta_strategies": "meta"
    }
    
    # Clean and convert to snake_case
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and")
    
    if clean_name in slug_map:
        return slug_map[clean_name]
    
    # Generate slug from first letters
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5])

def extract_boxed_content(text):
    """Helper function to extract content from \\boxed{} format"""
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return None
    
    start = match.end() - 1  # Position of opening brace
    brace_count = 0
    i = start
    
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1:i]  # Content between braces
        i += 1
    return None

_NO_ANSWER = "No final answer found"
_NUMBER_PATTERN = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def extract_answer(response):
    """Extract JSON, legacy answer formats, or an unadorned numerical answer."""
    if not isinstance(response, str):
        return _NO_ANSWER
    response = response.strip()
    if re.fullmatch(_NUMBER_PATTERN, response):
        return response
    try:
        # First try JSON parsing
        parsed = json.loads(response)
        answer = str(parsed.get("final_answer", _NO_ANSWER))
        return answer  
            
    except (json.JSONDecodeError, KeyError, AttributeError):
        # JSON parsing failed, use fallback logic
        matches = re.findall(r"Finish\[(.*?)\]", response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Try to get final answer from JSON style response with regex matching 
        # Try double quotes first
        matches = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Try single quotes
        matches = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Handle JSON format without quotes (for simple expressions)
        matches = re.findall(r'[\'"]final_answer[\'"]\s*:\s*([^,}]+)', response)
        if matches:
            answer = matches[-1].strip()
            # Clean up trailing characters
            answer = re.sub(r'[,}]*$', '', answer)
            return answer
        
        # Fallback for "The final answer is: X" pattern with boxed
        final_answer_pattern = r'[Tt]he final answer is:?\s*\$?\\boxed\{'
        match = re.search(final_answer_pattern, response)
        if match:
            # Extract boxed content starting from this match
            remaining_text = response[match.start():]
            boxed_content = extract_boxed_content(remaining_text)
            if boxed_content:
                return boxed_content
        
        # More general pattern for "final answer is X"
        matches = re.findall(r'[Tt]he final answer is:?\s*([^\n]+)', response)
        if matches:
            answer = matches[-1].strip()
            # Clean up common formatting
            answer = re.sub(r'^\$?\\boxed\{([^}]+)\}\$?$', r'\1', answer)
            answer = answer.replace('$', '').strip()
            # A prose sentence's final period must not truncate decimal answers.
            if answer.endswith('.') and re.fullmatch(_NUMBER_PATTERN, answer[:-1]):
                answer = answer[:-1]
            if answer:
                return answer
        
        return _NO_ANSWER
    
@lru_cache(maxsize=1)
def _get_token_encoding():
    """Load/cache token data only when token counting is actually requested."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(prompt: str) -> int:
    return len(_get_token_encoding().encode(prompt))


def compact_prediction(record):
    """Keep scoring evidence and a link to the sole raw LLM call record."""
    return {key: record.get(key) for key in (
        "index", "sample_id", "call_id", "stage", "prediction", "target",
        "is_correct", "api_success", "error",
    )}


def evaluate_single_test_sample(args_tuple, data_processor) -> Tuple[Dict, str]:
    """Return a record even when generation or answer evaluation fails."""
    (i, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode) = args_tuple
    result = {
        "index": i,
        "sample_id": task_dict.get("sample_id", task_dict.get("id", i)),
        "call_id": f"test_eval_{i}",
        "response": None,
        "prediction": _NO_ANSWER,
        "final_answer": _NO_ANSWER,
        "target": task_dict.get("target"),
        "is_correct": False,
        "api_success": False,
        "success": False,
        "error": None,
    }
    try:
        gen_response, _, _ = generator.generate(
            question=task_dict["question"],
            playbook=playbook,
            context=task_dict["context"],
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=result["call_id"],
            log_dir=log_dir,
        )
        result["response"] = gen_response
        if (
            not isinstance(gen_response, str)
            or not gen_response.strip()
            or gen_response.startswith("INCORRECT_DUE_TO_EMPTY_RESPONSE")
        ):
            raise ValueError("Model returned an empty response")
        result["api_success"] = True
        final_answer = extract_answer(gen_response)
        result.update({
            "prediction": final_answer,
            "final_answer": final_answer,
            "is_correct": bool(data_processor.answer_is_correct(final_answer, task_dict["target"])),
            "success": True,
        })
        return result, None
    except Exception as error:
        message = f"Error evaluating sample {i}: {type(error).__name__}: {str(error)}"
        result["error"] = message
        return result, message


def evaluate_test_set(data_processor, generator, playbook, test_samples,
                      max_tokens=4096, log_dir=None, max_workers=20,
                      use_json_mode=False, stage="eval", predictions_path=None,
                      compact_predictions=False) -> Tuple[Dict, Dict]:
    """Retain all N inputs and return input-ordered predictions, including errors.

    Formula accuracy uses the full N denominator. The processor still defines
    legacy task-specific metrics, including FiNER's label-level accuracy, with
    failed samples included as missing answers. Logs live under log_dir/stage;
    predictions_path can place the prediction file outside that log directory.
    Compact predictions link to raw calls instead of copying their responses.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    stage_path = Path(stage)
    if not str(stage).strip() or stage_path.is_absolute() or ".." in stage_path.parts:
        raise ValueError("stage must be a non-empty relative path without '..'")
    stage_log_dir = None
    if log_dir is not None:
        stage_log_dir = Path(log_dir) / stage_path
        stage_log_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = Path(predictions_path) if predictions_path is not None else (
        stage_log_dir / "predictions.jsonl" if stage_log_dir is not None else None
    )
    if prediction_file is not None:
        prediction_file.parent.mkdir(parents=True, exist_ok=True)

    count = len(test_samples)
    print(f"EVALUATING {stage}: {count} samples, {max_workers} workers")
    args_list = [
        (i, sample, generator, playbook, max_tokens,
         str(stage_log_dir) if stage_log_dir is not None else None, use_json_mode)
        for i, sample in enumerate(test_samples)
    ]
    predictions = [None] * count
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_single_test_sample, args, data_processor): args[0]
            for args in args_list
        }
        for future in as_completed(futures):
            result, error = future.result()
            result["stage"] = str(stage)
            if (compact_predictions and result["api_success"]
                    and result["prediction"] == _NO_ANSWER and not result["error"]):
                result["error"] = "AnswerParseError: no extractable final answer"
            predictions[result["index"]] = result
            if error:
                print(error)

    correct = sum(record["is_correct"] for record in predictions)
    api_success_count = sum(record["api_success"] for record in predictions)
    answers = [record["prediction"] for record in predictions]
    targets = [record["target"] for record in predictions]
    # Preserve task-specific scoring while never removing failed rows.
    if count and hasattr(data_processor, "evaluate_accuracy"):
        accuracy = data_processor.evaluate_accuracy(answers, targets)
    else:
        accuracy = correct / count if count else 0.0
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": count,
        "no_answer": sum(record["prediction"] == _NO_ANSWER for record in predictions),
        "api_success_count": api_success_count,
        "api_failure_count": count - api_success_count,
        "success_rate": api_success_count / count if count else 0.0,
        "stage": str(stage),
        "predictions": predictions,
    }
    error_logs = {
        "accuracy": accuracy,
        "errors": [
            {
                "index": record["index"],
                "sample_id": record["sample_id"],
                "prediction": record["prediction"],
                "ground_truth": record["target"],
                "api_success": record["api_success"],
                "error": record["error"],
            }
            for record in predictions if not record["is_correct"]
        ],
    }
    if compact_predictions:
        predictions = [compact_prediction(record) for record in predictions]
        results["predictions"] = predictions
    if prediction_file is not None:
        with prediction_file.open("w", encoding="utf-8") as handle:
            for record in predictions:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Final accuracy: {accuracy:.3f} ({correct}/{count}); API success: {api_success_count}/{count}")
    return results, error_logs
