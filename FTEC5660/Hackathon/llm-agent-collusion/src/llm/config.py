"""Load experiment credentials and resolve explicit trader model choices."""

import os

from dotenv import load_dotenv
from openai import OpenAI

from src import PROJECT_ROOT
from src.schemas import resolve_model as resolve_model_alias

ENV_FILE = PROJECT_ROOT / ".env"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def load_environment() -> None:
    load_dotenv(ENV_FILE, override=False)


def resolve_model(requested: str) -> str:
    load_environment()
    return resolve_model_alias(os.environ.get("LLM_MODEL") or requested)


def make_client(**kwargs) -> OpenAI:
    load_environment()
    base_url = os.environ.get("LLM_BASE_URL") or OPENROUTER_BASE_URL
    # Never send an OpenRouter credential to a custom endpoint.
    key = (os.environ.get("LLM_API_KEY") or "EMPTY") if os.environ.get("LLM_BASE_URL") else os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY in llm-agent-collusion/.env")
    return OpenAI(base_url=base_url, api_key=key, **kwargs)
