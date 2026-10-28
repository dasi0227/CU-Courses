"""Render the auction prompts from the bundled templates."""

from pathlib import Path
import re

from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _normalize_spacing(prompt: str) -> str:
    """Keep one blank line around headings and between content blocks."""
    prompt = re.sub(r"(?m)^(#{1,6}[ \t]+[^\n]+)$", r"\n\1\n", prompt)
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)*", "\n\n", prompt).strip()


def render_prompt_messages(
    template_dir: str,
    prompt_template: str,
    **kwargs,
) -> list[dict[str, str]]:
    """Render the template's system and user blocks as separate API messages."""
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(prompt_template)
    roles = ("system", "user")
    if any(role not in template.blocks for role in roles):
        raise ValueError(f"{prompt_template} must define system and user blocks")
    return [
        {
            "role": role,
            "content": _normalize_spacing("".join(template.blocks[role](template.new_context(kwargs)))),
        }
        for role in roles
    ]
