"""
llm_client.py
==============
Thin, optional wrapper around the Anthropic API. Every caller in this
project must work correctly whether or not an API key is configured --
this module is the single place that decides "is an LLM available right
now," so that decision doesn't get duplicated (and potentially handled
inconsistently) across the chatbot, the explainer, etc.

Set the ANTHROPIC_API_KEY environment variable before starting the server
to enable real LLM generation. Without it, `is_available()` returns False
and callers should use their own template/extractive fallback -- this
module never fabricates a key or silently no-ops with a fake answer.
"""

import os
from typing import List, Optional

_client = None
_checked = False

DEFAULT_MODEL = "claude-sonnet-4-5"


def is_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _get_client():
    global _client, _checked
    if _checked:
        return _client
    _checked = True
    if not is_available():
        _client = None
        return None
    try:
        import anthropic
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except Exception:
        _client = None
    return _client


def generate(system_prompt: str, user_message: str, max_tokens: int = 600) -> Optional[str]:
    """Returns the model's text response, or None if no LLM is configured
    or the call fails for any reason (network, auth, etc). Callers must
    handle the None case with their own fallback."""
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return "\n".join(parts).strip() or None
    except Exception:
        return None
