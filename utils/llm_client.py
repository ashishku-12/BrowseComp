"""
Thin wrapper around DeepSeek's OpenAI-compatible chat completion API.

`use_secondary=True` routes to the LOCAL, free model (utils/local_llm_client.py)
instead of DeepSeek - used only by Step 5 and Step 6a, which specifically need
a different-family model with no shared blind spots. Running that locally
means those two steps incur zero API billing.

`sample=True` (only meaningful when use_secondary=True) switches the local
model to sampled decoding for generation-diversity - see local_llm_client.py.
"""
import json
import re
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL_STRONG
from utils import local_llm_client

_primary_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _extract_json(text: str):
    """Model output should be pure JSON, but strip code fences defensively."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return json.loads(text)


def call_llm(system_prompt: str, user_prompt: str, model: str = None, json_mode: bool = True,
             temperature: float = 0.3, use_secondary: bool = False, sample: bool = False) -> dict:
    """
    Calls the LLM and returns a parsed dict. Raises on malformed JSON so
    the calling step can log a failure record instead of silently
    proceeding with garbage data.

    use_secondary=True -> local free model (no API cost, no `model` arg needed).
    use_secondary=False -> DeepSeek API, using `model` (defaults to the strong tier).
    sample=True -> local model uses sampled (non-deterministic) decoding for
    variety - only applies when use_secondary=True; ignored on the DeepSeek path.
    """
    if use_secondary:
        return local_llm_client.call_local_llm(system_prompt, user_prompt, json_mode=json_mode, sample=sample)

    model = model or DEEPSEEK_MODEL_STRONG
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = _primary_client.chat.completions.create(**kwargs)
    raw = resp.choices[0].message.content
    try:
        return _extract_json(raw)
    except Exception as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nRaw output:\n{raw}")


def call_llm_no_tools_text(user_prompt: str, model: str = None, use_secondary: bool = False, sample: bool = False) -> str:
    """Plain text call, no search tools available to the model at all.
    Used for Step 6a's blind-solve contamination check.

    use_secondary=True -> local free model, genuinely tool-less by construction.
    use_secondary=False -> DeepSeek API (no tools attached in this call either)."""
    if use_secondary:
        return local_llm_client.call_local_llm_text(user_prompt, sample=sample)

    model = model or DEEPSEEK_MODEL_STRONG
    resp = _primary_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer the question directly and concisely using only what you already know. Do not say you cannot search - just give your best answer."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()