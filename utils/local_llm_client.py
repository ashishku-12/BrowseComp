"""
Local secondary model client - FREE, no API billing, no external calls.

Used only for Step 5 (redundancy/shortcut check) and Step 6a (blind-solve
contamination check), where the whole point is a model from a DIFFERENT
family than the primary DeepSeek pipeline, so it doesn't share DeepSeek's
blind spots. Running it locally means these two steps cost nothing beyond
your own electricity - no API key required.

NOTE: currently every step in this pipeline is calling with use_secondary=True
(intentional, per your setup), so in practice this IS the model doing all
the work right now, not just steps 5/6a.

`sample=True` switches from greedy (deterministic) to sampled decoding -
used for GENERATION steps (Step 1 seed picking, Step 2 hop exploration)
where you want variety across calls. Judgment/matching steps (3, 5, 6a/6b)
should keep sample=False (the default) for consistent grading.
"""
import json
import re
import threading

from config import (
    LOCAL_MODEL_ID, LOCAL_MODEL_4BIT, LOCAL_MODEL_MAX_NEW_TOKENS,
    GENERATION_TEMPERATURE, GENERATION_TOP_P,
)

_model = None
_tokenizer = None
_load_lock = threading.Lock()


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:  # re-check after acquiring lock
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[local_llm] loading {LOCAL_MODEL_ID} (first call only, this can take a few minutes)...")

        quant_config = None
        if LOCAL_MODEL_4BIT:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        _tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16 if not LOCAL_MODEL_4BIT else None,
        )
        print("[local_llm] model loaded.")


def _generate(system_prompt: str, user_prompt: str, max_new_tokens: int = None, sample: bool = False) -> str:
    _load_model()
    import torch

    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": user_prompt})

    tokenized = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    input_ids = tokenized["input_ids"] if hasattr(tokenized, "input_ids") else tokenized
    input_ids = input_ids.to(_model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens or LOCAL_MODEL_MAX_NEW_TOKENS,
        pad_token_id=_tokenizer.eos_token_id,
    )
    if sample:
        gen_kwargs.update(do_sample=True, temperature=GENERATION_TEMPERATURE, top_p=GENERATION_TOP_P)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None)  # deterministic - consistent grading

    with torch.no_grad():
        output = _model.generate(input_ids, **gen_kwargs)
    new_tokens = output[0][input_ids.shape[-1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    # Models sometimes add a short preamble. Decode the first complete JSON
    # object instead of using a greedy brace regex, which can include invalid
    # prose or multiple objects.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("No complete JSON object found", text, 0)


def call_local_llm(system_prompt: str, user_prompt: str, json_mode: bool = True, sample: bool = False) -> dict:
    """Local equivalent of llm_client.call_llm, used for Step 5's structured verdict
    (and, since use_secondary=True is used everywhere in this setup, every other step too)."""
    full_system = system_prompt
    if json_mode:
        full_system += (
            "\n\n/no_think\n"
            "Respond with ONLY a single valid JSON object. No reasoning, no prose, "
            "no code fences, no explanation outside the JSON."
        )

    raw = _generate(full_system, user_prompt, sample=sample)
    try:
        return _extract_json(raw)
    except Exception as e:
        raise ValueError(f"Local model did not return valid JSON: {e}\nRaw output:\n{raw}")


def call_local_llm_text(user_prompt: str, sample: bool = False) -> str:
    """Local equivalent of call_llm_no_tools_text, used for Step 6a's blind-solve check.
    No search tools are available here by construction - it's a plain local
    generate() call with no tool access, which is exactly what a blind-solve
    contamination check needs."""
    system = ("Answer the question directly and concisely using only what you already "
              "know. Do not say you cannot search - just give your best answer.")
    return _generate(system, user_prompt, max_new_tokens=256, sample=sample)