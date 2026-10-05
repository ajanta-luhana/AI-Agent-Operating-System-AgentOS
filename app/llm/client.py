"""Ollama client with structured-output + retry loop (small local models are flaky)."""
import re
import httpx
from pydantic import BaseModel, ValidationError
from app.config import settings

USAGE = {"prompt": 0, "completion": 0}   # read by tracing decorator (single-threaded worker)


def _chat(messages, model=None, fmt=None, temperature=0.0) -> str:
    payload = {
        "model": model or settings.chat_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 4096},
    }
    if fmt:
        payload["format"] = fmt
    r = httpx.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=900)
    r.raise_for_status()
    data = r.json()
    USAGE["prompt"] += data.get("prompt_eval_count", 0)
    USAGE["completion"] += data.get("eval_count", 0)
    return data["message"]["content"]


def text(system: str, user: str, temperature: float = 0.2) -> str:
    return _chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                 temperature=temperature)


def code(system: str, user: str) -> str:
    """Returns python source (fences stripped) from the coder model."""
    raw = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                model=settings.code_model)
    m = re.search(r"```(?:python)?\n(.*?)```", raw, re.S)
    return (m.group(1) if m else raw).strip()


def structured(schema: type[BaseModel], system: str, user: str, retries: int = 3) -> BaseModel:
    messages = [
        {"role": "system", "content": system + "\nRespond ONLY with JSON matching the given schema."},
        {"role": "user", "content": user},
    ]
    last_err = None
    for _ in range(retries):
        raw = _chat(messages, fmt=schema.model_json_schema())
        try:
            return schema.model_validate_json(raw)
        except ValidationError as e:
            last_err = e
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"Invalid output: {e}\nFix it and return valid JSON only."}]
    raise ValueError(f"LLM failed to produce valid {schema.__name__}: {last_err}")


def embed(texts: list[str]) -> list[list[float]]:
    r = httpx.post(f"{settings.ollama_url}/api/embed",
                   json={"model": settings.embed_model, "input": texts}, timeout=300)
    r.raise_for_status()
    return r.json()["embeddings"]
