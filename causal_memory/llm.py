"""
LLM provider for the causal memory layer.

Speaks to Google Gemini (generativelanguage.googleapis.com, v1beta).
Falls back to OpenAI-compatible endpoints if no Gemini key is present.

Environment variables:
    GEMINI_API_KEY   -> Gemini (default)
    OPENAI_API_KEY   -> OpenAI-compatible /chat/completions (fallback)
    LLM_MODEL        -> override model name (defaults per provider)

Used for:
  - extract_causality(): propose CAUSES edges + mechanisms from an event batch
  - why(): turn a causal path into a natural-language answer
"""

import json
import os
import urllib.request
import urllib.error

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class LLMError(Exception):
    pass


class LLM:
    def __init__(self, model: str | None = None):
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.model = model
        if not self.gemini_key and not self.openai_key:
            raise LLMError(
                "No LLM credentials: set GEMINI_API_KEY or OPENAI_API_KEY."
            )

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        """One completion call, first healthy provider wins."""
        if self.gemini_key:
            try:
                return self._gemini(system, user, json_mode)
            except Exception as e:
                if not self.openai_key:
                    raise LLMError(f"Gemini call failed: {e}") from e
                # fall through to OpenAI
        if self.openai_key:
            return self._openai(system, user, json_mode)
        raise LLMError("All LLM providers failed.")

    # ------------------------------------------------------------------ gemini

    def _gemini(self, system: str, user: str, json_mode: bool) -> str:
        model = self.model or DEFAULT_GEMINI_MODEL
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.gemini_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
        }
        if json_mode:
            payload["generationConfig"] = {
                "responseMimeType": "application/json",
                "temperature": 0.2,
            }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
        parts = body["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)

    # ---------------------------------------------------------------- openai

    def _openai(self, system: str, user: str, json_mode: bool) -> str:
        model = self.model or DEFAULT_OPENAI_MODEL
        url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"]


# -------------------------------------------------------------------------- utils


def extract_json(text: str) -> dict:
    """Parse a JSON object out of an LLM response, tolerating fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise LLMError(f"Could not parse JSON from LLM output: {text[:300]}")