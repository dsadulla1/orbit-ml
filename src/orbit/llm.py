"""Thin client for any OpenAI-compatible chat completions endpoint."""
import re
from dataclasses import dataclass
from typing import Protocol

from .config import LLMSettings

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str) -> Completion: ...


class OpenAICompatLLM:
    def __init__(self, settings: LLMSettings | None = None):
        from openai import OpenAI

        self.settings = settings or LLMSettings()
        self.model = self.settings.model
        # Local servers (vLLM) accept any key; the SDK still requires a non-empty one.
        api_key = self.settings.api_key or ("EMPTY" if self.settings.base_url else None)
        self.client = OpenAI(base_url=self.settings.base_url, api_key=api_key,
                             timeout=self.settings.timeout)

    def complete(self, system: str, user: str) -> Completion:
        kwargs = {}
        # Only send sampling knobs when configured: some hosted reasoning models reject them.
        if self.settings.temperature is not None:
            kwargs["temperature"] = self.settings.temperature
        if self.settings.max_tokens is not None:
            kwargs["max_completion_tokens"] = self.settings.max_tokens
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            **kwargs,
        )
        text = _THINK.sub("", resp.choices[0].message.content or "")
        # Some chat templates open <think> in the prompt, so only the closing tag is returned.
        text = text.rpartition("</think>")[2].strip()
        usage = resp.usage
        return Completion(text,
                          getattr(usage, "prompt_tokens", 0) or 0,
                          getattr(usage, "completion_tokens", 0) or 0)
