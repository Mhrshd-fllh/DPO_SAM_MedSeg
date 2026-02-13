from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import os


@dataclass
class GPTResult:
    descriptions: List[str]


class OpenAIGPTAdapter:
    """
    Real GPT adapter using OpenAI API.
    Requires environment variable: OPENAI_API_KEY
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        system_prompt: str = "You are a helpful medical imaging assistant. Write short, generic descriptions.",
        max_tokens: int = 80,
        temperature: float = 0.2,
    ):
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} is not set. Export your OpenAI API key in the environment to use GPT prompting."
            )

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def describe(self, labels: List[str]) -> GPTResult:
        """
        labels: list[str] length B (e.g., organ/disease label).
        Returns one short generic description per label.
        """
        descs: List[str] = []

        for lab in labels:
            user_prompt = (
                "Write one short generic description (1 sentence) for a medical image related to this label: "
                f"{lab}. Avoid specific measurements. Keep it general."
            )

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            txt = resp.choices[0].message.content.strip()
            descs.append(txt)

        return GPTResult(descriptions=descs)
