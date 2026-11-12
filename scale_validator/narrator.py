"""LLM narrative interpretation for scale validation results."""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-4-argon",
}

SYSTEM_PROMPT = """You are a psychometrician and survey methodologist advising UX researchers on scale quality.

Given a JSON summary of a Likert scale validation, write a clear 3–4 sentence narrative for a researcher. Cover:
1. Whether the scale's internal consistency is acceptable (interpret Cronbach's alpha plainly)
2. Any items that should be reviewed or dropped (low item-total correlations)
3. How many factors the data suggests (Kaiser criterion)
4. One concrete recommendation

Use plain language suitable for a practitioner, not an academic. Do not say "it is worth noting"."""


class ScaleNarrator:
    def __init__(self, backend: str, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        if backend not in DEFAULT_MODELS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(DEFAULT_MODELS)}")
        self.backend = backend
        self.model = model or DEFAULT_MODELS[backend]
        self.api_key = api_key

    def narrate(self, result_dict: dict) -> str:
        # Trim matrix from prompt to save tokens
        slim = {k: v for k, v in result_dict.items() if k != "inter_item_correlation_matrix"}
        prompt = f"Interpret this scale validation result:\n\n{json.dumps(slim, indent=2)}"
        try:
            if self.backend == "openai":
                return self._call_openai(prompt)
            elif self.backend == "anthropic":
                return self._call_anthropic(prompt)
            else:
                return self._call_gemini(prompt)
        except Exception as exc:
            logger.warning("LLM narration failed: %s", exc)
            return ""

    def _call_openai(self, prompt: str) -> str:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("pip install 'scale-validator[openai]'") from exc
        kwargs = {"api_key": self.api_key} if self.api_key else {}
        client = openai.OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=300,
        )
        return resp.choices[0].message.content or ""

    def _call_anthropic(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install 'scale-validator[anthropic]'") from exc
        kwargs = {"api_key": self.api_key} if self.api_key else {}
        client = anthropic.Anthropic(**kwargs)
        resp = client.messages.create(
            model=self.model, max_tokens=300, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}], temperature=0.3,
        )
        return resp.content[0].text if resp.content else ""

    def _call_gemini(self, prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("pip install 'scale-validator[gemini]'") from exc
        if self.api_key:
            genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(model_name=self.model, system_instruction=SYSTEM_PROMPT)
        resp = model.generate_content(prompt, generation_config={"temperature": 0.3, "max_output_tokens": 300})
        return resp.text or ""
