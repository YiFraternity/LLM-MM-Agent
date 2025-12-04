import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from utils.retry_utils import (
    retry_on_api_error,
)

load_dotenv(override=True)


class LLM:
    """
    Unified LLM Wrapper supporting:
      - OpenAI models (o-series, GPT-4.1, GPT-5)
      - DeepSeek (OpenAI-compatible)
      - Anthropic Claude models
      - Prompt caching
      - Tenacity retries
      - Unified usage logging
    """

    def __init__(self, model_name=None, logger=None, user_id=None):
        self.model_name = model_name or os.getenv('MODEL_NAME')
        self.logger = logger or logging.getLogger(__name__)
        self.user_id = user_id
        self.usages = []

        # ---- Select provider based on model name ----
        if self.is_claude(self.model_name):
            import anthropic
            self.provider = "anthropic"
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            self.api_base = os.getenv("ANTHROPIC_API_BASE") or "https://api.anthropic.com/v1"
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not found")
            self.client = anthropic.Anthropic(api_key=self.api_key)

        else:
            self.provider = "openai"
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found")

            self.api_base = os.getenv("DEEPSEEK_API_BASE") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                max_retries=0,
            )

        self.logger.info(f"[INIT] provider={self.provider} model={self.model_name}")

    @staticmethod
    def is_deepseek(model):
        return "deepseek" in model.lower()

    @staticmethod
    def is_claude(model):
        return "claude" in model.lower()

    def _log_usage(self, usage):
        if usage:
            self.usages.append(usage)
            self.logger.info(
                f"[LLM] UserID={self.user_id}, Model={self.model_name}, Usage={usage}"
            )

    @retry_on_api_error(max_attempts=3, min_wait=30, max_wait=120)
    def _safe_completion(self, system: str, prompt: str, timeout: int = 60, **kwargs):
        if self.provider == "anthropic":
            return self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                max_tokens=32768,
                timeout=timeout,
                **kwargs,
            )
        else:
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': prompt},
                ],
                timeout=timeout,
                **kwargs,
            )

    def generate(self, prompt: str, system: str = '', usage: bool = True, timeout=1200, **kwargs):
        """Generate response with automatic retry (tenacity)"""
        self.logger.info(f"🚀  Calling OpenAI API | api-base={self.api_base} model={self.model_name}")
        try:
            response = self._safe_completion(system, prompt, timeout, **kwargs)

            if self.provider == "anthropic":
                answer = response.content[0].text
                usage_data = response.usage  # already dict-like

            else:
                answer = response.choices[0].message.content

                usage_data = {
                    'completion_tokens': response.usage.completion_tokens or 0,
                    'prompt_tokens': response.usage.prompt_tokens or 0,
                    'total_tokens': response.usage.total_tokens or 0,
                }
                if hasattr(response.usage, "prompt_tokens_details"):
                    details = response.usage.prompt_tokens_details
                    usage_data.update({
                        "cached_tokens": getattr(details, "cached_tokens", 0) or 0,
                    })
                if hasattr(response.usage, "completion_tokens_details"):
                    details = response.usage.completion_tokens_details
                    usage_data.update({
                        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
                    })

            if usage:
                self._log_usage(usage_data)
            return answer

        except Exception as e:
            err_msg = f"[LLM:{self.model_name}] generation failed for user {self.user_id}: {e}"
            self.logger.error(err_msg)
            raise type(e)(err_msg) from e

    def get_total_usage(self):
        total = {}
        for u in self.usages:
            for k, v in u.items():
                total[k] = total.get(k, 0) + v
        return total

    def clear_usage(self):
        self.usages = []
