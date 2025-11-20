"""
llm.py
"""
import os
import logging
import openai
from dotenv import load_dotenv
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
from openai import OpenAIError, RateLimitError, APIError

load_dotenv(override=True)


class LLM:
    def __init__(self, model_name=None, logger=None, user_id=None):
        self.model_name = model_name or os.getenv('MODEL_NAME')
        self.logger = logger or logging.getLogger(__name__)
        self.user_id = user_id
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.usages = []

        if self.model_name in ['deepseek-chat', 'deepseek-reasoner']:
            self.api_base = (
                os.getenv('DEEPSEEK_API_BASE')
                or os.getenv('OPENAI_API_BASE')
                or 'https://api.deepseek.com/v1'
            )
        else:
            self.api_base = os.getenv('OPENAI_API_BASE') or 'https://api.openai.com/v1'

        if not self.api_key:
            raise ValueError("API key not found in environment variables")

        self.client = openai.Client(api_key=self.api_key, base_url=self.api_base)

    def reset(self, api_key=None, api_base=None, model_name=None):
        if api_key:
            self.api_key = api_key
        if api_base:
            self.api_base = api_base
        if model_name:
            self.model_name = model_name
        self.client = openai.Client(api_key=self.api_key, base_url=self.api_base)

    def _log_usage(self, usage):
        if usage:
            self.usages.append(usage)
            self.logger.info(
                f"[LLM] UserID: {self.user_id}, Model: {self.model_name}, Usage: {usage}"
            )

    @retry(
        retry=retry_if_exception_type((OpenAIError, RateLimitError, APIError, ConnectionError, TimeoutError)),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING)
    )
    def _safe_completion(self, system: str, prompt: str, temperature: float = 0.7, timeout: int = 60):
        """Wrapper with tenacity-based retry"""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': prompt}
            ],
            temperature=temperature,
            top_p=0.95,
            timeout=timeout,
        )

    def generate(self, prompt: str, system: str = '', usage: bool = True, temperature: float = 0.7, timeout: int = 180):
        """Generate response with automatic retry (tenacity)"""
        self.logger.info(f"🚀  Calling OpenAI API | model={self.model_name}, temp={temperature}")
        try:
            response = self._safe_completion(system, prompt, temperature, timeout)
            answer = response.choices[0].message.content
            usage_data = {
                'completion_tokens': response.usage.completion_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'total_tokens': response.usage.total_tokens,
            }
            if usage:
                self._log_usage(usage_data)
            return answer

        except Exception as e:
            err_msg = f"[LLM:{self.model_name}] generation failed for user {self.user_id}: {e}"
            self.logger.error(err_msg)
            return err_msg

    def get_total_usage(self):
        total_usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }
        for usage in self.usages:
            for key, value in usage.items():
                total_usage[key] += value
        return total_usage

    def clear_usage(self):
        self.usages = []
