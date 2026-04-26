import json
import time
import logging
from functools import wraps
from typing import Callable, Optional
from requests.exceptions import ConnectionError, Timeout as RequestsTimeout
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    before_sleep_log,
)
from openai import (
    APIConnectionError,
    APIStatusError,
    RateLimitError,
    APITimeoutError,
)


# ------------------------------
# 🔧 Logging Configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------
# ⚠️ Custom Exceptions
# ------------------------------
class LogicError(Exception):
    """Raised when logical or structural output is invalid."""
    pass


class EmptyOutputError(Exception):
    """Raised when LLM output is empty or malformed."""
    pass


# ------------------------------
# 🚦 Retry Decorators
# ------------------------------
def retry_on_api_error(max_attempts=5, min_wait=2, max_wait=20, multiplier=2, wait_time=None):
    """
    通用的 API 调用重试装饰器。
    支持实例方法（self, ...）且能自定义固定等待时间或指数退避。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait_strategy = (
                wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait)
                if wait_time is None
                else wait_fixed(wait_time)
            )

            retry_wrapper = retry(
                retry=retry_if_exception_type((
                    APIConnectionError,
                    APIStatusError,
                    RateLimitError,
                    APITimeoutError,
                    ConnectionError,
                    RequestsTimeout,
                    TimeoutError,
                )),
                wait=wait_strategy,
                stop=stop_after_attempt(max_attempts),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            )(func)

            return retry_wrapper(*args, **kwargs)

        return wrapper
    return decorator


# ------------------------------
# 🧩 Output Validation Decorators
# ------------------------------
def ensure_parsed_python_code(func):
    """
    执行 LLM 调用后自动提取 Python 代码并验证。
    确保输出包含 ```python``` 代码块，且代码长度合理。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        raw_text = func(*args, **kwargs)

        if not isinstance(raw_text, str) or not raw_text.strip():
            raise LogicError(f"[{func.__name__}] Output is empty or invalid type.")

        if "```python" not in raw_text:
            raise LogicError(f"[{func.__name__}] No ```python``` code block found.")

        try:
            code = raw_text.split("```python", 1)[1].split("```", 1)[0].strip()
        except Exception as e:
            raise LogicError(f"[{func.__name__}] Failed to extract Python code: {e}")

        if len(code) < 10:
            raise LogicError(f"[{func.__name__}] Extracted code too short ({len(code)} chars).")

        return code
    return wrapper


def ensure_parsed_json_output(func):
    """
    提取并验证 LLM 输出中的 JSON 内容。
    确保返回值为合法 JSON。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        raw_text = func(*args, **kwargs)

        if not isinstance(raw_text, str) or not raw_text.strip():
            raise LogicError(f"[{func.__name__}] Output is empty or invalid type.")

        # 提取 JSON 块
        if "```json" not in raw_text:
            raise LogicError(f"[{func.__name__}] No ```json``` code block found.")

        try:
            json_str = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
            parsed = json.loads(json_str)
        except Exception as e:
            raise LogicError(f"[{func.__name__}] Failed to parse JSON: {e}")

        # 可选：结构校验
        if not isinstance(parsed, dict):
            raise LogicError(f"[{func.__name__}] Parsed JSON is not a dictionary.")

        return parsed
    return wrapper

def async_retry_on_api_error(
    max_attempts=5,
    min_wait=2,
    max_wait=20,
    multiplier=2,
    wait_time=None,
):
    wait_strategy = (
        wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait)
        if wait_time is None
        else wait_fixed(wait_time)
    )

    return retry(
        retry=retry_if_exception_type((
            APIConnectionError,
            APIStatusError,
            RateLimitError,
            APITimeoutError,
            ConnectionError,
            RequestsTimeout,
            TimeoutError,
        )),
        wait=wait_strategy,
        stop=stop_after_attempt(max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

def reflective_retry_on_logic_error(
    max_attempts: int = 3,
    wait_time: float = 2,
    reflection_template: Optional[str] = None,
    error_types=(LogicError, EmptyOutputError)
):
    """
    ✅ 反思性重试装饰器（简化版）
    - 捕获逻辑错误或空输出错误；
    - 每次失败自动反思并修改 prompt/文本参数；
    - 使用固定等待，无需依赖 tenacity 的生成器；
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):

            args_list = list(args)
            last_error = None

            # ---- 识别 prompt 参数 ----
            def append_to_prompt(text: str):
                """始终在 prompt 末尾添加反思语句"""
                # 优先在 kwargs["prompt"] 加
                if "prompt" in kwargs and isinstance(kwargs["prompt"], str):
                    kwargs["prompt"] += text
                    return

                # 否则可能 prompt 在 args 的第二个位置 (self, prompt, ...)
                if len(args_list) > 1 and isinstance(args_list[1], str):
                    args_list[1] = args_list[1] + text
                    return

                # 找不到 prompt → 不追加
                logger.warning("⚠️ No valid prompt parameter found to append reflection text.")

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args_list, **kwargs)

                except error_types as e:
                    last_error = e
                    logger.warning(f"⚠️ [{func.__name__}] Attempt {attempt}/{max_attempts} failed: {e}")

                    if attempt == max_attempts:
                        logger.error(f"❌ [{func.__name__}] All retry attempts failed.")
                        raise

                    # 生成反思性提示
                    reflective_instruction = (
                        reflection_template.format(error=str(e))
                        if reflection_template
                        else f"\n\n⚠️ Previous attempt failed: {e}. Please correct and regenerate valid output."
                    )

                    append_to_prompt(reflective_instruction)

                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

            # 最后仍失败，抛出异常
            raise last_error

        return wrapper

    return decorator
