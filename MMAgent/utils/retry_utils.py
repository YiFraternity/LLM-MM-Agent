import json
import logging
from functools import wraps
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    before_sleep_log,
)
from openai import OpenAIError, RateLimitError, APIError
from requests.exceptions import ConnectionError, Timeout as RequestsTimeout


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
def retry_on_api_error(
    max_attempts: int = 5,
    min_wait: int = 2,
    max_wait: int = 20,
    multiplier: int = 2,
    wait_time: int | None = None,
):
    """
    通用的 API 调用重试装饰器，处理 OpenAI 和网络相关错误。
    支持两种等待机制：
      1. 若传入 wait_time，则使用固定等待（适合逻辑可控的重试）
      2. 否则默认使用指数退避等待（适合网络波动与限流）
    """
    wait_strategy = (
        wait_fixed(wait_time)
        if wait_time is not None
        else wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait)
    )

    return retry(
        retry=retry_if_exception_type((
            OpenAIError,
            RateLimitError,
            APIError,
            ConnectionError,
            RequestsTimeout,
            TimeoutError,  # Python 内置 TimeoutError
        )),
        wait=wait_strategy,
        stop=stop_after_attempt(max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # ✅ 保留原始异常栈信息，方便调试
    )

def retry_on_logic_error(max_attempts=3, wait_time=3):
    """
    用于在抛出 LogicError 时重试（逻辑级别错误）。
    比如 LLM 输出为空、格式错误等。
    """
    return retry(
        retry=retry_if_exception_type(LogicError),
        stop=stop_after_attempt(max_attempts),
        wait=wait_fixed(wait_time),
        reraise=True,
    )


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
