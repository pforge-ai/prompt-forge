# ptforge/utils/__init__.py
"""
提供各种通用工具函数和类。
包括重试策略、JSON解析、提示构建、序列化支持等。

Provides various utility functions and classes.
Includes retry strategies, JSON parsing, prompt building, serialization support, etc.
"""

from .retry import (
    RetryError,
    RetryStrategy,
    ExponentialBackoff,
    ExponentialBackoffWithJitter,
    LinearBackoff,
    FixedDelayWithJitter,
    retry,
)

from .json_helper import (
    parse_json_robust,
    extract_json_str,
    fix_malformed_json,
    extract_action_json,
)

from .prompt_builder import (
    PromptBuilder,
    TokenBudgetManager,
)

from .serialization import (
    save_experiment,
    load_experiment,
    find_experiments,
    save_checkpoint,
    load_checkpoint,
    export_template_to_json,
    SerializationError,
)

# Define public API
__all__ = [
    # Retry utilities
    "RetryError",
    "RetryStrategy",
    "ExponentialBackoff",
    "ExponentialBackoffWithJitter",
    "LinearBackoff",
    "FixedDelayWithJitter",
    "retry",
    
    # JSON utilities
    "parse_json_robust",
    "extract_json_str",
    "fix_malformed_json",
    "extract_action_json",
    
    # Prompt building utilities
    "PromptBuilder",
    "TokenBudgetManager",
    
    # Serialization utilities
    "save_experiment",
    "load_experiment",
    "find_experiments",
    "save_checkpoint",
    "load_checkpoint",
    "export_template_to_json",
    "SerializationError",
]