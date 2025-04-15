# prompt_forge/llms/__init__.py

# 导入基类 (Import base class)
from prompt_forge.core.base import BaseLLMClient

# 导入具体实现 (Import concrete implementations)
from .clients import OpenAIClient

# 定义公开接口 (Define public interface)
__all__ = [
    "BaseLLMClient",
    "OpenAIClient",
    # Add other client implementations here
]
