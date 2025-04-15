# prompt_forge/datasets/__init__.py

# 导入基类 (Import base class)
from prompt_forge.core.base import BaseDataset

# 导入具体加载器 (Import concrete loaders)
from .loaders import JsonlDataset, CsvDataset

# 定义公开接口 (Define public interface)
__all__ = [
    "BaseDataset",
    "JsonlDataset",
    "CsvDataset",
    # Add other dataset loaders here as they are implemented
]
