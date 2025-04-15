# prompt_forge/metrics/__init__.py

# 导入基类和结果类 (Import base class and result class)
from ptforge.core.base import BaseMetric, MetricResult

# 导入具体实现 (Import concrete implementations)
from .implementations import ExactMatchAccuracy

# 定义公开接口 (Define public interface)
__all__ = [
    "BaseMetric",
    "MetricResult",
    "ExactMatchAccuracy",
    # Add other metric implementations here
    # 例如: "RougeScore", "SemanticSimilarity"
]
