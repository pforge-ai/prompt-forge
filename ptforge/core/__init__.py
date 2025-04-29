# filename: ptforge/core/__init__.py
"""
Core components for the Prompt Forge framework, including optimizers, base types, and configuration.
"""

from .base import (
    BaseDataset,
    BaseLLMClient,
    BaseMetric,
    MetricResult,
    UpdateGranularity,
)
from .config import OptimizationConfig
from .optimizer import PromptOptimizer, OptimizationResult
from .rl_optimizer import RLPromptOptimizer, RLOptimizationResult
from .action_memory import ActionMemory, ActionRecord # If needed directly
from .reward_calculator import ( # If needed directly
    RewardCalculator,
    RelativeImprovementReward,
    AbsoluteChangeReward,
    ThresholdReward,
    CompoundReward,
    EnhancedRewardCalculator,
    create_default_reward_calculator,
)
from .action_executor import ActionExecutor, BatchActionExecutor # If needed directly


__all__ = [
    # Base types
    "BaseDataset",
    "BaseLLMClient",
    "BaseMetric",
    "MetricResult",
    "UpdateGranularity",

    # Configuration
    "OptimizationConfig",

    # Optimizers & Results
    "PromptOptimizer",
    "OptimizationResult",
    "RLPromptOptimizer",
    "RLOptimizationResult",

    # RL Components (Export if users might need to customize/use directly)
    "ActionMemory",
    "ActionRecord",
    "RewardCalculator",
    "create_default_reward_calculator",
    "RelativeImprovementReward", # Expose specific calculators?
    "AbsoluteChangeReward",
    "ThresholdReward",
    "CompoundReward",
    "EnhancedRewardCalculator",
    "ActionExecutor",
    "BatchActionExecutor",
]