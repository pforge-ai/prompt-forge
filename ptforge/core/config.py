# filename: ptforge/core/config.py
# ptforge/core/config.py

import dataclasses
from typing import Optional, Dict, Any

# 导入 UpdateGranularity 和 BaseDataset (注意 BaseDataset 仅用于类型提示)
# Import UpdateGranularity and BaseDataset (Note: BaseDataset is for type hint only)
from .base import UpdateGranularity, BaseDataset

@dataclasses.dataclass
class OptimizationConfig:
    """
    用于配置 PromptForge 优化过程的数据类。
    Dataclass for configuring the PromptForge optimization process.
    Includes settings for both standard and RL-based optimization.
    """
    # --- General Settings ---
    epochs: int = 3  # 默认优化 3 轮 (Default to 3 optimization epochs) - Used by PromptOptimizer
    batch_size: int = 8 # 默认批次大小为 8 (Default batch size to 8)
    max_steps: Optional[int] = None # (可选) 优化总步数 (Optional total optimization steps) - Used by RLPromptOptimizer

    # --- PromptOptimizer Settings ---
    update_granularity: UpdateGranularity = UpdateGranularity.SECTION_REPHRASE # 默认更新粒度 (Default update granularity)

    # --- Validation & Early Stopping ---
    validation_dataset: Optional[BaseDataset] = None # 默认无验证集 (No validation set by default)
    early_stopping_patience: Optional[int] = None # 默认不启用早停 (Early stopping disabled by default)
    target_score: Optional[float] = None # (可选) 目标分数，达到则停止 (Optional target score to stop at)

    # --- RLPromptOptimizer Settings ---
    initial_exploration_rate: float = 0.3      # RL: Initial exploration rate
    min_exploration_rate: float = 0.05         # RL: Minimum exploration rate
    max_exploration_rate: float = 0.5         # RL: Maximum exploration rate
    exploration_increase_patience: int = 5     # RL: Increase exploration if no improvement for N steps
    exploration_increase_factor: float = 0.05  # RL: How much to increase exploration rate
    exploration_decrease_factor: float = 0.02  # RL: How much to decrease exploration rate on success
    initial_temperature: float = 1.0           # RL: Initial simulated annealing temperature
    final_temperature: float = 0.01            # RL: Final simulated annealing temperature
    temperature_decay_steps: Optional[int] = None # RL: Steps over which temperature decays (default: max_steps)
    action_memory_size: int = 20               # RL: Max entries in action memory
    action_memory_decay: float = 0.9           # RL: Time decay factor for action memory weights
    reset_exploration_per_epoch: bool = False # RL: Whether to reset exploration rate each epoch (if using epochs)

    # --- Monitoring & Checkpointing Settings ---
    use_live_monitor: bool = True                  # Enable/disable tqdm progress bar
    generate_report_at_end: bool = False           # Generate a final report markdown file
    tracker_config: Optional[Dict[str, Any]] = None # Configuration for OptimizationTracker (e.g., name, dir)


    def __post_init__(self):
        # 添加基本的配置验证 (Add basic configuration validation)
        if self.epochs <= 0 and self.max_steps is None:
            # Allow epochs=0 if max_steps is set for RL
             self.epochs = 0 # Or set a default like 1? Ensure it's non-negative.
             # raise ValueError("Either epochs must be positive or max_steps must be set.")
        elif self.epochs < 0:
             raise ValueError("Epochs cannot be negative.")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        if self.early_stopping_patience is not None:
            if self.early_stopping_patience <= 0:
                raise ValueError("Early stopping patience must be positive if set.")
            # Relaxing this constraint: Early stopping might be based on training score if no validation set
            # if self.validation_dataset is None:
            #     self.logger.warning("Early stopping is enabled but no validation dataset is provided. Stopping will be based on training score changes.")

        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive if set.")

        if self.temperature_decay_steps is None:
             self.temperature_decay_steps = self.max_steps # Default decay over all steps if max_steps is defined

        # Ensure tracker_config is a dict if provided
        if self.tracker_config is not None and not isinstance(self.tracker_config, dict):
             raise TypeError("tracker_config must be a dictionary if provided.")