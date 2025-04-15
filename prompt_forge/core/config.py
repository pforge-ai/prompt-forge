# prompt_forge/core/config.py

import dataclasses
from typing import Optional

# 导入 UpdateGranularity 和 BaseDataset (注意 BaseDataset 仅用于类型提示)
# Import UpdateGranularity and BaseDataset (Note: BaseDataset is for type hint only)
from .base import UpdateGranularity, BaseDataset

@dataclasses.dataclass
class OptimizationConfig:
    """
    用于配置 PromptForge 优化过程的数据类。
    Dataclass for configuring the PromptForge optimization process.

    Attributes:
        epochs: 优化的总轮数。 (Total number of optimization epochs.)
        batch_size: 每个优化步骤处理的数据批次大小。 (Batch size for processing data in each optimization step.)
        update_granularity: 控制 Prompt 更新范围和方式的粒度级别。 (Granularity level controlling the scope and manner of prompt updates.)
        validation_dataset: (可选) 用于评估和选择最佳 Prompt 的验证数据集。如果提供，将优先使用验证集分数来确定最佳模板。(Optional validation dataset used for evaluating and selecting the best prompt. If provided, validation scores will be prioritized for determining the best template.)
        early_stopping_patience: (可选) 早停耐心值。如果在指定轮数内验证分数没有提升，则提前停止优化。(Optional early stopping patience. If validation score doesn't improve for the specified number of epochs, stop optimization early.)
        # 可以根据需要添加更多配置项，例如:
        # More configuration options can be added as needed, e.g.:
        # - random_seed: int = 42
        # - max_steps_per_epoch: Optional[int] = None
    """
    epochs: int = 3  # 默认优化 3 轮 (Default to 3 optimization epochs)
    batch_size: int = 8 # 默认批次大小为 8 (Default batch size to 8)
    update_granularity: UpdateGranularity = UpdateGranularity.SECTION_REPHRASE # 默认更新粒度 (Default update granularity)

    validation_dataset: Optional[BaseDataset] = None # 默认无验证集 (No validation set by default)
    early_stopping_patience: Optional[int] = None # 默认不启用早停 (Early stopping disabled by default)

    def __post_init__(self):
        # 添加基本的配置验证 (Add basic configuration validation)
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if self.early_stopping_patience is not None:
            if self.early_stopping_patience <= 0:
                raise ValueError("Early stopping patience must be positive if set.")
            if self.validation_dataset is None:
                raise ValueError("Validation dataset must be provided if early stopping is enabled.")

