# prompt_forge/updaters/base_updater.py

import abc
from typing import Any, Dict, List

# 导入核心基础类型
# Import core base types
from prompt_forge.core.base import (
    BasePromptTemplate,
    MetricResult,
    UpdateGranularity,
)


class BasePromptUpdater(abc.ABC):
    """
    Prompt 更新器的抽象基类。
    负责根据评估结果和指定的更新粒度，提出对 Prompt 模板的修改建议。
    Abstract base class for prompt updaters.
    Responsible for proposing modifications to the prompt template based on
    evaluation results and the specified update granularity.
    """

    @abc.abstractmethod
    def propose_update(
        self,
        current_template: BasePromptTemplate,
        batch_data: List[Dict[str, Any]],
        predictions: List[str],
        detailed_results: Dict[str, MetricResult], # 使用更新后的名称和类型 (Use updated name and type)
        overall_score: float,
        update_level: UpdateGranularity,
        **kwargs,
    ) -> BasePromptTemplate:
        """
        根据评估结果和更新粒度，提出对 Prompt 模板的修改建议。

        Args:
            current_template: 当前的 Prompt 模板实例。
                                (The current prompt template instance.)
            batch_data: 当前批次的数据。
                        (Data for the current batch.)
            predictions: 模型对当前批次的预测输出。
                            (Model predictions for the current batch.)
            detailed_results: 包含所有指标详细结果的字典 (键是指标名，值是 MetricResult)。
                                (Dictionary containing detailed results for all metrics (key is metric name, value is MetricResult).)
            overall_score: 当前批次的综合评估分数。
                            (The overall evaluation score for the current batch.)
            update_level: 控制更新范围和方式的粒度级别。
                            (The granularity level controlling the scope and manner of updates.)
            **kwargs: 其他可能需要的参数 (例如传递给 OptimizerLLM 的额外配置)。
                        (Other potentially necessary arguments, e.g., extra config for an OptimizerLLM.)

        Returns:
            一个**新的** Prompt 模板实例，包含了建议的修改。实现者应确保返回新对象。
            (A **new** prompt template instance containing the proposed modifications. Implementers should ensure a new object is returned.)
        """
        pass
