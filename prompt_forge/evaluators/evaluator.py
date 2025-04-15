# prompt_forge/evaluators/evaluator.py

import logging
from typing import Any, Dict, List, Tuple

# 导入 BaseMetric 和 MetricResult
# Import BaseMetric and MetricResult
from prompt_forge.core.base import BaseMetric, MetricResult

# 设置日志记录器 (Set up logger)
logger = logging.getLogger(__name__)

class Evaluator:
    """
    负责执行多个评估指标并计算加权综合得分及返回详细结果。
    Responsible for executing multiple evaluation metrics, calculating a weighted overall score,
    and returning detailed results.
    """

    def __init__(self, metrics: List[Tuple[BaseMetric, float]]):
        """
        初始化评估器。

        Args:
            metrics: 一个包含 (metric_instance, weight) 元组的列表。
                     权重必须为正数。权重的总和不强制要求为 1。
                     (A list of tuples, each containing (metric_instance, weight).
                     Weights must be positive. The sum of weights is not enforced to be 1.)

        Raises:
            ValueError: 如果指标列表为空，或者任何权重不为正数。
                        (If the metrics list is empty, or if any weight is not positive.)
            TypeError: 如果传入的 metric 不是 BaseMetric 的实例。
                       (If a provided metric is not an instance of BaseMetric.)
        """
        if not metrics:
            raise ValueError("Metrics list cannot be empty.")

        validated_metrics = []
        total_weight = 0.0
        for metric, weight in metrics:
            if not isinstance(metric, BaseMetric):
                raise TypeError(f"Object {metric} is not an instance of BaseMetric.")
            if not isinstance(weight, (int, float)) or weight <= 0:
                raise ValueError(
                    f"Weight for metric '{metric.name}' must be a positive number, got {weight}."
                )
            validated_metrics.append((metric, float(weight)))
            total_weight += float(weight)

        self.metrics = validated_metrics
        self.total_weight = total_weight
        logger.info(
            f"Evaluator initialized with {len(self.metrics)} metrics. Total weight: {self.total_weight:.2f}"
        )

    def evaluate(
        self, predictions: List[str], references: List[Any]
    ) -> Tuple[float, Dict[str, MetricResult]]: # <--- 返回类型修改
        """
        对给定的预测和参考执行所有配置的指标评估。

        Args:
            predictions: 模型生成的预测输出列表。 (List of predicted outputs from the model.)
            references: 对应的参考输出列表。 (List of corresponding reference outputs.)

        Returns:
            一个元组，包含:
            - overall_score: 所有成功计算指标得分的加权平均值。
                             (The weighted average score across all successfully computed metrics.)
            - detailed_results: 一个字典，键是指标名称，值是对应的 MetricResult 对象，
                                包含分数、描述和可选的详细信息。
                                (A dictionary where keys are metric names and values are the
                                corresponding MetricResult objects, containing score, description, and optional details.)

        Raises:
            ValueError: 如果 predictions 和 references 的长度不匹配。
                        (If the lengths of predictions and references do not match.)
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: predictions ({len(predictions)}) and references ({len(references)})"
            )

        # 重命名 internal_scores 为 detailed_results
        # Rename internal_scores to detailed_results
        detailed_results: Dict[str, MetricResult] = {}
        weighted_score_sum = 0.0
        valid_weights_sum = 0.0 # 成功计算的指标的权重总和 (Sum of weights for successfully computed metrics)

        if not predictions:
            logger.warning("Received empty predictions list. Returning zero scores and empty details.")
            # 为每个指标创建一个空的 MetricResult
            # Create an empty MetricResult for each metric
            for metric, _ in self.metrics:
                 detailed_results[metric.name] = MetricResult(
                     name=metric.name,
                     score=0.0,
                     description=metric.__doc__ or "N/A", # 获取 docstring
                     details={"info": "Input predictions list was empty."}
                 )
            return 0.0, detailed_results


        for metric, weight in self.metrics:
            metric_name = metric.name
            metric_description = metric.__doc__ # 获取 docstring

            try:
                # 调用 metric.compute() 并期望返回 MetricResult 对象
                # Call metric.compute() expecting a MetricResult object
                result: MetricResult = metric.compute(predictions, references)

                if not isinstance(result, MetricResult):
                     logger.error(f"Metric '{metric_name}' compute() did not return a MetricResult object, got {type(result)}. Skipping.")
                     # 可以创建一个表示错误的 MetricResult
                     # Can create an error MetricResult
                     detailed_results[metric_name] = MetricResult(
                         name=metric_name,
                         score=0.0, # Assign 0 score on type error
                         description=metric_description,
                         details={"error": f"Invalid return type: {type(result)}"}
                     )
                     continue # 跳过这个指标 (Skip this metric)

                # 补充描述信息 (如果指标实现未提供)
                # Add description if not provided by the metric implementation
                if result.description is None:
                    result.description = metric_description or "No description provided."

                detailed_results[metric_name] = result
                weighted_score_sum += result.score * weight
                valid_weights_sum += weight
                logger.debug(f"Metric '{metric_name}' computed score: {result.score:.4f}")

            except Exception as e:
                logger.error(
                    f"Error computing metric '{metric_name}': {e}", exc_info=True
                )
                # 创建一个表示错误的 MetricResult
                # Create an error MetricResult
                detailed_results[metric_name] = MetricResult(
                    name=metric_name,
                    score=0.0, # Assign 0 score on error
                    description=metric_description,
                    details={"error": str(e)}
                )
                # 不将权重计入 valid_weights_sum (Do not add weight to valid_weights_sum if metric failed)


        # 基于成功计算的指标计算总分
        # Calculate overall score based on successfully computed metrics
        if valid_weights_sum > 0:
             overall_score = weighted_score_sum / valid_weights_sum
        else:
             logger.error("No metrics computed successfully or total valid weight is zero. Returning overall score 0.")
             overall_score = 0.0


        logger.info(f"Evaluation completed. Overall score: {overall_score:.4f}")
        # 返回总分和详细结果字典
        # Return overall score and the dictionary of detailed results
        return overall_score, detailed_results

