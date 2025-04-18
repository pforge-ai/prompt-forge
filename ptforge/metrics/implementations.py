# prompt_forge/metrics/implementations.py

import logging
from typing import Any, Dict, List, Optional

# 导入基类和结果类
# Import base class and result class
from ptforge.core.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class ExactMatchAccuracy(BaseMetric):
    """
    计算精确匹配准确率。
    比较预测字符串与参考字符串是否完全相同。

    Calculates exact match accuracy.
    Compares if the prediction string is exactly identical to the reference string.
    """

    def __init__(self, case_sensitive: bool = True):
        """
        初始化精确匹配准确率指标。

        Args:
            case_sensitive: 是否区分大小写进行比较。默认为 True。
                            (Whether the comparison should be case-sensitive. Defaults to True.)
        """
        self._case_sensitive = case_sensitive
        logger.debug(f"ExactMatchAccuracy initialized (case_sensitive={self._case_sensitive})")

    @property
    def name(self) -> str:
        """指标名称 (Name of the metric)"""
        return f"ExactMatchAccuracy(case_sensitive={self._case_sensitive})"

    def compute(
        self, predictions: List[str], references: List[Any]
    ) -> MetricResult:
        """
        计算精确匹配准确率分数和详细信息。

        Args:
            predictions: 模型生成的预测输出字符串列表。
                            (List of predicted output strings generated by the model.)
            references: 对应的参考输出字符串列表。如果元素不是字符串，会尝试转换。
                        (List of corresponding reference output strings. Non-string elements will be attempted to convert.)

        Returns:
            一个 MetricResult 对象，包含分数、描述和匹配计数等详细信息。
            (A MetricResult object containing the score, description, and details like match counts.)
        """
        if len(predictions) != len(references):
                # 通常 Evaluator 会先检查，但这里也加上以防万一
                # Usually checked by Evaluator first, but added here just in case
            raise ValueError("Predictions and references lists must have the same length.")

        if not predictions:
            return MetricResult(
                name=self.name,
                score=0.0, # 或者 1.0? 定义空列表的准确率可能需要讨论，0.0 比较安全
                description=self.__doc__,
                details={"match_count": 0, "total_count": 0, "info": "Empty prediction list."}
            )

        match_count = 0
        processed_references = []
        type_conversion_warnings = 0

        # 预处理参考，确保是字符串
        # Preprocess references to ensure they are strings
        for ref in references:
            if isinstance(ref, str):
                processed_references.append(ref)
            else:
                try:
                    processed_references.append(str(ref))
                    if type_conversion_warnings < 5: # Limit warnings
                            logger.warning(f"Reference item '{ref}' (type: {type(ref)}) was converted to string for comparison.")
                    type_conversion_warnings += 1
                except Exception as e:
                        logger.error(f"Could not convert reference item '{ref}' to string: {e}. Skipping comparison for this item.")
                        processed_references.append(None) # Mark as None to skip comparison

        if type_conversion_warnings >= 5:
                logger.warning(f"Total {type_conversion_warnings} reference items converted to string.")


        # 进行比较 (Perform comparison)
        for i, pred in enumerate(predictions):
            ref = processed_references[i]
            if ref is None: # Skip if reference conversion failed
                continue

            # 确保 pred 也是字符串 (Ensure pred is also a string)
            if not isinstance(pred, str):
                    logger.warning(f"Prediction item '{pred}' (type: {type(pred)}) at index {i} is not a string. Treating as mismatch.")
                    continue # Treat non-string prediction as mismatch

            pred_to_compare = pred if self._case_sensitive else pred.lower()
            ref_to_compare = ref if self._case_sensitive else ref.lower()

            if pred_to_compare == ref_to_compare:
                match_count += 1

        total_count = len(predictions) # 总数基于预测列表长度 (Total count based on predictions list length)
        accuracy = (match_count / total_count) if total_count > 0 else 0.0

        details = {
            "match_count": match_count,
            "total_count": total_count,
            "case_sensitive": self._case_sensitive,
        }
        if type_conversion_warnings > 0:
            details["reference_conversion_warnings"] = type_conversion_warnings

        logger.info(f"{self.name} computed: {accuracy:.4f} ({match_count}/{total_count} matches)")

        return MetricResult(
            name=self.name,
            score=accuracy,
            description=self.__doc__, # 使用类的 docstring 作为描述 (Use class docstring as description)
            details=details
        )

