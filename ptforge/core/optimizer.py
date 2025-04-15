# prompt_forge/core/optimizer.py

import logging
import copy
import time
import dataclasses
from typing import Any, Dict, List, Optional, Tuple

# 导入核心组件和类型
# Import core components and types
from .base import (
    BaseDataset,
    BaseLLMClient,
    # BasePromptTemplate is imported below after try-except
    MetricResult,
    UpdateGranularity,
)

from ptforge.templates.base_template import BasePromptTemplate

from .config import OptimizationConfig
from ptforge.evaluators.evaluator import Evaluator
from ptforge.updaters.base_updater import BasePromptUpdater

# 设置日志记录器
# Set up logger
logger = logging.getLogger(__name__)
# Basic logging configuration (can be customized by the user later)
# Avoid configuring root logger directly here if this is a library
# Let the user configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


# 重命名 OptimizationState 为 OptimizationResult
# Rename OptimizationState to OptimizationResult
@dataclasses.dataclass
class OptimizationResult:
    """持有优化过程最终结果的数据类。"""
    best_score: float
    best_template: BasePromptTemplate
    history: List[Dict[str, Any]] # 记录每轮或关键步骤的信息 (Record info per epoch or key steps)


# 重命名 PromptForgeTrainer 为 PromptOptimizer
# Rename PromptForgeTrainer to PromptOptimizer
class PromptOptimizer:
    """
    核心优化器类，负责协调整个 Prompt 优化过程。
    The core optimizer class responsible for orchestrating the entire prompt optimization process.
    """

    def __init__(
        self,
        target_model_client: BaseLLMClient,
        initial_template: BasePromptTemplate,
        dataset: BaseDataset,
        evaluator: Evaluator,
        updater: BasePromptUpdater,
        config: OptimizationConfig,
    ):
        """
        初始化 PromptOptimizer。

        Args:
            target_model_client: 用于生成响应的目标 LLM 客户端实例。
                                 (The target LLM client instance used for generating responses.)
            initial_template: 优化的起始 Prompt 模板实例。
                              (The initial prompt template instance to start optimization from.)
            dataset: 用于训练 (和可选验证) 的数据集实例。
                     (The dataset instance for training (and optional validation).)
            evaluator: 用于评估预测结果的评估器实例。
                       (The evaluator instance for assessing predictions.)
            updater: 用于提出 Prompt 修改建议的更新器实例。
                     (The prompt updater instance for proposing modifications.)
            config: 包含优化过程超参数的配置对象。
                    (The configuration object containing hyperparameters for the optimization process.)
        """
        self.target_model_client = target_model_client
        self.initial_template = copy.deepcopy(initial_template)
        self.dataset = dataset
        self.evaluator = evaluator
        self.updater = updater
        self.config = config
        self.validation_dataset = config.validation_dataset

        self._best_score: float = -float("inf")
        self._best_template: BasePromptTemplate = copy.deepcopy(self.initial_template)
        self._epochs_no_improve: int = 0
        self._history: List[Dict[str, Any]] = []

        logger.info("PromptOptimizer initialized.")

    def optimize(self) -> Tuple[BasePromptTemplate, OptimizationResult]: # <--- 返回类型更新 (Return type updated)
        """
        启动并执行 Prompt 优化过程。

        Returns:
            一个元组，包含:
            - best_template: 优化过程中找到的最佳 Prompt 模板。
                             (The best prompt template found during optimization.)
            - final_result: 包含最终结果信息 (如最佳分数、历史记录) 的对象。
                           (An object containing final result information, e.g., best score, history.)
        """
        logger.info(f"Starting optimization for {self.config.epochs} epochs...")
        current_template = copy.deepcopy(self.initial_template)

        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.time()
            logger.info(f"--- Starting Epoch {epoch}/{self.config.epochs} ---")

            # --- 训练阶段 (Training Phase) ---
            batch_overall_scores = []
            num_batches = (len(self.dataset) + self.config.batch_size - 1) // self.config.batch_size
            for batch_idx, batch_data in enumerate(
                self.dataset.get_batches(self.config.batch_size)
            ):
                batch_start_time = time.time()
                if not batch_data:
                    logger.warning(f"Epoch {epoch}, Batch {batch_idx+1}/{num_batches}: Skipping empty batch.")
                    continue

                logger.debug(f"Epoch {epoch}, Batch {batch_idx+1}/{num_batches}: Processing {len(batch_data)} samples.")

                # 1. 渲染 Prompts (Render Prompts)
                try:
                    # --- 修正这里：传递整个 data 字典 ---
                    # --- Fix here: Pass the entire data dictionary ---
                    prompts_batch = [current_template.render(data) for data in batch_data]
                    # -----------------------------------------
                    # Handle potential missing 'reference' key gracefully
                    references_batch = [data.get('reference') for data in batch_data]
                except KeyError as e:
                    # This error might now happen inside render if data dict doesn't have expected keys
                    logger.error(f"Missing expected key in data for Epoch {epoch}, Batch {batch_idx+1}: {e}", exc_info=True)
                    continue # Skip batch
                except Exception as e:
                    logger.error(f"Error rendering prompts in Epoch {epoch}, Batch {batch_idx+1}: {e}", exc_info=True)
                    continue

                # 2. 生成预测 (Generate Predictions)
                try:
                    predictions_batch = self.target_model_client.generate_batch(prompts_batch)
                    if len(predictions_batch) != len(prompts_batch):
                         logger.error(f"Epoch {epoch}, Batch {batch_idx+1}: Prediction count mismatch. Expected {len(prompts_batch)}, got {len(predictions_batch)}. Skipping batch.")
                         continue
                except Exception as e:
                    logger.error(f"Error generating predictions in Epoch {epoch}, Batch {batch_idx+1}: {e}", exc_info=True)
                    continue

                # 3. 评估结果 (Evaluate Results)
                try:
                    if any(ref is None for ref in references_batch) and len(references_batch) == len(predictions_batch):
                         logger.debug(f"Epoch {epoch}, Batch {batch_idx+1}: Some references are None.")

                    overall_score, detailed_results = self.evaluator.evaluate(predictions_batch, references_batch)
                    batch_overall_scores.append(overall_score)
                    batch_duration = time.time() - batch_start_time
                    logger.info(f"Epoch {epoch}, Batch {batch_idx+1}/{num_batches}: Score={overall_score:.4f}, Time={batch_duration:.2f}s")
                    logger.debug(f"Detailed results: {detailed_results}")

                except Exception as e:
                    logger.error(f"Error evaluating results in Epoch {epoch}, Batch {batch_idx+1}: {e}", exc_info=True)
                    overall_score = 0.0
                    detailed_results = {}
                    continue

                # 4. 更新 Prompt (Update Prompt)
                if self.config.update_granularity != UpdateGranularity.FIXED:
                    try:
                        proposed_template = self.updater.propose_update(
                            current_template=current_template,
                            batch_data=batch_data,
                            predictions=predictions_batch,
                            detailed_results=detailed_results,
                            overall_score=overall_score,
                            update_level=self.config.update_granularity,
                        )
                        if proposed_template is current_template:
                             logger.warning(f"Updater {type(self.updater).__name__} did not return a new template instance. Reusing the current one.")
                        current_template = copy.deepcopy(proposed_template)
                        logger.debug(f"Epoch {epoch}, Batch {batch_idx+1}: Prompt template updated.")

                    except Exception as e:
                        logger.error(f"Error updating prompt in Epoch {epoch}, Batch {batch_idx+1}: {e}", exc_info=True)


            # --- Epoch 结束处理 (End of Epoch Processing) ---
            epoch_duration = time.time() - epoch_start_time
            avg_training_score = sum(batch_overall_scores) / len(batch_overall_scores) if batch_overall_scores else 0.0
            logger.info(f"--- Epoch {epoch} Summary ---")
            logger.info(f"Average Training Score: {avg_training_score:.4f}")
            logger.info(f"Epoch Duration: {epoch_duration:.2f}s")

            current_score_for_comparison = avg_training_score
            validation_score = None
            validation_details = None

            # --- 验证阶段 (Validation Phase, if applicable) ---
            if self.validation_dataset:
                logger.info(f"Running validation for Epoch {epoch}...")
                val_score, val_details = self._evaluate_on_validation_set(current_template)
                logger.info(f"Validation Score: {val_score:.4f}")
                logger.debug(f"Validation Details: {val_details}")
                validation_score = val_score
                current_score_for_comparison = validation_score

            # --- 更新最佳结果 (Update Best Result) ---
            if current_score_for_comparison > self._best_score:
                self._best_score = current_score_for_comparison
                self._best_template = copy.deepcopy(current_template)
                self._epochs_no_improve = 0
                logger.info(f"*** New best score found: {self._best_score:.4f} (Epoch {epoch}) ***")
            else:
                self._epochs_no_improve += 1
                logger.info(f"Score did not improve. Best score remains {self._best_score:.4f}. Epochs without improvement: {self._epochs_no_improve}")

            # --- 记录历史 (Record History) ---
            epoch_log = {
                "epoch": epoch,
                "avg_training_score": avg_training_score,
                "validation_score": validation_score,
                "best_score_so_far": self._best_score,
                "duration_seconds": epoch_duration,
                "validation_details": val_details if self.validation_dataset else None,
            }
            self._history.append(epoch_log)

            # --- 检查早停 (Check Early Stopping) ---
            if (
                self.config.early_stopping_patience is not None
                and self.validation_dataset is not None
                and self._epochs_no_improve >= self.config.early_stopping_patience
            ):
                logger.warning(
                    f"Early stopping triggered after {epoch} epochs due to no validation score improvement "
                    f"for {self.config.early_stopping_patience} consecutive epochs."
                )
                break

        logger.info("--- Optimization Finished ---")
        logger.info(f"Best Score Achieved: {self._best_score:.4f}")

        final_result = OptimizationResult(
            best_score=self._best_score,
            best_template=self._best_template,
            history=self._history
        )

        return self._best_template, final_result


    def _evaluate_on_validation_set(
        self, template: BasePromptTemplate
    ) -> Tuple[float, Dict[str, MetricResult]]:
        """
        在验证集上评估给定的 Prompt 模板。
        Evaluates the given prompt template on the validation dataset.
        """
        if not self.validation_dataset:
            logger.error("Attempted to evaluate on validation set, but none was provided.")
            return -float("inf"), {}

        all_predictions = []
        all_references = []
        logger.info(f"Starting validation evaluation on {len(self.validation_dataset)} samples...")
        val_start_time = time.time()
        num_val_batches = (len(self.validation_dataset) + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx, batch_data in enumerate(self.validation_dataset.get_batches(self.config.batch_size)):
            if not batch_data: continue
            logger.debug(f"Validation Batch {batch_idx+1}/{num_val_batches}")
            try:
                # --- 修正这里：传递整个 data 字典 ---
                # --- Fix here: Pass the entire data dictionary ---
                prompts = [template.render(data) for data in batch_data]
                # -----------------------------------------
                refs = [data.get('reference') for data in batch_data]
                preds = self.target_model_client.generate_batch(prompts)

                if len(preds) != len(prompts):
                     logger.error(f"Validation: Prediction count mismatch in Batch {batch_idx+1}. Expected {len(prompts)}, got {len(preds)}. Skipping.")
                     continue

                all_predictions.extend(preds)
                all_references.extend(refs)
            except KeyError as e:
                 logger.error(f"Missing expected key in validation data for Batch {batch_idx+1}: {e}", exc_info=True)
                 continue
            except Exception as e:
                logger.error(f"Error during validation batch {batch_idx+1} processing: {e}", exc_info=True)

        val_duration = time.time() - val_start_time
        logger.info(f"Validation data processing finished in {val_duration:.2f}s.")

        if not all_predictions:
             logger.warning("Validation resulted in no predictions after processing all batches. Returning zero score.")
             return 0.0, {}

        try:
            logger.info(f"Evaluating {len(all_predictions)} validation predictions...")
            overall_score, detailed_results = self.evaluator.evaluate(all_predictions, all_references)
            logger.info(f"Validation evaluation completed. Score: {overall_score:.4f}")
            return overall_score, detailed_results
        except Exception as e:
            logger.error(f"Error during final validation evaluation: {e}", exc_info=True)
            return 0.0, {}

