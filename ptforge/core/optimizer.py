# filename: ptforge/core/optimizer.py
# prompt_forge/core/optimizer.py

import logging
import copy
import time
import dataclasses
from typing import Any, Dict, List, Optional, Tuple

# 导入核心组件和类型
# Import core components and types
# Assuming these relative imports work within your project structure
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
# --- New Imports ---
from ptforge.monitoring.progress_tracker import OptimizationTracker, LiveProgressMonitor
from ptforge.utils.serialization import save_checkpoint, load_checkpoint, SerializationError
# -----------------

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
        # --- Logger Initialization ---
        self.logger = logging.getLogger(self.__class__.__name__)
        # ---------------------------

        self.target_model_client = target_model_client
        self.initial_template = copy.deepcopy(initial_template)
        self.dataset = dataset
        self.evaluator = evaluator
        self.updater = updater
        self.config = config
        self.validation_dataset = config.validation_dataset

        # --- State Initialization ---
        self.current_template: BasePromptTemplate = copy.deepcopy(self.initial_template) # Track current template
        self._best_score: float = -float("inf")
        self._best_template: BasePromptTemplate = copy.deepcopy(self.initial_template)
        self._epochs_no_improve: int = 0
        self.history: List[Dict[str, Any]] = [] # Renamed from _history for checkpoint consistency
        self._current_step = 0 # Track total steps/batches processed
        # --------------------------

        # --- Monitoring Initialization ---
        # Allow passing tracker config via main config's 'tracker_config' attribute
        tracker_config = getattr(self.config, 'tracker_config', {})
        self.tracker = OptimizationTracker(
            experiment_name=tracker_config.get('experiment_name', f"opt_{int(time.time())}"),
            save_dir=tracker_config.get('save_dir', './optimization_logs'),
            save_history=tracker_config.get('save_history', True),
            autosave_interval=tracker_config.get('autosave_interval', 10), # e.g., 10 minutes
            track_memory_usage=tracker_config.get('track_memory_usage', False)
        )
        # Allow enabling monitor via main config's 'use_live_monitor' attribute
        self.monitor = LiveProgressMonitor(self.tracker) if getattr(self.config, 'use_live_monitor', True) else None # Default to True?
        # -------------------------------

        self.logger.info("PromptOptimizer initialized.")
        self.logger.debug(f"Config: {self.config}")
        self.logger.debug(f"Tracker Config: {tracker_config}")


    def optimize(self) -> Tuple[BasePromptTemplate, OptimizationResult]:
        """
        启动并执行 Prompt 优化过程。

        Returns:
            一个元组，包含:
            - best_template: 优化过程中找到的最佳 Prompt 模板。
                             (The best prompt template found during optimization.)
            - final_result: 包含最终结果信息 (如最佳分数、历史记录) 的对象。
                           (An object containing final result information, e.g., best score, history.)
        """
        self.logger.info(f"Starting optimization for {self.config.epochs} epochs...")
        # Use self.current_template which is now an instance variable

        # --- Calculate total steps for monitor ---
        total_steps = 0
        num_batches_per_epoch = 0
        try:
             # Ensure dataset has length and batch_size > 0
             if hasattr(self.dataset, '__len__') and self.config.batch_size > 0:
                  num_batches_per_epoch = (len(self.dataset) + self.config.batch_size - 1) // self.config.batch_size
                  total_steps = self.config.epochs * num_batches_per_epoch
             else:
                  self.logger.warning("Could not determine dataset length or batch size is invalid. Progress bar total might be inaccurate.")
                  total_steps = self.config.epochs # Fallback, less accurate

        except TypeError:
             self.logger.warning("Error determining dataset length. Progress bar total might be inaccurate.")
             total_steps = self.config.epochs # Fallback

        if total_steps <= 0:
            self.logger.warning(f"Calculated total_steps is {total_steps}. Progress bar might not display correctly.")
        # --------------------------------------

        if self.monitor:
            self.monitor.start(total_steps=max(1, total_steps)) # Ensure total is at least 1 for tqdm

        start_time = time.time() # Overall optimization start time
        self._current_step = 0 # Reset step counter at the start of optimization

        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.time()
            self.logger.info(f"--- Starting Epoch {epoch}/{self.config.epochs} ---")

            # --- 训练阶段 (Training Phase) ---
            batch_overall_scores = []

            for batch_idx, batch_data in enumerate(
                self.dataset.get_batches(self.config.batch_size)
            ):
                batch_start_time = time.time()
                if not batch_data:
                    self.logger.warning(f"Epoch {epoch}, Batch {batch_idx+1}/{num_batches_per_epoch if num_batches_per_epoch else '?'}: Skipping empty batch.")
                    continue

                self.logger.debug(f"Epoch {epoch}, Batch {batch_idx+1}/{num_batches_per_epoch if num_batches_per_epoch else '?'}: Processing {len(batch_data)} samples.")

                step_info: Dict[str, Any] = { # Store info for this step/batch
                     "epoch": epoch,
                     "batch": batch_idx + 1,
                     "step": self._current_step,
                     "template_before": self.current_template.render({}, ignore_missing_variables=True), # Snapshot before update
                     "accepted": False, # Default until template potentially changes
                }

                # 1. 渲染 Prompts (Render Prompts)
                try:
                    prompts_batch = [self.current_template.render(data) for data in batch_data]
                    references_batch = [data.get('reference') for data in batch_data]
                except KeyError as e:
                    self.logger.error(f"Missing expected key in data for Epoch {epoch}, Batch {batch_idx+1}: {e}", exc_info=True)
                    step_info["error"] = f"Rendering failed: Missing key {e}"
                    self.tracker.update(step_info) # Log failed step
                    self._current_step += 1
                    if self.monitor: self.monitor.update()
                    continue # Skip batch
                except Exception as e:
                    self.logger.error(f"Error rendering prompts in Epoch {epoch}, Batch {batch_idx+1}: {e}", exc_info=True)
                    step_info["error"] = f"Rendering failed: {e}"
                    self.tracker.update(step_info) # Log failed step
                    self._current_step += 1
                    if self.monitor: self.monitor.update()
                    continue

                # 2. 生成预测 (Generate Predictions)
                try:
                    predictions_batch = self.target_model_client.generate_batch(prompts_batch)
                    if len(predictions_batch) != len(prompts_batch):
                         self.logger.error(f"Epoch {epoch}, Batch {batch_idx+1}: Prediction count mismatch. Expected {len(prompts_batch)}, got {len(predictions_batch)}. Skipping batch.")
                         step_info["error"] = "Prediction count mismatch"
                         self.tracker.update(step_info) # Log failed step
                         self._current_step += 1
                         if self.monitor: self.monitor.update()
                         continue
                    step_info["predictions"] = predictions_batch # Optionally track predictions
                except Exception as e:
                    self.logger.error(f"Error generating predictions in Epoch {epoch}, Batch {batch_idx+1}: {e}", exc_info=True)
                    step_info["error"] = f"Prediction failed: {e}"
                    self.tracker.update(step_info) # Log failed step
                    self._current_step += 1
                    if self.monitor: self.monitor.update()
                    continue

                # 3. 评估结果 (Evaluate Results)
                try:
                    # Evaluate predictions generated with self.current_template *before* update
                    overall_score, detailed_results = self.evaluator.evaluate(predictions_batch, references_batch)
                    batch_overall_scores.append(overall_score)
                    step_info["before_score"] = overall_score # Score *before* potential update
                    # Convert MetricResult objects to dict for JSON serialization in tracker
                    step_info["detailed_results"] = {k: dataclasses.asdict(v) for k, v in detailed_results.items()}

                    batch_duration = time.time() - batch_start_time
                    self.logger.info(f"Epoch {epoch}, Batch {batch_idx+1}/{num_batches_per_epoch if num_batches_per_epoch else '?'}: Score={overall_score:.4f}, Time={batch_duration:.2f}s")
                    # Use logger.debug for potentially verbose detailed results
                    self.logger.debug(f"Detailed results: {step_info['detailed_results']}")

                except Exception as e:
                    self.logger.error(f"Error evaluating results in Epoch {epoch}, Batch {batch_idx+1}: {e}", exc_info=True)
                    overall_score = 0.0 # Treat as 0 if evaluation fails
                    detailed_results = {}
                    step_info["before_score"] = overall_score
                    step_info["error"] = f"Evaluation failed: {e}"
                    # Log the failed step and continue to next batch
                    self.tracker.update(step_info)
                    self._current_step += 1
                    if self.monitor: self.monitor.update()
                    continue

                # 4. 更新 Prompt (Update Prompt)
                update_error = None
                if self.config.update_granularity != UpdateGranularity.FIXED:
                    try:
                        # Store original template reference for comparison
                        original_template_ref = self.current_template
                        proposed_template = self.updater.propose_update(
                            current_template=original_template_ref, # Pass the instance variable
                            batch_data=batch_data,
                            predictions=predictions_batch,
                            detailed_results=detailed_results, # Pass original MetricResult objects here
                            overall_score=overall_score,
                            update_level=self.config.update_granularity,
                        )
                        # Update self.current_template if the updater returned a valid new one
                        if proposed_template and proposed_template is not original_template_ref:
                            # Basic check: Ensure it's still a BasePromptTemplate instance
                            if isinstance(proposed_template, BasePromptTemplate):
                                self.current_template = copy.deepcopy(proposed_template) # Update instance var
                                step_info["accepted"] = True # Mark that template changed
                                step_info["template_after"] = self.current_template.render({}, ignore_missing_variables=True) # Snapshot after update
                                self.logger.debug(f"Epoch {epoch}, Batch {batch_idx+1}: Prompt template updated.")
                            else:
                                update_error = f"Updater returned non-template object: {type(proposed_template)}"
                                self.logger.warning(f"{update_error}. Ignoring update.")
                        elif proposed_template is original_template_ref:
                             self.logger.debug(f"Epoch {epoch}, Batch {batch_idx+1}: Updater returned the same template instance. No update applied.")
                        else:
                             self.logger.debug(f"Epoch {epoch}, Batch {batch_idx+1}: Updater proposed no changes.")

                    except Exception as e:
                        update_error = f"Updater failed: {e}"
                        self.logger.error(f"Error updating prompt in Epoch {epoch}, Batch {batch_idx+1}: {e}", exc_info=True)

                if update_error:
                     step_info["error"] = update_error

                # Add after_score (which is the same as before_score in this optimizer style,
                # as evaluation happens *before* the update modifies the template for the *next* batch)
                step_info["after_score"] = overall_score # Score reflects state *before* this batch's update

                # --- Update Tracker and Monitor ---
                self.tracker.update(step_info)
                self._current_step += 1
                if self.monitor:
                    # Pass current overall score to monitor description maybe?
                    best_score_display = f"{self._best_score:.4f}" if self._best_score > -float("inf") else "N/A"
                    # Update monitor description more dynamically
                    desc = f"Ep {epoch} B {batch_idx+1} Score {overall_score:.3f}|Best {best_score_display}"
                    if step_info["accepted"]:
                         desc += " (Updated)"
                    self.monitor.progress_bar.set_description(desc)
                    self.monitor.update() # Updates the tqdm bar itself (advances by 1 implicitly if bar exists)
                # ----------------------------------

            # --- Epoch 结束处理 (End of Epoch Processing) ---
            epoch_duration = time.time() - epoch_start_time
            avg_training_score = sum(batch_overall_scores) / len(batch_overall_scores) if batch_overall_scores else 0.0
            self.logger.info(f"--- Epoch {epoch} Summary ---")
            self.logger.info(f"Average Training Score: {avg_training_score:.4f}")
            self.logger.info(f"Epoch Duration: {epoch_duration:.2f}s")

            current_score_for_comparison = avg_training_score
            validation_score = None
            validation_details_dict = None

            # --- 验证阶段 (Validation Phase, if applicable) ---
            if self.validation_dataset:
                self.logger.info(f"Running validation for Epoch {epoch}...")
                # Evaluate the *current* template after the epoch's updates
                val_score, val_details = self._evaluate_on_validation_set(self.current_template)
                self.logger.info(f"Validation Score: {val_score:.4f}")
                self.logger.debug(f"Validation Details: {val_details}")
                validation_score = val_score
                validation_details_dict = {k: dataclasses.asdict(v) for k, v in val_details.items()} # Convert for JSON
                current_score_for_comparison = validation_score # Use validation score for best check

            # --- 更新最佳结果 (Update Best Result) ---
            if current_score_for_comparison > self._best_score:
                self._best_score = current_score_for_comparison
                # Save a *copy* of the current_template when it becomes the best
                self._best_template = copy.deepcopy(self.current_template)
                self._epochs_no_improve = 0
                self.logger.info(f"*** New best score found: {self._best_score:.4f} (Epoch {epoch}) ***")
            else:
                self._epochs_no_improve += 1
                self.logger.info(f"Score did not improve. Best score remains {self._best_score:.4f}. Epochs without improvement: {self._epochs_no_improve}")

            # --- 记录历史 (Record Epoch History - distinct from step history in tracker) ---
            epoch_log = {
                "epoch": epoch,
                "avg_training_score": avg_training_score,
                "validation_score": validation_score,
                "best_score_so_far": self._best_score,
                "duration_seconds": epoch_duration,
                "validation_details": validation_details_dict, # Use the converted dict
                "template_hash": hash(self.current_template.render({}, ignore_missing_variables=True)), # Basic template identifier
            }
            self.history.append(epoch_log) # Append to the main history list

            # --- 检查早停 (Check Early Stopping) ---
            if (
                self.config.early_stopping_patience is not None
                and self.validation_dataset is not None # Only stop based on validation score
                and self._epochs_no_improve >= self.config.early_stopping_patience
            ):
                self.logger.warning(
                    f"Early stopping triggered after {epoch} epochs due to no validation score improvement "
                    f"for {self.config.early_stopping_patience} consecutive epochs."
                )
                break # Exit epoch loop

        # --- Finalization ---
        if self.monitor:
            self.monitor.stop()

        optimization_duration = time.time() - start_time
        self.logger.info("--- Optimization Finished ---")
        self.logger.info(f"Total Duration: {optimization_duration:.2f}s")
        self.logger.info(f"Completed {self._current_step} steps across {epoch} epochs.")
        self.logger.info(f"Best Score Achieved: {self._best_score:.4f}")


        # --- Save final tracker state ---
        try:
            final_summary_path = self.tracker.save()
            self.logger.info(f"Final monitoring data saved. Summary: {final_summary_path}")
        except Exception as e:
             self.logger.error(f"Failed to save final monitoring data: {e}", exc_info=True)


        # --- Optionally generate report/plots ---
        if getattr(self.config, 'generate_report_at_end', False):
            try:
                 report_path = self.tracker.generate_report()
                 if report_path:
                      self.logger.info(f"Generated final optimization report: {report_path}")
                 # Optionally also generate plots
                 # self.tracker.plot_progress()
                 # self.tracker.plot_action_statistics() # Less relevant here? Updater actions aren't tracked directly
            except Exception as e:
                 self.logger.error(f"Failed to generate final report/plots: {e}", exc_info=True)
        # ------------------------------------

        # Prepare final result object
        final_result = OptimizationResult(
            best_score=self._best_score,
            best_template=self._best_template, # Return the best template copy
            history=self.history # Return epoch history
        )

        # Return the best template found and the result object
        return self._best_template, final_result


    def _evaluate_on_validation_set(
        self, template: BasePromptTemplate
    ) -> Tuple[float, Dict[str, MetricResult]]:
        """
        在验证集上评估给定的 Prompt 模板。
        Evaluates the given prompt template on the validation dataset.
        """
        if not self.validation_dataset:
            # This case should ideally not be reached if called correctly
            self.logger.warning("Attempted to evaluate on validation set, but none was provided.")
            return -float("inf"), {}

        all_predictions = []
        all_references = []
        self.logger.info(f"Starting validation evaluation...") # Removed dataset length log for safety
        val_start_time = time.time()

        num_val_batches = '?'
        try:
            # Calculate num batches for validation if possible
            if hasattr(self.validation_dataset, '__len__') and self.config.batch_size > 0:
                num_val_batches = (len(self.validation_dataset) + self.config.batch_size - 1) // self.config.batch_size
        except TypeError:
            self.logger.warning("Could not determine validation dataset length.")

        batch_generator = self.validation_dataset.get_batches(self.config.batch_size)

        for batch_idx, batch_data in enumerate(batch_generator):
            if not batch_data: continue
            self.logger.debug(f"Validation Batch {batch_idx+1}/{num_val_batches}")
            try:
                prompts = [template.render(data) for data in batch_data]
                refs = [data.get('reference') for data in batch_data]
                preds = self.target_model_client.generate_batch(prompts)

                if len(preds) != len(prompts):
                     self.logger.error(f"Validation: Prediction count mismatch in Batch {batch_idx+1}. Expected {len(prompts)}, got {len(preds)}. Skipping.")
                     continue

                all_predictions.extend(preds)
                all_references.extend(refs)
            except KeyError as e:
                 self.logger.error(f"Missing expected key in validation data for Batch {batch_idx+1}: {e}", exc_info=True)
                 continue # Skip batch
            except Exception as e:
                self.logger.error(f"Error during validation batch {batch_idx+1} processing: {e}", exc_info=True)
                # Maybe skip batch instead of failing whole validation? For now, continue.

        val_duration = time.time() - val_start_time
        self.logger.info(f"Validation data processing finished in {val_duration:.2f}s.")

        if not all_predictions:
             self.logger.warning("Validation resulted in no predictions after processing all batches. Returning zero score.")
             return 0.0, {} # Return empty dict for details

        try:
            self.logger.info(f"Evaluating {len(all_predictions)} validation predictions...")
            # Ensure evaluator handles potential empty lists gracefully if all batches failed
            overall_score, detailed_results = self.evaluator.evaluate(all_predictions, all_references)
            self.logger.info(f"Validation evaluation completed. Score: {overall_score:.4f}")
            return overall_score, detailed_results
        except Exception as e:
            self.logger.error(f"Error during final validation evaluation: {e}", exc_info=True)
            return 0.0, {} # Return empty dict for details

    # --- Checkpointing Methods ---
    def save_optimizer_checkpoint(self, checkpoint_path: str):
        """Saves the current state of the optimizer to a checkpoint file using pickle."""
        # Note: Checkpoint functions expect specific attributes like _best_template, _best_score, current_template, history
        # Ensure these are correctly maintained as instance variables.
        # RLOptimizer specific attributes (action_memory, exploration_context) won't be saved here.
        try:
            # We pass `self` and the function will try to pickle relevant attributes.
            # Currently serialization.py save_checkpoint doesn't explicitly save optimizer steps/epoch state.
            # It saves: _best_template, _best_score, current_template, exploration_context(if exists), action_memory(if exists), history(if exists)
            # Add current step to the state saved? Need to modify save_checkpoint or handle it manually.
            # For now, rely on the existing save_checkpoint behavior.
            save_checkpoint(
                optimizer=self,
                checkpoint_path=checkpoint_path,
                include_memory=False, # PromptOptimizer doesn't have action_memory
                include_history=True # We have self.history (epoch history)
            )
            self.logger.info(f"Optimizer checkpoint successfully saved to {checkpoint_path}")
        except SerializationError as e:
            self.logger.error(f"Failed to save optimizer checkpoint: {e}")
        except AttributeError as e:
            # This might happen if expected attributes (_best_template, current_template etc.) are missing
            self.logger.error(f"Failed to save checkpoint due to missing attribute: {e}. Check optimizer state structure.", exc_info=True)
        except Exception as e:
             self.logger.error(f"An unexpected error occurred saving checkpoint: {e}", exc_info=True)


    def load_optimizer_checkpoint(self, checkpoint_path: str):
        """Loads the optimizer state from a checkpoint file."""
        try:
            # This will attempt to restore attributes onto `self` based on what's in the pickle file
            load_checkpoint(checkpoint_path=checkpoint_path, optimizer=self)
            self.logger.info(f"Optimizer checkpoint successfully loaded from {checkpoint_path}")

            # Post-load actions:
            # - Reset tracker/monitor? The tracker state isn't saved in the checkpoint.
            # - Reset epochs_no_improve? Should be recalculated based on loaded history if possible.
            # - Need to restore _current_step if we want accurate progress bar resuming.
            #   This needs `save_checkpoint` to save it or we infer it (e.g. from tracker history if saved separately)

            # Simple approach: Log warning and reset internal counters/tracker state.
            self.logger.warning("Optimizer state loaded. Monitoring/progress tracking will restart.")
            self._epochs_no_improve = 0 # Reset counter
            self._current_step = 0 # Reset step counter, progress bar will restart
            # Re-initialize tracker? Or just let it continue logging to same dir? Let it continue.

            # Ensure mandatory attributes exist after loading
            if not hasattr(self, 'current_template'):
                 self.logger.warning("Loaded checkpoint missing 'current_template'. Using initial_template.")
                 self.current_template = copy.deepcopy(self.initial_template)
            if not hasattr(self, '_best_template'):
                 self.logger.warning("Loaded checkpoint missing '_best_template'. Using initial_template.")
                 self._best_template = copy.deepcopy(self.initial_template)
            if not hasattr(self, '_best_score'):
                 self.logger.warning("Loaded checkpoint missing '_best_score'. Setting to -inf.")
                 self._best_score = -float('inf')
            if not hasattr(self, 'history'):
                 self.logger.warning("Loaded checkpoint missing 'history'. Initializing empty list.")
                 self.history = []


        except SerializationError as e:
            self.logger.error(f"Failed to load optimizer checkpoint: {e}")
            raise # Re-raise so caller knows loading failed
        except FileNotFoundError:
            self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
            raise # Re-raise
        except Exception as e:
             self.logger.error(f"An unexpected error occurred loading checkpoint: {e}", exc_info=True)
             raise # Re-raise
    # ---------------------------