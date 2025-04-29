# filename: ptforge/core/rl_optimizer.py
# ptforge/core/rl_optimizer.py
"""
基于强化学习思想的提示词优化器。
通过记忆动作-奖励，不断探索和利用优化空间，学习有效的提示词修改策略。

Prompt optimizer based on reinforcement learning ideas.
Learns effective prompt modification strategies by memorizing action-rewards, exploring and exploiting the optimization space.
"""

import copy
import math
import random
import logging
import time
import dataclasses # <-- Added import
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, Set, TypeVar

# --- Standard Imports ---
from ptforge.core.base import BaseLLMClient, BasePromptTemplate, MetricResult
from ptforge.templates.base_template import BasePromptTemplate
from ptforge.templates.action_space import StructuredAction, ActionType, ActionSpace
from ptforge.core.action_memory import ActionMemory, ActionRecord
from ptforge.core.action_executor import ActionExecutor, BatchActionExecutor
from ptforge.core.reward_calculator import RewardCalculator, create_default_reward_calculator
from ptforge.utils.json_helper import parse_json_robust, extract_action_json
from ptforge.core.config import OptimizationConfig
# --- New Imports ---
from ptforge.monitoring.progress_tracker import OptimizationTracker, LiveProgressMonitor
from ptforge.utils.serialization import save_checkpoint, load_checkpoint, SerializationError
# -----------------

# logger = logging.getLogger(__name__) # Initialize logger inside class

# 定义类型变量
# Define type variables
T_Template = TypeVar('T_Template', bound=BasePromptTemplate)

# Use dataclass for RLOptimizationResult for better structure
@dataclasses.dataclass
class RLOptimizationResult:
    """
    强化学习优化的结果。
    包含最终模板、历史记录和统计信息。

    Results of RL optimization.
    Contains final template, history, and statistics.
    """
    # Fields without defaults first
    best_template: BasePromptTemplate
    best_score: float
    start_score: float
    history: List[Dict[str, Any]] # This should store step history from RL loop
    action_memory: ActionMemory # Moved before fields with defaults
    optimization_time: float
    # Fields with defaults last
    tracker_history: Optional[List[Dict[str, Any]]] = None # History from OptimizationTracker
    tracker_summary: Optional[Dict[str, Any]] = None # Summary from OptimizationTracker

    def get_improvement(self) -> float:
        """
        获取相对于起始分数的改进百分比。

        Get improvement percentage relative to start score.

        Returns:
            改进百分比 (Improvement percentage)
        """
        if abs(self.start_score) < 1e-10:
            # Handle division by zero or near-zero
            return 100.0 * math.copysign(1, self.best_score) if self.best_score != 0 else 0.0

        return (self.best_score - self.start_score) / abs(self.start_score) * 100

    def get_summary(self) -> Dict[str, Any]:
        """
        获取优化结果的概要信息。

        Get summary of optimization results.

        Returns:
            包含概要信息的字典 (Dictionary containing summary information)
        """
        num_steps = len(self.history) if self.history else (self.tracker_summary['steps'] if self.tracker_summary else 0)
        return {
            "best_score": self.best_score,
            "start_score": self.start_score,
            "improvement_percent": self.get_improvement(),
            "total_steps": num_steps,
            "optimization_time_seconds": self.optimization_time,
            "action_stats": self.action_memory.get_statistics(),
            "tracker_summary": self.tracker_summary or "Not Available"
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        将优化结果转换为可序列化的字典。

        Convert optimization result to serializable dictionary.

        Returns:
            可序列化的字典 (Serializable dictionary)
        """
        # Note: best_template is not directly serializable to JSON.
        # Consider saving template content separately if needed.
        return {
            "best_score": self.best_score,
            "start_score": self.start_score,
            "history": self.history, # Step history
            "tracker_history": self.tracker_history, # Tracker history
            "action_memory": self.action_memory.to_dict_list(),
            "optimization_time": self.optimization_time,
            "summary": self.get_summary(),
            # Add template representation?
            # "best_template_content": self.best_template.render({}, ignore_missing_variables=True)
        }


class RLPromptOptimizer:
    """
    基于强化学习思想的提示词优化器。
    (Does NOT inherit from PromptOptimizer)

    Prompt optimizer based on reinforcement learning ideas.
    """

    def __init__(self,
                target_llm_client: BaseLLMClient,
                optimizer_llm_client: BaseLLMClient,
                initial_template: BasePromptTemplate,
                # Evaluator now takes template and SINGLE batch
                evaluator: Callable[[BasePromptTemplate, List[Dict[str, Any]]], Tuple[float, Dict[str, MetricResult]]],
                config: Optional[OptimizationConfig] = None,
                reward_calculator: Optional[RewardCalculator] = None,
                action_memory: Optional[ActionMemory] = None):
        """
        初始化RL提示词优化器。

        Args:
            target_llm_client: 目标LLM客户端，用于执行提示词
                              (Target LLM client for executing prompts)
            optimizer_llm_client: 优化器LLM客户端，用于生成动作
                                 (Optimizer LLM client for generating actions)
            initial_template: 初始提示词模板 (Initial prompt template)
            evaluator: 评估函数，接收模板和数据批次，返回(分数, 详细结果)
                      (Evaluation function that takes template and data batch and returns (score, detailed results))
            config: 优化配置 (Optimization configuration)
            reward_calculator: 奖励计算器 (Reward calculator)
            action_memory: 动作记忆 (Action memory)
        """
        # --- Logger Initialization ---
        self.logger = logging.getLogger(self.__class__.__name__)
        # ---------------------------

        self.target_llm_client = target_llm_client
        self.optimizer_llm_client = optimizer_llm_client
        self.initial_template = copy.deepcopy(initial_template)
        self.evaluator = evaluator # Evaluator function, not Evaluator class instance
        self.config = config or OptimizationConfig()
        self.reward_calculator = reward_calculator or create_default_reward_calculator()
        self.action_memory = action_memory or ActionMemory(max_entries=getattr(self.config,'action_memory_size', 20),
                                                          decay_factor=getattr(self.config,'action_memory_decay', 0.9))

        # --- State Initialization ---
        self.current_template: BasePromptTemplate = copy.deepcopy(self.initial_template)
        self._best_score: float = -float("inf") # Instance variable for best score found
        self._best_template: BasePromptTemplate = copy.deepcopy(self.initial_template) # Instance variable for best template
        # --------------------------

        # --- Monitoring Initialization ---
        tracker_config = getattr(self.config, 'tracker_config', {})
        self.tracker = OptimizationTracker(
            experiment_name=tracker_config.get('experiment_name', f"rl_opt_{int(time.time())}"),
            save_dir=tracker_config.get('save_dir', './optimization_logs'),
            save_history=tracker_config.get('save_history', True),
            autosave_interval=tracker_config.get('autosave_interval', 10),
            track_memory_usage=tracker_config.get('track_memory_usage', False)
        )
        self.monitor = LiveProgressMonitor(self.tracker) if getattr(self.config, 'use_live_monitor', True) else None # Default to True?
        # -------------------------------


        # --- RL Specific State ---
        self.exploration_context = {
            "explored_actions": set(),  # 记录已探索的动作 (Record explored actions)
            # global_best_score will be initialized after initial evaluation
            "consecutive_no_improvement": 0,  # 连续无改进次数 (Consecutive steps without improvement)
            "exploration_rate": getattr(self.config, 'initial_exploration_rate', 0.3),  # 初始探索率 (Initial exploration rate)
            "temperature": getattr(self.config, 'initial_temperature', 1.0),  # 初始温度 (Initial temperature)
        }
        # -----------------------

        # Step history (distinct from tracker history)
        self.history: List[Dict[str, Any]] = []
        self._current_step = 0 # Track total steps processed

        self.logger.info(f"Initialized RLPromptOptimizer with batch_size={self.config.batch_size}, max_steps={self.config.max_steps}") # Use max_steps if defined
        self.logger.debug(f"Config: {self.config}")
        self.logger.debug(f"Tracker Config: {tracker_config}")


    def optimize(self, dataset: List[Dict[str, Any]], validation_dataset: Optional[List[Dict[str, Any]]] = None) -> RLOptimizationResult:
        """
        执行RL风格的提示词优化。

        Args:
            dataset: 优化用数据集 (Dataset for optimization)
            validation_dataset: 验证数据集 (Validation dataset)

        Returns:
            优化结果 (Optimization result)
        """
        overall_start_time = time.time()
        self._current_step = 0 # Reset step counter

        # Use instance variable for current template
        # current_template = copy.deepcopy(self.initial_template) # Removed

        # --- Initial Evaluation ---
        # Evaluate initial template on the *entire* dataset for a stable starting point? Or just first batch?
        # Let's evaluate on the first batch for speed, or full if dataset is small? Configurable?
        # For now, evaluate on first batch.
        try:
             initial_batch = self._get_batches(dataset, self.config.batch_size)[0]
             initial_score, _ = self._evaluate_template_on_batch(self.current_template, initial_batch)
        except IndexError:
             self.logger.error("Dataset is empty or could not form initial batch. Cannot start optimization.")
             initial_score = -float('inf') # Or handle differently? Raise error?
             # Return an empty/error result
             return RLOptimizationResult(
                  best_template=self.current_template, best_score=initial_score, start_score=initial_score,
                  history=[], action_memory=self.action_memory, optimization_time=0, tracker_summary=None
             )
        except Exception as e:
             self.logger.error(f"Error during initial evaluation: {e}", exc_info=True)
             initial_score = -float('inf') # Treat as failure

        # Initialize best score/template with initial state
        self._best_score = initial_score
        self._best_template = copy.deepcopy(self.current_template)
        start_score = initial_score # Store the initial score for the result object

        # Initialize exploration context's global best score
        self.exploration_context["global_best_score"] = initial_score
        self.logger.info(f"Initial template score (on first batch): {initial_score:.4f}")
        # -------------------------

        # Determine total steps based on config (epochs * batches_per_epoch OR max_steps)
        total_steps = self.config.max_steps
        if total_steps is None:
             try:
                  if hasattr(dataset, '__len__') and self.config.batch_size > 0:
                       num_batches_per_epoch = (len(dataset) + self.config.batch_size - 1) // self.config.batch_size
                       total_steps = self.config.epochs * num_batches_per_epoch
                  else:
                       raise ValueError("Cannot determine steps from epochs without dataset length")
             except (TypeError, ValueError, AttributeError) as e:
                  self.logger.warning(f"Could not determine total steps from config/dataset ({e}). Progress bar might be inaccurate or run indefinitely.")
                  total_steps = float('inf') # Run indefinitely? Or set a large default? Let's use float('inf')

        if total_steps <= 0:
            self.logger.warning(f"Total steps calculated as {total_steps}. Setting to 1.")
            total_steps = 1

        if self.monitor:
            self.monitor.start(total_steps=total_steps if total_steps != float('inf') else 0) # tqdm total=0 runs indefinitely

        # Main optimization loop (step-based)
        continue_optimizing = True
        while continue_optimizing:
            step_start_time = time.time()

            # 1. Sample a batch from the dataset
            # Simple random sampling per step, or iterate through batches? Iterate for now.
            # This requires tracking epoch/batch index or just using _current_step with modulo
            if not dataset:
                self.logger.error("Dataset is empty, stopping optimization.")
                break
            batch_idx = self._current_step % ((len(dataset) + self.config.batch_size - 1) // self.config.batch_size)
            epoch = (self._current_step // ((len(dataset) + self.config.batch_size - 1) // self.config.batch_size)) + 1
            try:
                 # Get batch based on current step index
                 start_idx = batch_idx * self.config.batch_size
                 end_idx = start_idx + self.config.batch_size
                 batch = dataset[start_idx:end_idx]
                 if not batch: # Handle potential empty batch if dataset size not multiple of batch_size
                      self.logger.warning(f"Step {self._current_step}: Encountered empty batch, skipping step.")
                      self._current_step += 1
                      if self._current_step >= total_steps: continue_optimizing = False
                      continue # Skip to next step
            except Exception as e:
                 self.logger.error(f"Error getting batch for step {self._current_step}: {e}", exc_info=True)
                 # Decide whether to stop or skip
                 self._current_step += 1 # Advance step counter even on error
                 if self._current_step >= total_steps: continue_optimizing = False
                 continue # Skip to next step

            step_info: Dict[str, Any] = {
                 "step": self._current_step,
                 "epoch": epoch, # Approximate epoch based on step
                 "batch_index_in_epoch": batch_idx,
                 "template_before_hash": hash(self.current_template.render({}, ignore_missing_variables=True)),
                 "accepted": False,
                 # Add exploration context?
                 "exploration_rate": self.exploration_context["exploration_rate"],
                 "temperature": self.exploration_context["temperature"],
            }


            # 2. Evaluate current template on the sampled batch
            try:
                 before_score, detailed_results = self._evaluate_template_on_batch(self.current_template, batch)
                 step_info["before_score"] = before_score
                 step_info["detailed_results_before"] = {k: dataclasses.asdict(v) for k, v in detailed_results.items()}
            except Exception as e:
                 self.logger.error(f"Error evaluating template before action at step {self._current_step}: {e}", exc_info=True)
                 step_info["error"] = f"Eval before failed: {e}"
                 self.tracker.update(step_info)
                 self._current_step += 1
                 if self.monitor: self.monitor.update()
                 if self._current_step >= total_steps: continue_optimizing = False
                 continue # Skip step

            # 3. Generate optimization action
            try:
                 action = self._generate_action(self.current_template, batch, before_score, detailed_results) # Pass original MetricResult
                 step_info["action"] = action.to_dict()
            except Exception as e:
                 self.logger.error(f"Error generating action at step {self._current_step}: {e}", exc_info=True)
                 step_info["error"] = f"Action generation failed: {e}"
                 self.tracker.update(step_info)
                 self._current_step += 1
                 if self.monitor: self.monitor.update()
                 if self._current_step >= total_steps: continue_optimizing = False
                 continue # Skip step

            # 4. Execute action to get new template
            executor = ActionExecutor(self.current_template)
            try:
                 new_template = executor.execute(action)
            except Exception as e:
                 self.logger.error(f"Error executing action at step {self._current_step}: {e}", exc_info=True)
                 step_info["error"] = f"Action execution failed: {e}"
                 new_template = self.current_template # Keep original on error

            # 5. Evaluate new template performance on the same batch
            try:
                 # Only evaluate if template actually changed
                 if new_template is not self.current_template:
                      after_score, new_detailed_results = self._evaluate_template_on_batch(new_template, batch)
                 else:
                      after_score, new_detailed_results = before_score, detailed_results # No change

                 step_info["after_score"] = after_score
                 step_info["detailed_results_after"] = {k: dataclasses.asdict(v) for k, v in new_detailed_results.items()}
            except Exception as e:
                 self.logger.error(f"Error evaluating template after action at step {self._current_step}: {e}", exc_info=True)
                 step_info["error"] = f"Eval after failed: {e}"
                 # Should we proceed without after_score? Reward calc will fail. Let's use before_score.
                 after_score = before_score
                 step_info["after_score"] = after_score
                 # Continue processing to record the failed step

            # 6. Calculate reward
            try:
                 reward = self.reward_calculator.calculate(
                     before_score=before_score,
                     after_score=after_score,
                     action=action,
                     context=self.exploration_context # Pass context for potentially richer rewards
                 )
                 step_info["reward"] = reward
            except Exception as e:
                 self.logger.error(f"Error calculating reward at step {self._current_step}: {e}", exc_info=True)
                 step_info["error"] = f"Reward calculation failed: {e}"
                 reward = 0 # Assign neutral reward on error?

            # 7. Record experience in ActionMemory
            # Check if action needs conversion to dict? Assume it does if it's StructuredAction
            action_dict_for_memory = action.to_dict() if hasattr(action, 'to_dict') else action
            self.action_memory.add(
                 action=action_dict_for_memory,
                 reward=reward,
                 before_score=before_score,
                 after_score=after_score
            )

            # 8. Decide whether to accept the new template
            accept = False
            if new_template is not self.current_template: # Only consider accepting if template actually changed
                 accept = self._should_accept_template(before_score, after_score, reward)

            step_info["accepted"] = accept

            if accept:
                 self.current_template = new_template # Update the instance variable
                 self.logger.info(f"Step {self._current_step}: Accepted new template. Score: {after_score:.4f} -> {before_score:.4f}. Reward: {reward:.2f}")
                 # Update consecutive_no_improvement based on score change
                 if after_score > before_score:
                      self.exploration_context["consecutive_no_improvement"] = 0
                 else:
                      self.exploration_context["consecutive_no_improvement"] += 1
            else:
                 # If not accepted, score remains 'before_score'
                 self.logger.info(f"Step {self._current_step}: Rejected new template. Score: {before_score:.4f}. Reward: {reward:.2f}")
                 self.exploration_context["consecutive_no_improvement"] += 1


            # 9. Update global best score and template
            current_eval_score = after_score if accept else before_score
            validation_score = None # Placeholder for potential validation

            # --- Optional Validation Step ---
            # Run validation periodically? e.g., every N steps or if score improved?
            run_validation = False
            validation_frequency = getattr(self.config, 'validation_frequency', None) # e.g., validate every 10 steps
            if validation_dataset and validation_frequency and (self._current_step % validation_frequency == 0):
                 run_validation = True
            # Also run validation if the current training score is the best seen so far?
            if validation_dataset and current_eval_score > self.exploration_context["global_best_score"]:
                 run_validation = True

            score_to_compare_for_best = current_eval_score # Default to train score

            if run_validation:
                 self.logger.info(f"Step {self._current_step}: Running validation...")
                 try:
                      # Evaluate the template that resulted from this step (current_template)
                      val_score, val_details = self._evaluate_template(self.current_template, validation_dataset)
                      validation_score = val_score
                      step_info["validation_score"] = validation_score
                      step_info["validation_details"] = {k: dataclasses.asdict(v) for k, v in val_details.items()}
                      score_to_compare_for_best = validation_score # Use validation score for best check
                      self.logger.info(f"Step {self._current_step}: Validation score = {validation_score:.4f}")
                 except Exception as e:
                      self.logger.error(f"Error during periodic validation at step {self._current_step}: {e}", exc_info=True)
                      step_info["error"] = f"Validation failed: {e}"
            # --- End Validation Step ---

            # Update best score and template using the chosen score (train or validation)
            if score_to_compare_for_best > self._best_score:
                 self._best_score = score_to_compare_for_best
                 self._best_template = copy.deepcopy(self.current_template) # Save copy
                 self.logger.info(f"*** Step {self._current_step}: New best score found: {self._best_score:.4f} {'(Validation)' if run_validation and score_to_compare_for_best == validation_score else '(Training)'} ***")
                 # Reset no improvement counter when best improves
                 # self.exploration_context["consecutive_no_improvement"] = 0 # Already handled by accept logic based on train score? Maybe reset here too?

            # Update global best score used by reward calculator context (always based on train score?)
            # Or should this also use validation score if available? Let's stick to training score for context for now.
            if current_eval_score > self.exploration_context["global_best_score"]:
                 self.exploration_context["global_best_score"] = current_eval_score


            # 10. Record step history (distinct from tracker history)
            self.history.append({k: v for k, v in step_info.items() if k not in ['detailed_results_before', 'detailed_results_after', 'validation_details']}) # Keep history lean

            # 11. Adjust exploration strategy
            self._adjust_exploration_strategy(epoch, batch_idx, reward) # Pass necessary context

            # --- Update Tracker and Monitor ---
            step_info["best_score_so_far"] = self._best_score # Add best score to tracker
            self.tracker.update(step_info)
            if self.monitor:
                 # Update monitor description
                 best_score_display = f"{self._best_score:.3f}" if self._best_score > -float("inf") else "N/A"
                 val_score_display = f"|Val {validation_score:.3f}" if validation_score is not None else ""
                 desc = f"Step {self._current_step} Score {current_eval_score:.3f}|Best {best_score_display}{val_score_display}|Rew {reward:.1f}"
                 if accept: desc += " (Acc)"
                 self.monitor.progress_bar.set_description(desc)
                 self.monitor.update()
            # ----------------------------------

            # Increment step counter
            self._current_step += 1

            # Check stopping conditions
            if self._current_step >= total_steps:
                 self.logger.info(f"Reached max_steps ({total_steps}). Stopping.")
                 continue_optimizing = False
            if self._should_stop_early():
                 self.logger.info(f"Early stopping triggered at step {self._current_step}.")
                 continue_optimizing = False
            # Check for time limit?
            # time_limit = getattr(self.config, 'max_time_seconds', None)
            # if time_limit and (time.time() - overall_start_time > time_limit):
            #      self.logger.info(f"Reached time limit ({time_limit}s). Stopping.")
            #      continue_optimizing = False


        # --- Finalization ---
        if self.monitor:
            self.monitor.stop()

        optimization_duration = time.time() - overall_start_time
        self.logger.info("--- RL Optimization Finished ---")
        self.logger.info(f"Total Duration: {optimization_duration:.2f}s")
        self.logger.info(f"Completed {self._current_step} steps.")
        self.logger.info(f"Best Score Achieved: {self._best_score:.4f}")

        # --- Save final tracker state and get summary/history ---
        tracker_summary = None
        tracker_history = None
        try:
            final_summary_path = self.tracker.save()
            self.logger.info(f"Final monitoring data saved. Summary: {final_summary_path}")
            tracker_summary = self.tracker.get_summary()
            tracker_history = self.tracker.history # Get tracker history if needed
        except Exception as e:
             self.logger.error(f"Failed to save/get final monitoring data: {e}", exc_info=True)

        # --- Optionally generate report/plots ---
        if getattr(self.config, 'generate_report_at_end', False):
            try:
                 report_path = self.tracker.generate_report()
                 if report_path:
                      self.logger.info(f"Generated final optimization report: {report_path}")
                 # Optionally plot progress and action stats
                 self.tracker.plot_progress()
                 self.tracker.plot_action_statistics()
            except Exception as e:
                 self.logger.error(f"Failed to generate final report/plots: {e}", exc_info=True)
        # ------------------------------------


        # --- Create final result object ---
        final_result = RLOptimizationResult(
            best_template=self._best_template,
            best_score=self._best_score,
            start_score=start_score,
            history=self.history, # Step history collected during optimize
            tracker_history=tracker_history, # History from tracker object
            action_memory=self.action_memory,
            optimization_time=optimization_duration,
            tracker_summary=tracker_summary
        )

        return final_result


    def _get_batches(self, dataset: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        """Helper to split dataset into batches."""
        if not dataset: return []
        if batch_size <= 0: batch_size = len(dataset) # Use full dataset if invalid batch size
        return [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]

    def _evaluate_template(self, template: BasePromptTemplate, dataset: List[Dict[str, Any]]) -> Tuple[float, Dict[str, MetricResult]]:
        """Evaluates template on the entire dataset using the provided evaluator function."""
        # This might be slow for large validation sets. Consider sampling?
        if not dataset:
            self.logger.warning("Evaluation dataset is empty.")
            return -float('inf'), {}
        self.logger.info(f"Evaluating template on {len(dataset)} samples...")
        return self.evaluator(template, dataset) # Call the evaluator function passed in __init__

    def _evaluate_template_on_batch(self, template: BasePromptTemplate, batch: List[Dict[str, Any]]) -> Tuple[float, Dict[str, MetricResult]]:
        """Evaluates template on a single batch using the provided evaluator function."""
        if not batch:
            self.logger.warning("Evaluation batch is empty.")
            return -float('inf'), {}
        # Directly use the callable evaluator
        return self.evaluator(template, batch)

    # --- Action Generation Methods ---
    def _generate_action(self,
                        template: BasePromptTemplate,
                        batch: List[Dict[str, Any]],
                        current_score: float,
                        detailed_results: Dict[str, MetricResult]) -> StructuredAction:
        """Generates optimization action using exploration/exploitation."""
        if random.random() < self.exploration_context["exploration_rate"]:
            self.logger.debug("Generating action via exploration.")
            return self._explore_action(template)
        else:
            self.logger.debug("Generating action via exploitation (LLM).")
            return self._exploit_action(template, batch, current_score, detailed_results)

    def _explore_action(self, template: BasePromptTemplate) -> StructuredAction:
        """Exploration strategy: Randomly sample from action space."""
        action_space = ActionSpace(template)
        return action_space.sample()

    def _exploit_action(self,
                       template: BasePromptTemplate,
                       batch: List[Dict[str, Any]],
                       current_score: float,
                       detailed_results: Dict[str, MetricResult]) -> StructuredAction:
        """Exploitation strategy: Use LLM to suggest an action based on context."""
        prompt = self._create_action_generation_prompt(template, batch, current_score, detailed_results)
        response = ""
        try:
            response = self.optimizer_llm_client.generate(prompt) # Assuming generate handles potential errors
            # Attempt parsing
            action_dict = parse_json_robust(response, default_value={}, expected_type=dict)
            if not action_dict or "target_section" not in action_dict:
                 action_dict = extract_action_json(response) # Try regex fallback

            if not action_dict or "target_section" not in action_dict:
                self.logger.warning("Failed to parse valid action JSON from LLM response, falling back to exploration.")
                self.logger.debug(f"LLM Response causing fallback: {response[:500]}...") # Log snippet
                return self._explore_action(template)

            # Create action from dictionary
            action_space = ActionSpace(template)
            return action_space.create_from_dict(action_dict)

        except Exception as e:
            self.logger.error(f"Error during LLM action generation or parsing: {e}", exc_info=True)
            self.logger.debug(f"LLM Response causing error: {response[:500]}...") # Log snippet
            return self._explore_action(template) # Fallback on error

    def _create_action_generation_prompt(self,
                                        template: BasePromptTemplate,
                                        batch: List[Dict[str, Any]],
                                        current_score: float,
                                        detailed_results: Dict[str, MetricResult]) -> str:
        """Creates the prompt for the optimizer LLM to generate an action."""
        # Extract optimizable sections
        sections_str = ""
        optimizable_sections = template.get_optimizable_sections()
        if not optimizable_sections:
             self.logger.warning("Template has no optimizable sections defined!")
             # Handle this case - maybe allow modifying non-optimizable? For now, list all sections.
             optimizable_sections = {name: template.get_section(name) for name in template.list_sections()}

        for section_name, content in optimizable_sections.items():
            # Truncate long sections for the prompt context
            display_content = str(content or '(Empty)')
            if len(display_content) > 300: # Limit context per section
                 display_content = display_content[:147] + "..." + display_content[-147:]
            sections_str += f"--- {section_name} ---\n{display_content}\n\n"

        # Format batch examples (limit number and length)
        batch_examples = ""
        for i, example in enumerate(batch[:min(3, len(batch))]): # Show max 3 examples
            input_text = str(example.get("input", "(No input)"))
            reference = str(example.get("reference", "(No reference)"))
            if len(input_text) > 150: input_text = input_text[:72] + "..." + input_text[-72:]
            if len(reference) > 150: reference = reference[:72] + "..." + reference[-72:]
            batch_examples += f"Example {i+1}:\nInput: {input_text}\nReference: {reference}\n\n"

        # Format detailed results (scores only, maybe top/bottom K details?)
        results_str = ""
        for metric_name, result in detailed_results.items():
            results_str += f"- {metric_name}: {result.score:.4f}\n"
            # Optionally add brief details if available and concise
            # if result.details and isinstance(result.details, dict) and len(str(result.details)) < 100:
            #     results_str += f"  Details: {result.details}\n"

        # Format recent history from ActionMemory
        history_str = self.action_memory.format_for_prompt(max_records=5, format_type='detailed')

        # Construct the meta-prompt
        # Added explicit mention of section names for target_section.
        valid_sections = list(optimizable_sections.keys())
        prompt = f"""You are a Prompt Engineering Optimizer. Your goal is to suggest ONE specific change to improve the prompt template based on performance feedback.

CURRENT OPTIMIZABLE TEMPLATE SECTIONS:
{sections_str}
Valid sections for 'target_section' are: {valid_sections}

CURRENT PERFORMANCE (Score: {current_score:.4f}):
{results_str}
RECENT DATA EXAMPLES:
{batch_examples}
RECENT OPTIMIZATION HISTORY (Action -> Reward):
{history_str}

Based on the template, performance, and history, suggest ONE specific, actionable change (an ACTION) to apply to ONE section from the list above ({valid_sections}) to improve the performance score.
Consider the feedback and history. If performance is decreasing or stuck, try a different type of change or target section.

Return your suggested ACTION as a **single JSON object** with these fields:
- "target_section": The section to modify (MUST be one of {valid_sections}).
- "action_type": The type of action (CHOOSE ONE: REPLACE, MODIFY, EXPAND, SIMPLIFY, RESTRUCTURE).
- "content": The new text content to use for the section based on the action_type. For MODIFY, this might describe the change. For REPLACE, it's the full new content.

Your output MUST be only the JSON object. Ensure "target_section" is valid.

```json
{{
  "target_section": "SECTION_NAME",
  "action_type": "ACTION_TYPE",
  "content": "New content for the section..."
}}
```"""
        return prompt

    # --- Acceptance and Exploration Adjustment ---
    def _should_accept_template(self, before_score: float, after_score: float, reward: float) -> bool:
        """Decides whether to accept the new template based on scores and annealing."""
        # Always accept if score improves significantly (e.g., > epsilon)
        if after_score > before_score + 1e-6:
            return True

        # Optionally accept based on reward (if reward function is trusted)
        # if reward > getattr(self.config, 'reward_acceptance_threshold', 0):
        #    return True

        # Simulated Annealing: Accept worse solutions with a probability based on temperature
        temperature = self.exploration_context.get("temperature", 0) # Default to 0 if missing
        if temperature > 1e-6: # Avoid division by zero or negligible temperature
            delta = after_score - before_score # Will be <= 0 here
            # Ensure delta isn't extremely negative causing exp underflow
            delta_scaled = delta / temperature
            if delta_scaled < -50: # Prevent math range error
                 acceptance_prob = 0.0
            else:
                 acceptance_prob = math.exp(delta_scaled)

            accept_random = random.random() < acceptance_prob
            if accept_random:
                 self.logger.debug(f"Annealing accepted worse score ({after_score:.4f} < {before_score:.4f}) with prob {acceptance_prob:.3f} at temp {temperature:.3f}")
            return accept_random

        return False # If no improvement and temperature is ~0, reject

    def _adjust_exploration_strategy(self, epoch: int, batch_idx: int, reward: float) -> None:
        """Adjusts exploration rate and temperature."""
        # Adjust exploration rate based on consecutive no improvement
        consecutive_no_improve = self.exploration_context["consecutive_no_improvement"]
        max_no_improve_patience = getattr(self.config, 'exploration_increase_patience', 5) # Steps before increasing rate
        rate_increase_factor = getattr(self.config, 'exploration_increase_factor', 0.05)
        rate_decrease_factor = getattr(self.config, 'exploration_decrease_factor', 0.02)
        min_exploration_rate = getattr(self.config, 'min_exploration_rate', 0.05)
        max_exploration_rate = getattr(self.config, 'max_exploration_rate', 0.5)

        current_rate = self.exploration_context["exploration_rate"]
        if consecutive_no_improve > max_no_improve_patience:
            current_rate += rate_increase_factor
        elif reward > 0: # Or based on score improvement?
             # Only decrease if we actually improved the score
             # if score improved... (need access to scores here?)
             current_rate -= rate_decrease_factor

        self.exploration_context["exploration_rate"] = max(min_exploration_rate, min(max_exploration_rate, current_rate))

        # Adjust temperature (Simulated Annealing) - simple linear decay for now
        # Needs total_steps calculation to be reliable.
        initial_temp = getattr(self.config, 'initial_temperature', 1.0)
        final_temp = getattr(self.config, 'final_temperature', 0.01)
        temp_decay_steps = getattr(self.config, 'temperature_decay_steps', self.config.max_steps or 100) # Decay over max_steps or default

        if temp_decay_steps > 0:
             progress = min(1.0, self._current_step / temp_decay_steps)
             current_temp = initial_temp + (final_temp - initial_temp) * progress
             self.exploration_context["temperature"] = max(final_temp, current_temp)
        else:
             self.exploration_context["temperature"] = final_temp


        # self.logger.debug( # Reduce log frequency?
        #     f"Adjusted strategy: rate={self.exploration_context['exploration_rate']:.2f}, "
        #     f"temp={self.exploration_context['temperature']:.3f}, no_improve={consecutive_no_improve}"
        # )

    def _should_stop_early(self) -> bool:
        """Checks early stopping conditions."""
        # Stop if no improvement for patience steps (based on training score changes)
        patience = getattr(self.config, 'early_stopping_patience', None)
        if patience is not None and self.exploration_context["consecutive_no_improvement"] >= patience:
             self.logger.info(f"Early stopping: No improvement for {patience} steps.")
             return True

        # Stop if score reaches a target threshold?
        target_score = getattr(self.config, 'target_score', None)
        if target_score is not None and self._best_score >= target_score:
             self.logger.info(f"Early stopping: Target score {target_score} reached (Best: {self._best_score:.4f}).")
             return True

        return False

    # --- Checkpointing Methods ---
    def save_optimizer_checkpoint(self, checkpoint_path: str):
        """Saves the current state of the RL optimizer to a checkpoint file."""
        # save_checkpoint expects optimizer object and will try to pickle attributes.
        # It specifically looks for: _best_template, _best_score, current_template,
        # exploration_context, action_memory, history.
        try:
            save_checkpoint(
                optimizer=self,
                checkpoint_path=checkpoint_path,
                include_memory=True, # RL Optimizer uses action_memory
                include_history=True # We have self.history (step history)
            )
            self.logger.info(f"RL Optimizer checkpoint successfully saved to {checkpoint_path}")
        except SerializationError as e:
            self.logger.error(f"Failed to save RL optimizer checkpoint: {e}")
        except AttributeError as e:
            self.logger.error(f"Failed to save checkpoint due to missing attribute: {e}. Check optimizer state structure.", exc_info=True)
        except Exception as e:
             self.logger.error(f"An unexpected error occurred saving checkpoint: {e}", exc_info=True)


    def load_optimizer_checkpoint(self, checkpoint_path: str):
        """Loads the RL optimizer state from a checkpoint file."""
        try:
            # Restores attributes onto `self`
            load_checkpoint(checkpoint_path=checkpoint_path, optimizer=self)
            self.logger.info(f"RL Optimizer checkpoint successfully loaded from {checkpoint_path}")

            # Post-load actions specific to RL:
            # - Restore _current_step ? Need to save it first in save_checkpoint.
            # - Reset tracker/monitor state.
            # - Ensure exploration_context and action_memory were loaded correctly.

            # Simple approach: Log warning and reset internal counters/tracker state.
            self.logger.warning("RL Optimizer state loaded. Monitoring/progress tracking will restart.")
            self._current_step = 0 # Reset step counter, progress bar will restart
            # Re-initialize tracker? Or let it continue logging? Let it continue.

            # Check for essential loaded attributes
            if not hasattr(self, 'current_template'):
                 self.logger.warning("Loaded checkpoint missing 'current_template'. Using initial_template.")
                 self.current_template = copy.deepcopy(self.initial_template)
            if not hasattr(self, '_best_template'):
                 self.logger.warning("Loaded checkpoint missing '_best_template'. Using current_template.")
                 self._best_template = copy.deepcopy(self.current_template)
            if not hasattr(self, '_best_score'):
                 self.logger.warning("Loaded checkpoint missing '_best_score'. Setting to -inf.")
                 self._best_score = -float('inf')
            if not hasattr(self, 'history'):
                 self.logger.warning("Loaded checkpoint missing 'history'. Initializing empty list.")
                 self.history = []
            if not hasattr(self, 'action_memory'):
                 self.logger.warning("Loaded checkpoint missing 'action_memory'. Initializing new one.")
                 self.action_memory = ActionMemory() # Re-initialize if missing
            if not hasattr(self, 'exploration_context'):
                 self.logger.warning("Loaded checkpoint missing 'exploration_context'. Initializing default.")
                 self.exploration_context = { # Re-initialize default if missing
                      "explored_actions": set(),
                      "global_best_score": self._best_score, # Use loaded best score if available
                      "consecutive_no_improvement": 0,
                      "exploration_rate": getattr(self.config, 'initial_exploration_rate', 0.3),
                      "temperature": getattr(self.config, 'initial_temperature', 1.0),
                 }
            else:
                 # Ensure loaded context has all keys, add defaults if missing
                 defaults = {
                     "explored_actions": set(), "global_best_score": self._best_score,
                     "consecutive_no_improvement": 0, "exploration_rate": 0.3, "temperature": 1.0,
                 }
                 for key, default_val in defaults.items():
                      if key not in self.exploration_context:
                           self.logger.warning(f"Loaded exploration_context missing '{key}'. Setting default.")
                           self.exploration_context[key] = default_val


        except SerializationError as e:
            self.logger.error(f"Failed to load RL optimizer checkpoint: {e}")
            raise # Re-raise
        except FileNotFoundError:
            self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
            raise # Re-raise
        except Exception as e:
             self.logger.error(f"An unexpected error occurred loading checkpoint: {e}", exc_info=True)
             raise # Re-raise
    # ---------------------------