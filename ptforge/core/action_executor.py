# ptforge/core/action_executor.py
"""
负责执行提示词优化动作。
将抽象动作应用到具体模板上，确保动作执行的一致性和可靠性。

Responsible for executing prompt optimization actions.
Applies abstract actions to concrete templates, ensuring consistency and reliability of action execution.
"""

import logging
import copy
from typing import Dict, Any, List, Optional, Union, Tuple, Type, cast

from ptforge.templates.base_template import BasePromptTemplate
from ptforge.templates.action_space import StructuredAction, ActionType, ActionSpace
from ptforge.utils.json_helper import parse_json_robust

logger = logging.getLogger(__name__)


class ActionExecutionError(Exception):
    """
    动作执行过程中发生的错误。
    
    Error that occurs during action execution.
    """
    pass


class ActionExecutor:
    """
    执行提示词优化动作的核心组件。
    
    Core component for executing prompt optimization actions.
    """
    
    def __init__(self, 
                 template: BasePromptTemplate, 
                 restrict_to_optimizable: bool = True,
                 enable_recovery: bool = True):
        """
        初始化动作执行器。
        
        Initialize action executor.
        
        Args:
            template: 要操作的提示词模板 (Prompt template to operate on)
            restrict_to_optimizable: 是否限制为可优化部分 (Whether to restrict to optimizable sections)
            enable_recovery: 是否启用错误恢复机制 (Whether to enable error recovery mechanisms)
        """
        self.template = template
        self.action_space = ActionSpace(template, restrict_to_optimizable)
        self.enable_recovery = enable_recovery
        
        logger.debug(f"Initialized ActionExecutor with template {type(template).__name__}")
    
    def execute(self, action: Union[StructuredAction, Dict[str, Any]]) -> BasePromptTemplate:
        """
        执行一个动作，返回更新后的模板副本。
        
        Execute an action, returning an updated copy of the template.
        
        Args:
            action: 要执行的动作，可以是StructuredAction对象或字典 
                   (Action to execute, can be a StructuredAction object or a dictionary)
            
        Returns:
            更新后的模板副本 (Updated copy of the template)
            
        Raises:
            ActionExecutionError: 如果动作执行失败 (If action execution fails)
        """
        # 标准化动作对象
        # Normalize action object
        if isinstance(action, dict):
            try:
                structured_action = self.action_space.create_from_dict(action)
            except ValueError as e:
                raise ActionExecutionError(f"Invalid action dictionary: {e}")
        else:
            structured_action = action
        
        try:
            # 应用动作
            # Apply action
            new_template = self.action_space.apply_action(structured_action)
            logger.info(f"Successfully executed action: {structured_action}")
            return new_template
            
        except Exception as e:
            logger.error(f"Failed to execute action: {e}", exc_info=True)
            
            if self.enable_recovery:
                return self._handle_execution_error(structured_action, e)
            else:
                raise ActionExecutionError(f"Action execution failed: {e}")
    
    def execute_from_llm_response(self, llm_response: str) -> Tuple[BasePromptTemplate, StructuredAction]:
        """
        从LLM响应中提取并执行动作。
        
        Extract and execute an action from an LLM response.
        
        Args:
            llm_response: LLM的原始响应文本 (Raw response text from LLM)
            
        Returns:
            包含更新后的模板和执行的动作的元组 
            (Tuple containing updated template and executed action)
            
        Raises:
            ActionExecutionError: 如果无法从响应中提取或执行动作
                                  (If action cannot be extracted or executed from response)
        """
        # 解析动作
        # Parse action
        action_dict = parse_json_robust(llm_response, default_value={}, expected_type=dict)
        
        if not action_dict:
            # 尝试使用正则表达式提取
            # Try to extract using regex
            from ptforge.utils.json_helper import extract_action_json
            action_dict = extract_action_json(llm_response)
            
            if not action_dict:
                raise ActionExecutionError("Failed to extract action from LLM response")
        
        try:
            structured_action = self.action_space.create_from_dict(action_dict)
            new_template = self.execute(structured_action)
            return new_template, structured_action
            
        except Exception as e:
            logger.error(f"Failed to execute action from LLM response: {e}", exc_info=True)
            raise ActionExecutionError(f"Failed to execute action from LLM response: {e}")
    
    def _handle_execution_error(self, action: StructuredAction, error: Exception) -> BasePromptTemplate:
        """
        处理动作执行错误，尝试恢复。
        
        Handle action execution errors with recovery attempts.
        
        Args:
            action: 失败的动作 (Failed action)
            error: 发生的错误 (Occurred error)
            
        Returns:
            尽可能恢复的模板副本，或原始模板的副本作为后备
            (Recovered template copy if possible, or copy of original template as fallback)
        """
        logger.warning(f"Attempting to recover from action execution error: {error}")
        
        # 创建模板副本作为后备
        # Create template copy as fallback
        template_copy = copy.deepcopy(self.template)
        
        # 尝试不同的恢复策略，从最简单到最复杂
        # Try different recovery strategies, from simplest to most complex
        
        # 策略1：保留原始内容
        # Strategy 1: Keep original content
        if action.action_type in [ActionType.MODIFY, ActionType.SIMPLIFY, ActionType.RESTRUCTURE]:
            logger.info("Recovery strategy: Keeping original content")
            return template_copy
        
        # 策略2：对于REPLACE和EXPAND，尝试简化内容
        # Strategy 2: For REPLACE and EXPAND, try to simplify content
        if action.action_type in [ActionType.REPLACE, ActionType.EXPAND] and action.content:
            try:
                logger.info("Recovery strategy: Simplifying content")
                # 取内容的前几行作为简化版本
                # Take first few lines as simplified version
                simplified_content = "\n".join(action.content.split("\n")[:3])
                
                # 创建新的简化动作
                # Create new simplified action
                simplified_action = StructuredAction(
                    action_type=action.action_type,
                    target_section=action.target_section,
                    content=simplified_content,
                    parameters=action.parameters
                )
                
                return self.action_space.apply_action(simplified_action)
            except Exception as e:
                logger.warning(f"Simplified content recovery failed: {e}")
        
        # 所有恢复策略都失败，返回原始模板副本
        # All recovery strategies failed, return copy of original template
        logger.warning("All recovery strategies failed, returning original template")
        return template_copy
    
    def batch_execute(self, 
                     actions: List[Union[StructuredAction, Dict[str, Any]]],
                     stop_on_error: bool = False) -> Tuple[BasePromptTemplate, List[StructuredAction], List[Exception]]:
        """
        批量执行多个动作，返回最终模板和成功/失败的动作列表。
        
        Execute multiple actions in batch, returning final template and lists of successful/failed actions.
        
        Args:
            actions: 要执行的动作列表 (List of actions to execute)
            stop_on_error: 是否在首次错误时停止 (Whether to stop on first error)
            
        Returns:
            元组，包含(最终模板, 成功动作列表, 错误列表)
            (Tuple containing (final template, list of successful actions, list of errors))
        """
        current_template = copy.deepcopy(self.template)
        successful_actions: List[StructuredAction] = []
        errors: List[Exception] = []
        
        for i, action in enumerate(actions):
            try:
                # 更新执行器的模板和动作空间
                # Update executor's template and action space
                self.template = current_template
                self.action_space = ActionSpace(current_template, self.action_space.restrict_to_optimizable)
                
                # 执行动作
                # Execute action
                if isinstance(action, dict):
                    structured_action = self.action_space.create_from_dict(action)
                else:
                    structured_action = action
                    
                current_template = self.execute(structured_action)
                successful_actions.append(structured_action)
                
            except Exception as e:
                logger.error(f"Error executing action {i}: {e}", exc_info=True)
                errors.append(e)
                
                if stop_on_error:
                    break
        
        return current_template, successful_actions, errors


class BatchActionExecutor:
    """
    高效批量执行动作的组件，支持并行执行。
    
    Component for efficiently executing actions in batch, with support for parallel execution.
    """
    
    def __init__(self, 
                 template: BasePromptTemplate, 
                 restrict_to_optimizable: bool = True,
                 parallel: bool = False,
                 max_workers: int = 4):
        """
        初始化批量动作执行器。
        
        Initialize batch action executor.
        
        Args:
            template: 要操作的提示词模板 (Prompt template to operate on)
            restrict_to_optimizable: 是否限制为可优化部分 (Whether to restrict to optimizable sections)
            parallel: 是否并行执行 (Whether to execute in parallel)
            max_workers: 并行执行时的最大工作线程数 (Maximum number of worker threads for parallel execution)
        """
        self.template = template
        self.restrict_to_optimizable = restrict_to_optimizable
        self.parallel = parallel
        self.max_workers = max_workers
        
        logger.debug(f"Initialized BatchActionExecutor with parallel={parallel}, max_workers={max_workers}")
    
    def execute(self, 
               actions: List[Union[StructuredAction, Dict[str, Any]]]) -> List[Tuple[BasePromptTemplate, Optional[Exception]]]:
        """
        执行多个独立的动作，每个动作应用到原始模板的副本上。
        
        Execute multiple independent actions, each applied to a copy of the original template.
        
        Args:
            actions: 要执行的动作列表 (List of actions to execute)
            
        Returns:
            结果列表，每项包含(更新后的模板, 可能的错误)
            (List of results, each containing (updated template, possible error))
        """
        if not actions:
            return []
        
        if self.parallel and len(actions) > 1:
            return self._execute_parallel(actions)
        else:
            return self._execute_sequential(actions)
    
    def _execute_sequential(self, 
                          actions: List[Union[StructuredAction, Dict[str, Any]]]) -> List[Tuple[BasePromptTemplate, Optional[Exception]]]:
        """
        顺序执行多个独立的动作。
        
        Execute multiple independent actions sequentially.
        
        Args:
            actions: 要执行的动作列表 (List of actions to execute)
            
        Returns:
            结果列表 (List of results)
        """
        results = []
        
        for action in actions:
            executor = ActionExecutor(copy.deepcopy(self.template), self.restrict_to_optimizable)
            
            try:
                new_template = executor.execute(action)
                results.append((new_template, None))
            except Exception as e:
                logger.error(f"Error executing action: {e}", exc_info=True)
                results.append((copy.deepcopy(self.template), e))
        
        return results
    
    def _execute_parallel(self, 
                        actions: List[Union[StructuredAction, Dict[str, Any]]]) -> List[Tuple[BasePromptTemplate, Optional[Exception]]]:
        """
        并行执行多个独立的动作。
        
        Execute multiple independent actions in parallel.
        
        Args:
            actions: 要执行的动作列表 (List of actions to execute)
            
        Returns:
            结果列表 (List of results)
        """
        from concurrent.futures import ThreadPoolExecutor
        
        def execute_single(action):
            executor = ActionExecutor(copy.deepcopy(self.template), self.restrict_to_optimizable)
            try:
                return executor.execute(action), None
            except Exception as e:
                logger.error(f"Error executing action in parallel: {e}", exc_info=True)
                return copy.deepcopy(self.template), e
        
        # 使用线程池并行执行
        # Use thread pool for parallel execution
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(actions))) as executor:
            return list(executor.map(execute_single, actions))
    
    def find_best_action(self, 
                        actions: List[Union[StructuredAction, Dict[str, Any]]],
                        evaluation_func) -> Tuple[Optional[BasePromptTemplate], Optional[StructuredAction], float]:
        """
        尝试多个动作并返回根据评估函数得分最高的结果。
        
        Try multiple actions and return the result with the highest score according to the evaluation function.
        
        Args:
            actions: 要尝试的动作列表 (List of actions to try)
            evaluation_func: 评估函数，接收模板并返回分数 (Evaluation function that takes a template and returns a score)
            
        Returns:
            元组，包含(最佳模板, 最佳动作, 最佳分数)，如果所有动作都失败则为(None, None, -inf)
            (Tuple containing (best template, best action, best score), or (None, None, -inf) if all actions fail)
        """
        if not actions:
            return None, None, float('-inf')
        
        # 获取所有执行结果
        # Get all execution results
        results = self.execute(actions)
        
        # 评估所有成功的结果
        # Evaluate all successful results
        best_template = None
        best_action = None
        best_score = float('-inf')
        
        for (template, error), action in zip(results, actions):
            if error is not None:
                continue  # 跳过失败的动作 (Skip failed actions)
                
            try:
                score = evaluation_func(template)
                if score > best_score:
                    best_score = score
                    best_template = template
                    best_action = action if isinstance(action, StructuredAction) else \
                                  self.action_space.create_from_dict(action) if hasattr(self, 'action_space') else \
                                  ActionSpace(self.template, self.restrict_to_optimizable).create_from_dict(action)
            except Exception as e:
                logger.error(f"Error evaluating template: {e}", exc_info=True)
        
        return best_template, best_action, best_score