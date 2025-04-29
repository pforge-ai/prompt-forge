# ptforge/utils/prompt_builder.py
"""
高效构建提示工具。
支持历史记录压缩、动态模板等，为RL优化提供优质提示。

Efficient prompt building tools.
Supports history compression, dynamic templates, etc., providing quality prompts for RL optimization.
"""

import re
import copy
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Set, Callable

from ptforge.templates.base_template import BasePromptTemplate
from ptforge.templates.action_space import StructuredAction
from ptforge.core.action_memory import ActionMemory, ActionRecord
from ptforge.core.base import MetricResult

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    提示构建器，用于高效构建优化器提示。
    
    Prompt builder for efficiently constructing optimizer prompts.
    """
    
    def __init__(self,
                max_history_items: int = 5,
                max_batch_examples: int = 3,
                max_section_length: int = 1000,
                include_metrics: bool = True,
                compact_mode: bool = False):
        """
        初始化提示构建器。
        
        Initialize prompt builder.
        
        Args:
            max_history_items: 包含的最大历史条目数 (Maximum number of history items to include)
            max_batch_examples: 包含的最大批次示例数 (Maximum number of batch examples to include)
            max_section_length: 模板部分的最大长度 (Maximum length for template sections)
            include_metrics: 是否包含评估指标详情 (Whether to include metric details)
            compact_mode: 是否使用紧凑模式 (Whether to use compact mode)
        """
        self.max_history_items = max_history_items
        self.max_batch_examples = max_batch_examples
        self.max_section_length = max_section_length
        self.include_metrics = include_metrics
        self.compact_mode = compact_mode
        
        # 预定义的提示模板
        # Predefined prompt templates
        self._templates = {
            "action_generation": self._get_action_generation_template(),
            "action_evaluation": self._get_action_evaluation_template(),
            "multi_action_generation": self._get_multi_action_generation_template(),
        }
    
    def _get_action_generation_template(self) -> str:
        """
        获取动作生成提示模板。
        
        Get action generation prompt template.
        
        Returns:
            动作生成提示模板 (Action generation prompt template)
        """
        if self.compact_mode:
            return """You are a Prompt Optimizer.

TEMPLATE:
{sections_str}

SCORE: {current_score:.4f}
{results_str}

EXAMPLES:
{batch_examples}

HISTORY:
{history_str}

Suggest ONE action to improve the template as JSON:
```json
{{"target_section": "SECTION_NAME", "action_type": "ACTION_TYPE", "content": "New content..."}}
```"""
        else:
            return """You are a Prompt Engineering Optimizer.

Your goal is to suggest ONE specific change to improve the template.

CURRENT TEMPLATE SECTIONS:
{sections_str}

CURRENT PERFORMANCE:
Score: {current_score:.4f}
{results_str}

DATA EXAMPLES:
{batch_examples}

OPTIMIZATION HISTORY (Action and Result):
{history_str}

Based on the template, performance, and optimization history, suggest ONE specific action to improve the template. 
Focus on the most promising change that will improve performance.

Return your suggestion as a JSON object with these fields:
- target_section: The section to modify (must be one of the sections shown above)
- action_type: The type of action (REPLACE, MODIFY, EXPAND, SIMPLIFY, or RESTRUCTURE)
- content: The new content for the section

Your output must be a valid JSON object only:
```json
{{
  "target_section": "SECTION_NAME",
  "action_type": "ACTION_TYPE",
  "content": "New content for the section..."
}}
```"""
    
    def _get_action_evaluation_template(self) -> str:
        """
        获取动作评估提示模板。
        
        Get action evaluation prompt template.
        
        Returns:
            动作评估提示模板 (Action evaluation prompt template)
        """
        return """You are a Prompt Evaluation Expert.

Your task is to evaluate a proposed change to a prompt template.

CURRENT TEMPLATE SECTIONS:
{sections_str}

PROPOSED ACTION:
{action_str}

RESULTING TEMPLATE SECTIONS IF APPLIED:
{result_sections_str}

CURRENT PERFORMANCE:
Score: {current_score:.4f}
{results_str}

DATA EXAMPLES:
{batch_examples}

OPTIMIZATION HISTORY:
{history_str}

Analyze the proposed change and predict its impact on performance. Consider:
1. Does the change address current weaknesses?
2. Is it consistent with successful changes in history?
3. Does it maintain critical components?
4. Will it improve handling of the example data?

Return your evaluation as a JSON object:
```json
{{
  "prediction": "BETTER|WORSE|SAME",
  "confidence": 0.1-1.0,
  "reasoning": "Your analysis of why...",
  "expected_improvement": -100 to 100
}}
```"""
    
    def _get_multi_action_generation_template(self) -> str:
        """
        获取多动作生成提示模板。
        
        Get multi-action generation prompt template.
        
        Returns:
            多动作生成提示模板 (Multi-action generation prompt template)
        """
        return """You are a Prompt Engineering Optimizer.

Your goal is to suggest MULTIPLE diverse changes to improve the template.

CURRENT TEMPLATE SECTIONS:
{sections_str}

CURRENT PERFORMANCE:
Score: {current_score:.4f}
{results_str}

DATA EXAMPLES:
{batch_examples}

OPTIMIZATION HISTORY (Action and Result):
{history_str}

Generate {num_actions} different potential actions to improve the template.
Make them diverse - try different sections and approaches.

Return your suggestions as a JSON array of actions:
```json
[
  {{
    "target_section": "SECTION_NAME",
    "action_type": "ACTION_TYPE",
    "content": "New content for the section..."
  }},
  {{
    "target_section": "ANOTHER_SECTION",
    "action_type": "ANOTHER_TYPE",
    "content": "Different content..."
  }}
]
```"""
    
    def build_action_generation_prompt(self,
                                      template: BasePromptTemplate,
                                      batch: List[Dict[str, Any]],
                                      current_score: float,
                                      detailed_results: Dict[str, MetricResult],
                                      action_memory: ActionMemory) -> str:
        """
        构建动作生成提示。
        
        Build action generation prompt.
        
        Args:
            template: 当前模板 (Current template)
            batch: 数据批次 (Data batch)
            current_score: 当前分数 (Current score)
            detailed_results: 详细评估结果 (Detailed evaluation results)
            action_memory: 动作记忆 (Action memory)
            
        Returns:
            构建的提示 (Built prompt)
        """
        # 格式化模板部分
        # Format template sections
        sections_str = self._format_template_sections(template)
        
        # 格式化评估结果
        # Format evaluation results
        results_str = self._format_evaluation_results(detailed_results)
        
        # 格式化批次示例
        # Format batch examples
        batch_examples = self._format_batch_examples(batch)
        
        # 格式化历史记录
        # Format history
        history_str = action_memory.format_for_prompt(
            max_records=self.max_history_items,
            format_type='detailed' if not self.compact_mode else 'condensed'
        )
        
        # 使用模板填充提示
        # Fill prompt using template
        prompt = self._templates["action_generation"].format(
            sections_str=sections_str,
            current_score=current_score,
            results_str=results_str,
            batch_examples=batch_examples,
            history_str=history_str
        )
        
        return prompt
    
    def build_action_evaluation_prompt(self,
                                      template: BasePromptTemplate,
                                      action: StructuredAction,
                                      result_template: BasePromptTemplate,
                                      batch: List[Dict[str, Any]],
                                      current_score: float,
                                      detailed_results: Dict[str, MetricResult],
                                      action_memory: ActionMemory) -> str:
        """
        构建动作评估提示。
        
        Build action evaluation prompt.
        
        Args:
            template: 当前模板 (Current template)
            action: 要评估的动作 (Action to evaluate)
            result_template: 动作应用后的模板 (Template after action application)
            batch: 数据批次 (Data batch)
            current_score: 当前分数 (Current score)
            detailed_results: 详细评估结果 (Detailed evaluation results)
            action_memory: 动作记忆 (Action memory)
            
        Returns:
            构建的提示 (Built prompt)
        """
        # 格式化模板部分
        # Format template sections
        sections_str = self._format_template_sections(template)
        
        # 格式化动作
        # Format action
        action_str = self._format_action(action)
        
        # 格式化结果模板部分
        # Format result template sections
        result_sections_str = self._format_template_sections(result_template)
        
        # 格式化评估结果
        # Format evaluation results
        results_str = self._format_evaluation_results(detailed_results)
        
        # 格式化批次示例
        # Format batch examples
        batch_examples = self._format_batch_examples(batch)
        
        # 格式化历史记录
        # Format history
        history_str = action_memory.format_for_prompt(
            max_records=self.max_history_items,
            format_type='condensed'
        )
        
        # 使用模板填充提示
        # Fill prompt using template
        prompt = self._templates["action_evaluation"].format(
            sections_str=sections_str,
            action_str=action_str,
            result_sections_str=result_sections_str,
            current_score=current_score,
            results_str=results_str,
            batch_examples=batch_examples,
            history_str=history_str
        )
        
        return prompt
    
    def build_multi_action_generation_prompt(self,
                                           template: BasePromptTemplate,
                                           batch: List[Dict[str, Any]],
                                           current_score: float,
                                           detailed_results: Dict[str, MetricResult],
                                           action_memory: ActionMemory,
                                           num_actions: int = 3) -> str:
        """
        构建多动作生成提示。
        
        Build multi-action generation prompt.
        
        Args:
            template: 当前模板 (Current template)
            batch: 数据批次 (Data batch)
            current_score: 当前分数 (Current score)
            detailed_results: 详细评估结果 (Detailed evaluation results)
            action_memory: 动作记忆 (Action memory)
            num_actions: 要生成的动作数量 (Number of actions to generate)
            
        Returns:
            构建的提示 (Built prompt)
        """
        # 格式化模板部分
        # Format template sections
        sections_str = self._format_template_sections(template)
        
        # 格式化评估结果
        # Format evaluation results
        results_str = self._format_evaluation_results(detailed_results)
        
        # 格式化批次示例
        # Format batch examples
        batch_examples = self._format_batch_examples(batch)
        
        # 格式化历史记录
        # Format history
        history_str = action_memory.format_for_prompt(
            max_records=self.max_history_items,
            format_type='condensed'
        )
        
        # 使用模板填充提示
        # Fill prompt using template
        prompt = self._templates["multi_action_generation"].format(
            sections_str=sections_str,
            current_score=current_score,
            results_str=results_str,
            batch_examples=batch_examples,
            history_str=history_str,
            num_actions=num_actions
        )
        
        return prompt
    
    def _format_template_sections(self, template: BasePromptTemplate) -> str:
        """
        格式化模板部分。
        
        Format template sections.
        
        Args:
            template: 要格式化的模板 (Template to format)
            
        Returns:
            格式化的模板部分 (Formatted template sections)
        """
        sections_str = ""
        for section_name in template.list_sections():
            content = template.get_section(section_name)
            
            # 如果内容过长，截断
            # If content is too long, truncate
            if content and len(content) > self.max_section_length:
                content = content[:self.max_section_length - 3] + "..."
                
            sections_str += f"--- {section_name} ---\n{content or '(Empty)'}\n\n"
            
        return sections_str.strip()
    
    def _format_evaluation_results(self, detailed_results: Dict[str, MetricResult]) -> str:
        """
        格式化评估结果。
        
        Format evaluation results.
        
        Args:
            detailed_results: 详细评估结果 (Detailed evaluation results)
            
        Returns:
            格式化的评估结果 (Formatted evaluation results)
        """
        if not self.include_metrics:
            return ""
            
        results_str = ""
        for metric_name, result in detailed_results.items():
            results_str += f"{metric_name}: {result.score:.4f}\n"
            
            # 如果是紧凑模式，跳过详情
            # If compact mode, skip details
            if self.compact_mode:
                continue
                
            if result.details:
                details_str = str(result.details)
                if len(details_str) > 200:
                    details_str = details_str[:197] + "..."
                results_str += f"Details: {details_str}\n"
                
        return results_str.strip()
    
    def _format_batch_examples(self, batch: List[Dict[str, Any]]) -> str:
        """
        格式化批次示例。
        
        Format batch examples.
        
        Args:
            batch: 数据批次 (Data batch)
            
        Returns:
            格式化的批次示例 (Formatted batch examples)
        """
        batch_examples = ""
        for i, example in enumerate(batch[:self.max_batch_examples]):
            if self.compact_mode:
                input_text = example.get("input", "")
                reference = example.get("reference", "")
                if len(input_text) > 100:
                    input_text = input_text[:97] + "..."
                if len(reference) > 100:
                    reference = reference[:97] + "..."
                batch_examples += f"Ex{i+1}: I={input_text} | R={reference}\n"
            else:
                input_text = example.get("input", "(No input)")
                reference = example.get("reference", "(No reference)")
                batch_examples += f"Example {i+1}:\nInput: {input_text}\nReference: {reference}\n\n"
                
        return batch_examples.strip()
    
    def _format_action(self, action: StructuredAction) -> str:
        """
        格式化动作。
        
        Format action.
        
        Args:
            action: 要格式化的动作 (Action to format)
            
        Returns:
            格式化的动作 (Formatted action)
        """
        action_dict = action.to_dict()
        
        # 如果内容过长，截断
        # If content is too long, truncate
        if action_dict.get("content") and len(action_dict["content"]) > self.max_section_length:
            action_dict["content"] = action_dict["content"][:self.max_section_length - 3] + "..."
            
        return f"""Target Section: {action_dict.get('target_section', 'N/A')}
Action Type: {action_dict.get('action_type', 'N/A')}
Content: {action_dict.get('content', '(No content)')}"""
    
    def customize_template(self, template_name: str, template_text: str) -> None:
        """
        自定义提示模板。
        
        Customize prompt template.
        
        Args:
            template_name: 模板名称 (Template name)
            template_text: 模板文本 (Template text)
        """
        self._templates[template_name] = template_text
        logger.info(f"Customized template '{template_name}'")
    
    def get_all_template_names(self) -> List[str]:
        """
        获取所有模板名称。
        
        Get all template names.
        
        Returns:
            模板名称列表 (List of template names)
        """
        return list(self._templates.keys())
    
    def get_template(self, template_name: str) -> str:
        """
        获取指定模板的文本。
        
        Get text of specified template.
        
        Args:
            template_name: 模板名称 (Template name)
            
        Returns:
            模板文本 (Template text)
            
        Raises:
            KeyError: 如果模板不存在 (If template doesn't exist)
        """
        if template_name not in self._templates:
            raise KeyError(f"Template '{template_name}' not found")
            
        return self._templates[template_name]


class TokenBudgetManager:
    """
    代币预算管理器，用于优化提示长度以适应LLM上下文窗口。
    
    Token budget manager for optimizing prompt length to fit LLM context window.
    """
    
    def __init__(self, 
                max_tokens: int = 4000,
                token_estimator: Optional[Callable[[str], int]] = None):
        """
        初始化代币预算管理器。
        
        Initialize token budget manager.
        
        Args:
            max_tokens: 最大代币数 (Maximum token count)
            token_estimator: 代币数估算函数 (Token estimation function)
        """
        self.max_tokens = max_tokens
        self.token_estimator = token_estimator or self._default_token_estimator
        
    def _default_token_estimator(self, text: str) -> int:
        """
        默认代币数估算器（基于简单启发式）。
        
        Default token estimator (based on simple heuristics).
        
        Args:
            text: 要估算的文本 (Text to estimate)
            
        Returns:
            估算的代币数 (Estimated token count)
        """
        # 非常粗略的估算：平均每4个字符1个代币
        # Very rough estimation: average of 1 token per 4 characters
        return len(text) // 4 + 1
    
    def allocate_tokens(self, components: Dict[str, str], 
                       weights: Dict[str, float]) -> Dict[str, str]:
        """
        根据权重在组件间分配代币预算。
        
        Allocate token budget among components according to weights.
        
        Args:
            components: 提示组件字典 (Dictionary of prompt components)
            weights: 组件权重字典 (Dictionary of component weights)
            
        Returns:
            调整后的组件字典 (Dictionary of adjusted components)
        """
        # 估算当前代币总数
        # Estimate current total token count
        current_tokens = sum(self.token_estimator(text) for text in components.values())
        
        # 如果总数已经在预算内，无需调整
        # If total is already within budget, no adjustment needed
        if current_tokens <= self.max_tokens:
            return components.copy()
        
        # 计算总权重
        # Calculate total weight
        total_weight = sum(weights.get(key, 1.0) for key in components)
        
        # 计算每个组件的目标代币数
        # Calculate target token count for each component
        target_tokens = {}
        for key in components:
            component_weight = weights.get(key, 1.0)
            target_tokens[key] = int(self.max_tokens * (component_weight / total_weight))
        
        # 调整组件
        # Adjust components
        adjusted_components = {}
        for key, text in components.items():
            current_component_tokens = self.token_estimator(text)
            
            if current_component_tokens <= target_tokens[key]:
                # 该组件已经在目标代币数内
                # This component is already within target token count
                adjusted_components[key] = text
            else:
                # 需要截断该组件
                # Need to truncate this component
                adjusted_components[key] = self._truncate_text(text, target_tokens[key])
                
        return adjusted_components
    
    def _truncate_text(self, text: str, target_tokens: int) -> str:
        """
        截断文本以适应目标代币数。
        
        Truncate text to fit target token count.
        
        Args:
            text: 要截断的文本 (Text to truncate)
            target_tokens: 目标代币数 (Target token count)
            
        Returns:
            截断后的文本 (Truncated text)
        """
        # 估算文本的代币数
        # Estimate token count of text
        current_tokens = self.token_estimator(text)
        
        if current_tokens <= target_tokens:
            return text
            
        # 估算每个字符的代币数
        # Estimate tokens per character
        tokens_per_char = current_tokens / len(text)
        
        # 估算目标字符数
        # Estimate target character count
        target_chars = int(target_tokens / tokens_per_char)
        
        # 简单截断，保留开头部分
        # Simple truncation, keep beginning part
        return text[:target_chars - 3] + "..."