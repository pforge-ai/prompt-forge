# ptforge/templates/action_space.py
"""
定义提示词优化的结构化动作空间。
为优化过程提供清晰、一致的动作表示。

Defines the structured action space for prompt optimization.
Provides clear, consistent action representations for the optimization process.
"""

import random
import logging
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Set, Union, Tuple

from ptforge.templates.base_template import BasePromptTemplate

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """
    定义可能的动作类型。
    
    Defines possible action types.
    """
    REPLACE = "REPLACE"       # 完全替换部分内容 (Completely replace section content)
    MODIFY = "MODIFY"         # 修改部分内容 (Modify section content)
    EXPAND = "EXPAND"         # 扩展部分内容 (Expand section content)
    SIMPLIFY = "SIMPLIFY"     # 简化部分内容 (Simplify section content)
    RESTRUCTURE = "RESTRUCTURE"  # 重构部分内容 (Restructure section content)
    
    @classmethod
    def from_string(cls, value: str) -> 'ActionType':
        """
        从字符串转换为ActionType枚举。
        
        Convert from string to ActionType enum.
        
        Args:
            value: 动作类型字符串 (Action type string)
            
        Returns:
            对应的ActionType枚举 (Corresponding ActionType enum)
            
        Raises:
            ValueError: 如果字符串不匹配任何ActionType (If string doesn't match any ActionType)
        """
        try:
            return cls(value.upper())
        except ValueError:
            # 尝试从部分匹配或首字母获取
            # Try to get from partial match or first letter
            value_upper = value.upper()
            for action_type in cls:
                if action_type.value.startswith(value_upper) or action_type.value[0] == value_upper:
                    return action_type
            
            # 默认返回MODIFY
            # Default to MODIFY
            logger.warning(f"Unknown action type: {value}, defaulting to MODIFY")
            return cls.MODIFY


class StructuredAction:
    """
    结构化表示提示词修改动作。
    
    Structured representation of a prompt modification action.
    """
    
    def __init__(self, 
                 action_type: Union[str, ActionType], 
                 target_section: str, 
                 content: Optional[str] = None, 
                 parameters: Optional[Dict[str, Any]] = None,
                 old_content: Optional[str] = None):
        """
        初始化结构化动作。
        
        Initialize a structured action.
        
        Args:
            action_type: 动作类型 (Action type)
            target_section: 目标部分名称 (Target section name)
            content: 新内容(如适用) (New content, if applicable)
            parameters: 额外参数 (Additional parameters)
            old_content: 旧内容，用于记录 (Old content, for record keeping)
        """
        # 转换字符串到枚举
        # Convert string to enum
        if isinstance(action_type, str):
            self.action_type = ActionType.from_string(action_type)
        else:
            self.action_type = action_type
            
        self.target_section = target_section
        self.content = content
        self.parameters = parameters or {}
        self.old_content = old_content
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将动作转换为字典表示。
        
        Convert action to dictionary representation.
        
        Returns:
            动作的字典表示 (Dictionary representation of action)
        """
        return {
            "action_type": self.action_type.value,
            "target_section": self.target_section,
            "content": self.content,
            "parameters": self.parameters,
            "old_content": self.old_content
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuredAction':
        """
        从字典创建动作。
        
        Create action from dictionary.
        
        Args:
            data: 动作字典 (Action dictionary)
            
        Returns:
            创建的StructuredAction实例 (Created StructuredAction instance)
        """
        return cls(
            action_type=data["action_type"],
            target_section=data["target_section"],
            content=data.get("content"),
            parameters=data.get("parameters", {}),
            old_content=data.get("old_content")
        )
    
    def describe(self, include_content: bool = False) -> str:
        """
        生成动作的可读描述。
        
        Generate a readable description of the action.
        
        Args:
            include_content: 是否包含内容 (Whether to include content)
            
        Returns:
            动作描述 (Action description)
        """
        desc = f"{self.action_type.value} {self.target_section}"
        
        if self.parameters:
            param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
            desc += f" with {param_str}"
            
        if include_content and self.content:
            # 限制内容长度
            # Limit content length
            content_preview = (self.content[:50] + "...") if len(self.content) > 50 else self.content
            desc += f": {content_preview}"
            
        return desc
    
    def __str__(self) -> str:
        """
        返回动作的字符串表示。
        
        Return string representation of action.
        
        Returns:
            动作的字符串表示 (String representation of action)
        """
        return self.describe(include_content=False)
    
    def __repr__(self) -> str:
        """
        返回动作的详细字符串表示。
        
        Return detailed string representation of action.
        
        Returns:
            动作的详细字符串表示 (Detailed string representation of action)
        """
        return f"StructuredAction({self.to_dict()})"


class ActionSpace:
    """
    定义和管理动作空间。
    提供动作采样和生成功能。
    
    Defines and manages the action space.
    Provides action sampling and generation capabilities.
    """
    
    def __init__(self, template: BasePromptTemplate, restrict_to_optimizable: bool = True):
        """
        初始化动作空间。
        
        Initialize action space.
        
        Args:
            template: 提示词模板 (Prompt template)
            restrict_to_optimizable: 是否限制为可优化部分 (Whether to restrict to optimizable sections)
        """
        self.template = template
        self.restrict_to_optimizable = restrict_to_optimizable
        
        # 获取可用部分
        # Get available sections
        if restrict_to_optimizable:
            self.available_sections = set(template.get_optimizable_sections().keys())
        else:
            self.available_sections = set(template.list_sections())
        
        logger.debug(f"Initialized ActionSpace with {len(self.available_sections)} available sections")
    
    def get_available_actions(self) -> List[Tuple[ActionType, str]]:
        """
        获取所有可用的动作类型和目标部分组合。
        
        Get all available action type and target section combinations.
        
        Returns:
            (动作类型, 目标部分)元组的列表 (List of (action_type, target_section) tuples)
        """
        actions = []
        for section in self.available_sections:
            current_content = self.template.get_section(section)
            
            # 根据当前内容状态确定可用动作类型
            # Determine available action types based on current content state
            if current_content:  # 如果部分有内容 (If section has content)
                for action_type in ActionType:
                    actions.append((action_type, section))
            else:  # 如果部分为空 (If section is empty)
                # 只能替换或扩展空部分
                # Can only replace or expand empty sections
                actions.append((ActionType.REPLACE, section))
                actions.append((ActionType.EXPAND, section))
        
        return actions
    
    def sample(self) -> StructuredAction:
        """
        随机采样一个动作。
        
        Randomly sample an action.
        
        Returns:
            采样的动作 (Sampled action)
            
        Raises:
            ValueError: 如果没有可用动作 (If no actions are available)
        """
        available_actions = self.get_available_actions()
        if not available_actions:
            raise ValueError("No available actions in this action space")
        
        action_type, target_section = random.choice(available_actions)
        
        # 获取当前内容作为参考
        # Get current content for reference
        current_content = self.template.get_section(target_section)
        
        # 根据动作类型准备参数
        # Prepare parameters based on action type
        parameters = {}
        content = None
        
        # 在实际实现中，这里会有根据动作类型生成内容的逻辑
        # In a real implementation, logic for generating content based on action type would go here
        
        return StructuredAction(
            action_type=action_type,
            target_section=target_section,
            content=content,
            parameters=parameters,
            old_content=current_content
        )
    
    def create_action(self, 
                     action_type: Union[str, ActionType], 
                     target_section: str, 
                     content: Optional[str] = None, 
                     **parameters) -> StructuredAction:
        """
        创建一个动作。
        
        Create an action.
        
        Args:
            action_type: 动作类型 (Action type)
            target_section: 目标部分 (Target section)
            content: 新内容 (New content)
            **parameters: 额外参数 (Additional parameters)
            
        Returns:
            创建的动作 (Created action)
            
        Raises:
            ValueError: 如果目标部分不可用 (If target section is not available)
        """
        if target_section not in self.available_sections:
            raise ValueError(f"Section '{target_section}' is not available for actions")
        
        # 获取当前内容作为参考
        # Get current content for reference
        current_content = self.template.get_section(target_section)
        
        return StructuredAction(
            action_type=action_type,
            target_section=target_section,
            content=content,
            parameters=parameters,
            old_content=current_content
        )
    
    def create_from_dict(self, action_dict: Dict[str, Any]) -> StructuredAction:
        """
        从字典创建动作，同时验证动作有效性。
        
        Create action from dictionary and validate action validity.
        
        Args:
            action_dict: 动作字典 (Action dictionary)
            
        Returns:
            创建的动作 (Created action)
            
        Raises:
            ValueError: 如果动作无效 (If action is invalid)
        """
        if "target_section" not in action_dict:
            raise ValueError("Action dictionary must contain 'target_section'")
            
        target_section = action_dict["target_section"]
        if target_section not in self.available_sections:
            raise ValueError(f"Section '{target_section}' is not available for actions")
        
        action_type = action_dict.get("action_type", "MODIFY")
        content = action_dict.get("content")
        parameters = action_dict.get("parameters", {})
        
        # 获取当前内容
        # Get current content
        current_content = self.template.get_section(target_section)
        
        return StructuredAction(
            action_type=action_type,
            target_section=target_section,
            content=content,
            parameters=parameters,
            old_content=current_content
        )
    
    def apply_action(self, action: StructuredAction) -> BasePromptTemplate:
        """
        应用动作到模板，创建新模板。
        
        Apply action to template, creating a new template.
        
        Args:
            action: 要应用的动作 (Action to apply)
            
        Returns:
            更新后的模板副本 (Updated template copy)
            
        Raises:
            ValueError: 如果无法应用动作 (If action cannot be applied)
        """
        # 创建模板副本
        # Create copy of template
        new_template = self.template.__class__.__new__(self.template.__class__)
        new_template.__dict__ = self.template.__dict__.copy()
        
        # 从动作中获取信息
        # Get information from action
        section = action.target_section
        action_type = action.action_type
        new_content = action.content
        parameters = action.parameters
        
        if section not in self.available_sections:
            raise ValueError(f"Section '{section}' is not available for actions")
        
        # 获取当前内容
        # Get current content
        current_content = self.template.get_section(section)
        
        # 根据动作类型处理内容
        # Process content based on action type
        updated_content = None
        
        if action_type == ActionType.REPLACE:
            # 直接替换内容
            # Directly replace content
            if new_content is not None:
                updated_content = new_content
            else:
                raise ValueError("REPLACE action requires content")
                
        elif action_type == ActionType.MODIFY:
            # 修改内容 (简单实现，实际应用中可能更复杂)
            # Modify content (simple implementation, might be more complex in practice)
            if current_content is None:
                raise ValueError("Cannot MODIFY empty section")
                
            if new_content is not None:
                updated_content = new_content
            else:
                raise ValueError("MODIFY action requires content")
                
        elif action_type == ActionType.EXPAND:
            # 扩展内容
            # Expand content
            if current_content and new_content:
                updated_content = f"{current_content}\n\n{new_content}"
            elif new_content:
                updated_content = new_content
            else:
                raise ValueError("EXPAND action requires content")
                
        elif action_type == ActionType.SIMPLIFY:
            # 简化内容 (这里需要实际的简化逻辑)
            # Simplify content (actual simplification logic needed here)
            if current_content is None:
                raise ValueError("Cannot SIMPLIFY empty section")
                
            if new_content is not None:
                updated_content = new_content
            else:
                # 简单实现：保留前半部分
                # Simple implementation: keep first half
                updated_content = current_content.split("\n\n")[0] if "\n\n" in current_content else current_content
                
        elif action_type == ActionType.RESTRUCTURE:
            # 重构内容
            # Restructure content
            if current_content is None:
                raise ValueError("Cannot RESTRUCTURE empty section")
                
            if new_content is not None:
                updated_content = new_content
            else:
                raise ValueError("RESTRUCTURE action requires content")
        
        # 更新模板
        # Update template
        if updated_content is not None:
            new_template.update_section(section, updated_content)
        
        return new_template