# ptforge/core/reward_calculator.py
"""
计算提示词优化动作的奖励。
提供多种奖励计算策略，支持加权组合和归一化。

Calculates rewards for prompt optimization actions.
Provides multiple reward calculation strategies with support for weighted combinations and normalization.
"""

import math
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

from ptforge.templates.base_template import BasePromptTemplate
from ptforge.templates.action_space import StructuredAction
from ptforge.core.base import MetricResult

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    奖励计算器基类。
    定义奖励计算的基本接口。
    
    Base reward calculator class.
    Defines the basic interface for reward calculation.
    """
    
    def calculate(self, 
                before_score: float, 
                after_score: float, 
                action: Optional[StructuredAction] = None,
                context: Optional[Dict[str, Any]] = None) -> float:
        """
        计算动作的奖励。
        
        Calculate reward for an action.
        
        Args:
            before_score: 执行动作前的分数 (Score before action execution)
            after_score: 执行动作后的分数 (Score after action execution)
            action: 执行的动作 (Action executed)
            context: 计算上下文 (Calculation context)
            
        Returns:
            计算的奖励值 (Calculated reward value)
        """
        raise NotImplementedError("Subclasses must implement calculate()")


class RelativeImprovementReward(RewardCalculator):
    """
    基于相对改进百分比的奖励计算器。
    
    Reward calculator based on relative improvement percentage.
    """
    
    def __init__(self, 
                scale_factor: float = 100.0, 
                min_reward: float = -25.0, 
                max_reward: float = 100.0):
        """
        初始化相对改进奖励计算器。
        
        Initialize relative improvement reward calculator.
        
        Args:
            scale_factor: 缩放因子，用于调整奖励幅度 (Scale factor for adjusting reward magnitude)
            min_reward: 最小奖励值 (Minimum reward value)
            max_reward: 最大奖励值 (Maximum reward value)
        """
        self.scale_factor = scale_factor
        self.min_reward = min_reward
        self.max_reward = max_reward
    
    def calculate(self, 
                before_score: float, 
                after_score: float, 
                action: Optional[StructuredAction] = None,
                context: Optional[Dict[str, Any]] = None) -> float:
        """
        计算相对改进奖励。
        奖励 = (after_score - before_score) / before_score * scale_factor
        
        Calculate relative improvement reward.
        Reward = (after_score - before_score) / before_score * scale_factor
        
        Args:
            before_score: 执行动作前的分数 (Score before action execution)
            after_score: 执行动作后的分数 (Score after action execution)
            action: 执行的动作 (Action executed)
            context: 计算上下文 (Calculation context)
            
        Returns:
            相对改进百分比作为奖励 (Relative improvement percentage as reward)
        """
        # 处理零分或接近零分的情况
        # Handle zero or near-zero score cases
        if abs(before_score) < 1e-6:
            if after_score > 0:
                # 从零变为正数，给予正奖励
                # From zero to positive, give positive reward
                return min(self.max_reward, self.scale_factor * 0.5)
            elif after_score == 0:
                # 零到零，无变化
                # Zero to zero, no change
                return 0.0
            else:
                # 从零变为负数，给予负奖励
                # From zero to negative, give negative reward
                return max(self.min_reward, -self.scale_factor * 0.5)
        
        # 计算相对变化百分比
        # Calculate relative change percentage
        relative_change = (after_score - before_score) / abs(before_score)
        reward = relative_change * self.scale_factor
        
        # 限制奖励范围
        # Limit reward range
        return max(self.min_reward, min(self.max_reward, reward))


class AbsoluteChangeReward(RewardCalculator):
    """
    基于绝对变化的奖励计算器。
    
    Reward calculator based on absolute change.
    """
    
    def __init__(self, 
                scale_factor: float = 100.0, 
                min_reward: float = -25.0, 
                max_reward: float = 100.0):
        """
        初始化绝对变化奖励计算器。
        
        Initialize absolute change reward calculator.
        
        Args:
            scale_factor: 缩放因子，用于调整奖励幅度 (Scale factor for adjusting reward magnitude)
            min_reward: 最小奖励值 (Minimum reward value)
            max_reward: 最大奖励值 (Maximum reward value)
        """
        self.scale_factor = scale_factor
        self.min_reward = min_reward
        self.max_reward = max_reward
    
    def calculate(self, 
                before_score: float, 
                after_score: float, 
                action: Optional[StructuredAction] = None,
                context: Optional[Dict[str, Any]] = None) -> float:
        """
        计算绝对变化奖励。
        奖励 = (after_score - before_score) * scale_factor
        
        Calculate absolute change reward.
        Reward = (after_score - before_score) * scale_factor
        
        Args:
            before_score: 执行动作前的分数 (Score before action execution)
            after_score: 执行动作后的分数 (Score after action execution)
            action: 执行的动作 (Action executed)
            context: 计算上下文 (Calculation context)
            
        Returns:
            缩放后的绝对变化作为奖励 (Scaled absolute change as reward)
        """
        absolute_change = after_score - before_score
        reward = absolute_change * self.scale_factor
        
        # 限制奖励范围
        # Limit reward range
        return max(self.min_reward, min(self.max_reward, reward))


class ThresholdReward(RewardCalculator):
    """
    基于阈值的奖励计算器。
    只有当改进超过一定阈值时才给予奖励。
    
    Threshold-based reward calculator.
    Gives reward only when improvement exceeds a certain threshold.
    """
    
    def __init__(self, 
                threshold: float = 0.01, 
                positive_reward: float = 1.0, 
                negative_reward: float = -0.5,
                neutral_reward: float = 0.0):
        """
        初始化阈值奖励计算器。
        
        Initialize threshold reward calculator.
        
        Args:
            threshold: 改进必须超过的阈值 (Threshold that improvement must exceed)
            positive_reward: 超过阈值时的奖励 (Reward when exceeding threshold)
            negative_reward: 低于负阈值时的奖励 (Reward when below negative threshold)
            neutral_reward: 在阈值范围内的奖励 (Reward within threshold range)
        """
        self.threshold = threshold
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.neutral_reward = neutral_reward
    
    def calculate(self, 
                before_score: float, 
                after_score: float, 
                action: Optional[StructuredAction] = None,
                context: Optional[Dict[str, Any]] = None) -> float:
        """
        计算阈值奖励。
        
        Calculate threshold reward.
        
        Args:
            before_score: 执行动作前的分数 (Score before action execution)
            after_score: 执行动作后的分数 (Score after action execution)
            action: 执行的动作 (Action executed)
            context: 计算上下文 (Calculation context)
            
        Returns:
            基于阈值的奖励 (Threshold-based reward)
        """
        if before_score == 0:
            relative_change = 1.0 if after_score > 0 else 0.0
        else:
            relative_change = (after_score - before_score) / abs(before_score)
            
        if relative_change >= self.threshold:
            return self.positive_reward
        elif relative_change <= -self.threshold:
            return self.negative_reward
        else:
            return self.neutral_reward


class CompoundReward(RewardCalculator):
    """
    复合奖励计算器，组合多个奖励计算器。
    
    Compound reward calculator that combines multiple reward calculators.
    """
    
    def __init__(self, calculators: List[Tuple[RewardCalculator, float]]):
        """
        初始化复合奖励计算器。
        
        Initialize compound reward calculator.
        
        Args:
            calculators: 计算器和权重的列表 [(calculator, weight), ...] 
                         (List of calculators and weights [(calculator, weight), ...])
        """
        self.calculators = calculators
    
    def calculate(self, 
                before_score: float, 
                after_score: float, 
                action: Optional[StructuredAction] = None,
                context: Optional[Dict[str, Any]] = None) -> float:
        """
        计算复合奖励。
        
        Calculate compound reward.
        
        Args:
            before_score: 执行动作前的分数 (Score before action execution)
            after_score: 执行动作后的分数 (Score after action execution)
            action: 执行的动作 (Action executed)
            context: 计算上下文 (Calculation context)
            
        Returns:
            复合奖励 (Compound reward)
        """
        total_reward = 0.0
        total_weight = 0.0
        
        for calculator, weight in self.calculators:
            reward = calculator.calculate(before_score, after_score, action, context)
            total_reward += reward * weight
            total_weight += weight
            
        # 归一化
        # Normalize
        if total_weight > 0:
            return total_reward / total_weight
        else:
            return 0.0


class EnhancedRewardCalculator(RewardCalculator):
    """
    增强型奖励计算器，考虑多种因素。
    
    Enhanced reward calculator that considers multiple factors.
    """
    
    def __init__(self, 
                base_scale: float = 100.0,
                exploration_bonus: float = 10.0,
                complexity_penalty: float = 5.0,
                global_best_bonus: float = 25.0,
                local_improvement_scale: float = 100.0):
        """
        初始化增强型奖励计算器。
        
        Initialize enhanced reward calculator.
        
        Args:
            base_scale: 基础奖励缩放因子 (Base reward scaling factor)
            exploration_bonus: 探索奖励 (Exploration bonus)
            complexity_penalty: 复杂度惩罚系数 (Complexity penalty coefficient)
            global_best_bonus: 全局最佳奖励 (Global best bonus)
            local_improvement_scale: 局部改进缩放因子 (Local improvement scaling factor)
        """
        self.base_scale = base_scale
        self.exploration_bonus = exploration_bonus
        self.complexity_penalty = complexity_penalty
        self.global_best_bonus = global_best_bonus
        self.local_improvement_scale = local_improvement_scale
    
    def calculate(self, 
                before_score: float, 
                after_score: float, 
                action: Optional[StructuredAction] = None,
                context: Optional[Dict[str, Any]] = None) -> float:
        """
        计算增强型奖励。
        
        Calculate enhanced reward.
        
        Args:
            before_score: 执行动作前的分数 (Score before action execution)
            after_score: 执行动作后的分数 (Score after action execution)
            action: 执行的动作 (Action executed)
            context: 计算上下文 (Calculation context)
            
        Returns:
            增强型奖励 (Enhanced reward)
        """
        context = context or {}
        
        # 1. 计算基础相对改进奖励
        # 1. Calculate base relative improvement reward
        if abs(before_score) < 1e-6:
            base_reward = self.base_scale * after_score
        else:
            relative_change = (after_score - before_score) / abs(before_score)
            base_reward = relative_change * self.base_scale
        
        # 2. 探索奖励：如果动作类型或目标部分是新的
        # 2. Exploration bonus: if action type or target section is new
        exploration_reward = 0.0
        if action and 'explored_actions' in context:
            explored_actions = context['explored_actions']
            action_key = (action.action_type.value, action.target_section)
            
            if action_key not in explored_actions:
                exploration_reward = self.exploration_bonus
                explored_actions.add(action_key)
        
        # 3. 复杂度惩罚：针对过于复杂的内容
        # 3. Complexity penalty: for overly complex content
        complexity_penalty = 0.0
        if action and action.content:
            # 简单的复杂度衡量：内容长度和结构复杂性
            # Simple complexity measure: content length and structural complexity
            content_length = len(action.content)
            structure_complexity = action.content.count('\n') + action.content.count('{{')
            
            # 综合复杂度分数
            # Combined complexity score
            complexity_score = (content_length / 1000) + (structure_complexity / 10)
            complexity_penalty = self.complexity_penalty * complexity_score
        
        # 4. 全局最佳奖励：如果超过了历史最佳分数
        # 4. Global best bonus: if exceeding historical best score
        global_best_reward = 0.0
        if 'global_best_score' in context:
            global_best_score = context['global_best_score']
            if after_score > global_best_score:
                global_best_reward = self.global_best_bonus
                context['global_best_score'] = after_score
        
        # 5. 局部改进奖励：直接奖励分数提升
        # 5. Local improvement reward: directly reward score increase
        local_improvement = (after_score - before_score) * self.local_improvement_scale
        
        # 计算总奖励
        # Calculate total reward
        total_reward = base_reward + exploration_reward - complexity_penalty + global_best_reward + local_improvement
        
        logger.debug(f"Reward breakdown: base={base_reward:.2f}, exploration={exploration_reward:.2f}, " +
                    f"complexity_penalty={complexity_penalty:.2f}, global_best={global_best_reward:.2f}, " +
                    f"local_improvement={local_improvement:.2f}, total={total_reward:.2f}")
        
        return total_reward


def create_default_reward_calculator() -> RewardCalculator:
    """
    创建默认的奖励计算器。
    
    Create default reward calculator.
    
    Returns:
        默认的奖励计算器 (Default reward calculator)
    """
    # 创建一个复合奖励计算器，包含相对改进和阈值奖励
    # Create a compound reward calculator with relative improvement and threshold rewards
    relative_reward = RelativeImprovementReward(scale_factor=100.0)
    threshold_reward = ThresholdReward(threshold=0.01, positive_reward=10.0, negative_reward=-5.0)
    
    return CompoundReward([
        (relative_reward, 0.7),
        (threshold_reward, 0.3)
    ])