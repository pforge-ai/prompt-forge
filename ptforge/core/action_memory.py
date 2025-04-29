# ptforge/core/action_memory.py
"""
提供动作-奖励记忆管理。
用于记录历史动作及其奖励，并支持时间衰减权重。

Provides action-reward memory management.
Used for recording historical actions and their rewards, with support for time-decaying weights.
"""

import math
import logging
from typing import Any, Dict, List, Tuple, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class ActionRecord:
    """
    记录单个优化动作及其效果。
    
    Records a single optimization action and its effect.
    """
    
    def __init__(self, 
                 action: Dict[str, Any], 
                 reward: float, 
                 timestamp: Optional[datetime] = None,
                 before_score: Optional[float] = None,
                 after_score: Optional[float] = None,
                 step: Optional[int] = None):
        """
        初始化动作记录。
        
        Initialize an action record.
        
        Args:
            action: 执行的动作详情 (Details of the action executed)
            reward: 执行此动作获得的奖励 (Reward received for this action)
            timestamp: 动作执行的时间戳 (Timestamp when the action was executed)
            before_score: 执行前的分数 (Score before executing the action)
            after_score: 执行后的分数 (Score after executing the action)
            step: 在优化过程中的步数 (Step number in the optimization process)
        """
        self.action = action
        self.reward = reward
        self.timestamp = timestamp or datetime.now()
        self.before_score = before_score
        self.after_score = after_score
        self.step = step
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将记录转换为字典。
        
        Convert the record to a dictionary.
        
        Returns:
            包含所有记录字段的字典 (Dictionary containing all record fields)
        """
        return {
            "action": self.action,
            "reward": self.reward,
            "timestamp": self.timestamp.isoformat(),
            "before_score": self.before_score,
            "after_score": self.after_score,
            "step": self.step
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionRecord':
        """
        从字典创建记录。
        
        Create a record from a dictionary.
        
        Args:
            data: 包含记录字段的字典 (Dictionary containing record fields)
            
        Returns:
            新创建的ActionRecord实例 (Newly created ActionRecord instance)
        """
        timestamp = datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp")
        return cls(
            action=data["action"],
            reward=data["reward"],
            timestamp=timestamp,
            before_score=data.get("before_score"),
            after_score=data.get("after_score"),
            step=data.get("step")
        )


class ActionMemory:
    """
    管理动作-奖励记忆，支持时间衰减权重和经验回放。
    
    Manages action-reward memory with support for time-decaying weights and experience replay.
    """
    
    def __init__(self, 
                 max_entries: int = 20, 
                 decay_factor: float = 0.9, 
                 min_weight: float = 0.1):
        """
        初始化动作记忆管理器。
        
        Initialize action memory manager.
        
        Args:
            max_entries: 保留的最大记录数 (Maximum number of records to keep)
            decay_factor: 时间衰减因子，控制过去经验影响减弱的速度 (Time decay factor, controls how quickly past experiences lose influence)
            min_weight: 任何记录的最小权重 (Minimum weight for any record)
        """
        self.memory: List[ActionRecord] = []
        self.max_entries = max_entries
        self.decay_factor = decay_factor
        self.min_weight = min_weight
        self.current_step = 0
    
    def add(self, 
            action: Dict[str, Any], 
            reward: float, 
            before_score: Optional[float] = None, 
            after_score: Optional[float] = None) -> None:
        """
        添加新的动作-奖励记录。
        
        Add a new action-reward record.
        
        Args:
            action: 执行的动作详情 (Details of the action executed)
            reward: 执行此动作获得的奖励 (Reward received for this action)
            before_score: 执行前的分数 (Score before executing the action)
            after_score: 执行后的分数 (Score after executing the action)
        """
        record = ActionRecord(
            action=action,
            reward=reward,
            timestamp=datetime.now(),
            before_score=before_score,
            after_score=after_score,
            step=self.current_step
        )
        
        self.memory.append(record)
        self.current_step += 1
        
        # 如果超出最大容量，移除最旧的记录
        # If exceeding max capacity, remove oldest records
        if len(self.memory) > self.max_entries:
            self.memory = self.memory[-self.max_entries:]
            
        logger.debug(f"Added action record with reward {reward:.4f}. Memory size: {len(self.memory)}")
    
    def get_weighted_experiences(self) -> List[Tuple[Dict[str, Any], float, float]]:
        """
        获取带时间衰减权重的经验记录。
        
        Get experience records with time-decaying weights.
        
        Returns:
            记录列表，每条包含(动作, 奖励, 权重) 
            (List of records, each containing (action, reward, weight))
        """
        if not self.memory:
            return []
        
        weighted_experiences = []
        latest_step = self.memory[-1].step if self.memory else 0
        
        for record in self.memory:
            # 计算基于步数差的衰减权重
            # Calculate decay weight based on step difference
            steps_ago = latest_step - record.step
            weight = max(self.min_weight, self.decay_factor ** steps_ago)
            
            weighted_experiences.append((record.action, record.reward, weight))
        
        return weighted_experiences
    
    def get_weighted_by_recency(self) -> List[Tuple[ActionRecord, float]]:
        """
        根据时间近远获取带权重的记录。
        最近的记录权重最高。
        
        Get weighted records based on recency.
        Most recent records have highest weight.
        
        Returns:
            记录列表，每条包含(记录, 权重) 
            (List of records, each containing (record, weight))
        """
        if not self.memory:
            return []
            
        records_with_weights = []
        latest_step = self.memory[-1].step if self.memory else 0
        
        for record in self.memory:
            steps_ago = latest_step - record.step
            weight = max(self.min_weight, self.decay_factor ** steps_ago)
            records_with_weights.append((record, weight))
        
        return records_with_weights
    
    def get_weighted_by_success(self) -> List[Tuple[ActionRecord, float]]:
        """
        根据成功程度获取带权重的记录。
        奖励高的记录权重更高。
        
        Get weighted records based on success.
        Records with higher rewards have higher weights.
        
        Returns:
            记录列表，每条包含(记录, 权重) 
            (List of records, each containing (record, weight))
        """
        if not self.memory:
            return []
            
        # 找出最大绝对奖励值用于归一化
        # Find max absolute reward for normalization
        max_abs_reward = max(abs(record.reward) for record in self.memory) if self.memory else 1.0
        if max_abs_reward < 1e-10:
            max_abs_reward = 1.0  # 避免除零
        
        records_with_weights = []
        for record in self.memory:
            # 根据奖励大小计算权重，结合时间衰减
            # Calculate weight based on reward magnitude, combined with time decay
            latest_step = self.memory[-1].step if self.memory else 0
            steps_ago = latest_step - record.step
            time_factor = max(self.min_weight, self.decay_factor ** steps_ago)
            
            # 对于正奖励，权重更高；负奖励的权重降低但保持记忆
            # Higher weights for positive rewards; reduced but preserved memory for negative rewards
            if record.reward > 0:
                success_factor = 0.5 + 0.5 * (record.reward / max_abs_reward)
            else:
                success_factor = 0.3 * (1 - abs(record.reward) / max_abs_reward)
                
            weight = time_factor * success_factor
            records_with_weights.append((record, weight))
        
        return records_with_weights
    
    def format_for_prompt(self, max_records: int = 5, format_type: str = 'detailed') -> str:
        """
        格式化记忆用于提示输入。
        
        Format memory for prompt input.
        
        Args:
            max_records: 包含的最大记录数 (Maximum number of records to include)
            format_type: 格式类型，'detailed'或'condensed' (Format type, 'detailed' or 'condensed')
            
        Returns:
            格式化的记忆字符串 (Formatted memory string)
        """
        if not self.memory:
            return "No previous optimization history."
        
        # 获取带权重的经验
        # Get weighted experiences
        records_with_weights = self.get_weighted_by_success()
        
        # 按权重排序（权重高的优先）
        # Sort by weight (higher weights first)
        records_with_weights.sort(key=lambda x: x[1], reverse=True)
        
        # 选择权重最高的n条记录
        # Select top n records by weight
        top_records = records_with_weights[:max_records]
        
        if format_type == 'condensed':
            # 极简格式，例如: "R:ROLE→+3.2% | M:TASK→-1.5% | R:CONSTRAINTS→+4.7%"
            # Ultra-condensed format, e.g.: "R:ROLE→+3.2% | M:TASK→-1.5% | R:CONSTRAINTS→+4.7%"
            condensed_items = []
            
            for record, _ in top_records:
                # 目标部分
                # Target section
                section = record.action.get("target_section", "?")
                
                # 动作类型首字母
                # First letter of action type
                action_code = record.action.get("action_type", "MOD")[0]
                
                # 奖励的简化表示(+/-符号和数值)
                # Simplified reward representation (+/- sign and value)
                reward_str = f"{'+' if record.reward >= 0 else ''}{record.reward:.1f}%"
                
                # 组合成极简表示
                # Combine into ultra-compact representation
                condensed_items.append(f"{action_code}:{section}→{reward_str}")
            
            return " | ".join(condensed_items)
            
        else:  # 详细格式 (detailed format)
            # 详细但紧凑的格式
            # Detailed but compact format
            formatted_lines = []
            
            for record, weight in top_records:
                action_type = record.action.get("action_type", "MODIFY")
                section = record.action.get("target_section", "unknown")
                reward_sign = "+" if record.reward >= 0 else ""
                
                # 创建简洁的动作描述
                # Create concise action description
                if "content" in record.action and record.action["content"] is not None and len(record.action["content"]) > 40:
                    content_summary = record.action["content"][:37] + "..."
                else:
                    content_summary = record.action.get("content", "(no content)")
                
                line = (f"{action_type} {section}: {reward_sign}{record.reward:.2f}% - "
                       f"Weight: {weight:.2f} - {content_summary}")
                formatted_lines.append(line)
            
            return "\n".join(formatted_lines)
    
    def clear(self) -> None:
        """
        清空记忆。
        
        Clear the memory.
        """
        self.memory = []
        logger.info("Action memory cleared")
    
    def get_top_effective_actions(self, n: int = 3) -> List[ActionRecord]:
        """
        获取最有效的n个动作。
        
        Get the n most effective actions.
        
        Args:
            n: 返回的动作数量 (Number of actions to return)
            
        Returns:
            最有效动作的列表 (List of most effective actions)
        """
        if not self.memory:
            return []
            
        # 按奖励降序排序
        # Sort by reward in descending order
        sorted_records = sorted(self.memory, key=lambda x: x.reward, reverse=True)
        return sorted_records[:n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取记忆统计信息。
        
        Get memory statistics.
        
        Returns:
            包含统计信息的字典 (Dictionary containing statistics)
        """
        if not self.memory:
            return {
                "count": 0,
                "avg_reward": 0.0,
                "positive_ratio": 0.0,
                "max_reward": 0.0,
                "min_reward": 0.0
            }
            
        rewards = [record.reward for record in self.memory]
        positive_count = sum(1 for r in rewards if r > 0)
        
        return {
            "count": len(self.memory),
            "avg_reward": sum(rewards) / len(rewards),
            "positive_ratio": positive_count / len(rewards) if rewards else 0,
            "max_reward": max(rewards) if rewards else 0,
            "min_reward": min(rewards) if rewards else 0
        }
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        将整个记忆转换为字典列表，用于序列化。
        
        Convert the entire memory to a list of dictionaries for serialization.
        
        Returns:
            记录字典的列表 (List of record dictionaries)
        """
        return [record.to_dict() for record in self.memory]
    
    @classmethod
    def from_dict_list(cls, data: List[Dict[str, Any]]) -> 'ActionMemory':
        """
        从字典列表恢复记忆，用于反序列化。
        
        Restore memory from a list of dictionaries for deserialization.
        
        Args:
            data: 记录字典的列表 (List of record dictionaries)
            
        Returns:
            新建的ActionMemory实例 (Newly created ActionMemory instance)
        """
        memory = cls()
        memory.memory = [ActionRecord.from_dict(item) for item in data]
        memory.current_step = max((r.step for r in memory.memory), default=0) + 1
        return memory