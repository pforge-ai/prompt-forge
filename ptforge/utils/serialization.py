# ptforge/utils/serialization.py
"""
提供实验配置和结果的序列化/反序列化支持。
方便保存、加载优化状态和结果。

Provides serialization/deserialization support for experiment configurations and results.
Facilitates saving and loading optimization states and results.
"""

import os
import json
import logging
import datetime
import pickle
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast
from pathlib import Path

from ptforge.core.config import OptimizationConfig
from ptforge.templates.base_template import BasePromptTemplate
from ptforge.core.action_memory import ActionMemory

logger = logging.getLogger(__name__)

# 定义类型变量
# Define type variables
T = TypeVar('T')


class SerializationError(Exception):
    """
    序列化/反序列化过程中发生的错误。
    
    Error that occurs during serialization/deserialization.
    """
    pass


def save_experiment(experiment_name: str,
                  template: BasePromptTemplate,
                  result: Dict[str, Any],
                  config: OptimizationConfig,
                  action_memory: Optional[ActionMemory] = None,
                  save_dir: str = "./experiments",
                  extra_data: Optional[Dict[str, Any]] = None) -> str:
    """
    保存实验配置和结果。
    
    Save experiment configuration and results.
    
    Args:
        experiment_name: 实验名称 (Experiment name)
        template: 提示词模板 (Prompt template)
        result: 优化结果 (Optimization result)
        config: 优化配置 (Optimization configuration)
        action_memory: 动作记忆（可选） (Action memory (optional))
        save_dir: 保存目录 (Save directory)
        extra_data: 额外数据（可选） (Extra data (optional))
        
    Returns:
        保存的文件路径 (Path to saved file)
        
    Raises:
        SerializationError: 序列化过程中发生错误 (Error during serialization)
    """
    try:
        # 创建保存目录
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # 添加时间戳
        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_path = os.path.join(save_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(experiment_path, exist_ok=True)
        
        # 准备数据
        # Prepare data
        data = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "template": _serialize_template(template),
            "result": result,
            "config": _serialize_config(config),
        }
        
        if action_memory:
            data["action_memory"] = action_memory.to_dict_list()
            
        if extra_data:
            data["extra_data"] = extra_data
            
        # 保存主数据文件
        # Save main data file
        main_file_path = os.path.join(experiment_path, "experiment.json")
        with open(main_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        # 使用pickle单独保存模板对象（用于完整恢复）
        # Save template object separately with pickle (for full restoration)
        template_path = os.path.join(experiment_path, "template.pkl")
        with open(template_path, "wb") as f:
            pickle.dump(template, f)
            
        # 保存配置对象
        # Save config object
        config_path = os.path.join(experiment_path, "config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump(config, f)
            
        logger.info(f"Saved experiment to {experiment_path}")
        return experiment_path
        
    except Exception as e:
        error_msg = f"Error saving experiment: {e}"
        logger.error(error_msg, exc_info=True)
        raise SerializationError(error_msg) from e


def load_experiment(experiment_path: str) -> Dict[str, Any]:
    """
    加载实验配置和结果。
    
    Load experiment configuration and results.
    
    Args:
        experiment_path: 实验保存路径 (Experiment save path)
        
    Returns:
        加载的实验数据 (Loaded experiment data)
        
    Raises:
        SerializationError: 反序列化过程中发生错误 (Error during deserialization)
    """
    try:
        # 加载主数据文件
        # Load main data file
        main_file_path = os.path.join(experiment_path, "experiment.json")
        with open(main_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # 尝试加载pickle文件（如果存在）
        # Try to load pickle files (if they exist)
        try:
            template_path = os.path.join(experiment_path, "template.pkl")
            if os.path.exists(template_path):
                with open(template_path, "rb") as f:
                    data["template_object"] = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load template pickle: {e}")
            
        try:
            config_path = os.path.join(experiment_path, "config.pkl")
            if os.path.exists(config_path):
                with open(config_path, "rb") as f:
                    data["config_object"] = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load config pickle: {e}")
            
        # 如果有动作记忆数据，创建ActionMemory对象
        # If action memory data exists, create ActionMemory object
        if "action_memory" in data:
            memory = ActionMemory()
            memory_data = data["action_memory"]
            data["action_memory_object"] = ActionMemory.from_dict_list(memory_data)
            
        logger.info(f"Loaded experiment from {experiment_path}")
        return data
        
    except Exception as e:
        error_msg = f"Error loading experiment: {e}"
        logger.error(error_msg, exc_info=True)
        raise SerializationError(error_msg) from e


def find_experiments(base_dir: str = "./experiments") -> List[str]:
    """
    查找所有保存的实验。
    
    Find all saved experiments.
    
    Args:
        base_dir: 实验基础目录 (Experiments base directory)
        
    Returns:
        实验路径列表 (List of experiment paths)
    """
    result = []
    try:
        for item in os.listdir(base_dir):
            full_path = os.path.join(base_dir, item)
            if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "experiment.json")):
                result.append(full_path)
    except Exception as e:
        logger.warning(f"Error finding experiments: {e}")
        
    return sorted(result)


def _serialize_template(template: BasePromptTemplate) -> Dict[str, Any]:
    """
    序列化模板对象为可JSON化的字典。
    
    Serialize template object to JSON-serializable dictionary.
    
    Args:
        template: 模板对象 (Template object)
        
    Returns:
        序列化后的字典 (Serialized dictionary)
    """
    result = {
        "class_name": template.__class__.__name__,
        "module": template.__class__.__module__,
        "sections": {},
        "optimizable_sections": list(template._optimizable_sections),
    }
    
    # 保存所有部分的内容
    # Save content of all sections
    for section in template.list_sections():
        content = template.get_section(section)
        result["sections"][section] = content
        
    return result


def _serialize_config(config: OptimizationConfig) -> Dict[str, Any]:
    """
    序列化配置对象为可JSON化的字典。
    
    Serialize configuration object to JSON-serializable dictionary.
    
    Args:
        config: 配置对象 (Configuration object)
        
    Returns:
        序列化后的字典 (Serialized dictionary)
    """
    result = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "update_granularity": config.update_granularity.name,
    }
    
    # 添加其他可能的配置字段
    # Add other possible configuration fields
    if hasattr(config, "early_stopping_patience") and config.early_stopping_patience is not None:
        result["early_stopping_patience"] = config.early_stopping_patience
        
    return result


def save_checkpoint(optimizer: Any, 
                  checkpoint_path: str, 
                  include_memory: bool = True, 
                  include_history: bool = True) -> str:
    """
    保存优化器检查点，用于稍后恢复。
    
    Save optimizer checkpoint for later restoration.
    
    Args:
        optimizer: 优化器对象 (Optimizer object)
        checkpoint_path: 检查点保存路径 (Checkpoint save path)
        include_memory: 是否包含动作记忆 (Whether to include action memory)
        include_history: 是否包含历史记录 (Whether to include history)
        
    Returns:
        保存的文件路径 (Path to saved file)
        
    Raises:
        SerializationError: 序列化过程中发生错误 (Error during serialization)
    """
    try:
        # 创建保存目录
        # Create save directory
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # 提取需要保存的状态
        # Extract state to save
        checkpoint_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "template": optimizer._best_template,
            "best_score": optimizer._best_score,
            "current_template": optimizer.current_template,
            "exploration_context": optimizer.exploration_context,
        }
        
        if include_memory and hasattr(optimizer, "action_memory"):
            checkpoint_data["action_memory"] = optimizer.action_memory
            
        if include_history and hasattr(optimizer, "history"):
            checkpoint_data["history"] = optimizer.history
            
        # 使用pickle保存
        # Save using pickle
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
            
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        error_msg = f"Error saving checkpoint: {e}"
        logger.error(error_msg, exc_info=True)
        raise SerializationError(error_msg) from e


def load_checkpoint(checkpoint_path: str, optimizer: Any) -> None:
    """
    从检查点恢复优化器状态。
    
    Restore optimizer state from checkpoint.
    
    Args:
        checkpoint_path: 检查点文件路径 (Checkpoint file path)
        optimizer: 要恢复状态的优化器对象 (Optimizer object to restore state to)
        
    Raises:
        SerializationError: 反序列化过程中发生错误 (Error during deserialization)
    """
    try:
        # 使用pickle加载
        # Load using pickle
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)
            
        # 恢复状态
        # Restore state
        optimizer._best_template = checkpoint_data["template"]
        optimizer._best_score = checkpoint_data["best_score"]
        optimizer.current_template = checkpoint_data["current_template"]
        optimizer.exploration_context = checkpoint_data["exploration_context"]
        
        if "action_memory" in checkpoint_data and hasattr(optimizer, "action_memory"):
            optimizer.action_memory = checkpoint_data["action_memory"]
            
        if "history" in checkpoint_data and hasattr(optimizer, "history"):
            optimizer.history = checkpoint_data["history"]
            
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
    except Exception as e:
        error_msg = f"Error loading checkpoint: {e}"
        logger.error(error_msg, exc_info=True)
        raise SerializationError(error_msg) from e


def export_template_to_json(template: BasePromptTemplate, file_path: str) -> str:
    """
    将模板导出为JSON格式。
    
    Export template to JSON format.
    
    Args:
        template: 要导出的模板 (Template to export)
        file_path: 导出文件路径 (Export file path)
        
    Returns:
        导出的文件路径 (Path to exported file)
        
    Raises:
        SerializationError: 导出过程中发生错误 (Error during export)
    """
    try:
        # 创建保存目录
        # Create save directory
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 序列化模板
        # Serialize template
        template_data = _serialize_template(template)
        
        # 写入文件
        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Exported template to {file_path}")
        return file_path
        
    except Exception as e:
        error_msg = f"Error exporting template: {e}"
        logger.error(error_msg, exc_info=True)
        raise SerializationError(error_msg) from e