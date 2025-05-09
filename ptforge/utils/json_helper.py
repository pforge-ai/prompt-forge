# ptforge/utils/json_helper.py
"""
提供健壮的JSON解析和修复功能。
用于处理LLM生成的不完整或不规范的JSON响应。

Provides robust JSON parsing and fixing utilities.
Used for handling incomplete or malformed JSON responses generated by LLMs.
"""

import json
import re
import logging
from typing import Any, Dict, Optional, Union, Tuple, List, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')

def extract_json_str(text: str) -> str:
    """
    从文本中提取JSON字符串。
    处理LLM输出中可能的markdown代码块，前导文本，或不完整JSON。
    
    Extracts JSON string from text.
    Handles markdown code blocks, leading text, or incomplete JSON in LLM outputs.
    
    Args:
        text: 可能包含JSON的文本 (Text that may contain JSON)
    
    Returns:
        提取出的JSON字符串 (Extracted JSON string)
    """
    # 首先尝试查找markdown风格的JSON代码块
    # First, try to find markdown-style JSON code blocks
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    code_matches = re.findall(code_block_pattern, text)
    
    if code_matches:
        # 使用最后一个代码块(通常是最完整的)
        # Use the last code block (usually the most complete)
        potential_json = code_matches[-1].strip()
    else:
        # 没有代码块，查找可能的JSON对象
        # No code blocks, look for possible JSON objects
        # 使用正则尝试查找 { ... } 模式
        # Use regex to find { ... } pattern
        json_pattern = r"\{[\s\S]*\}"
        json_matches = re.findall(json_pattern, text)
        
        if json_matches:
            # 使用最长的匹配(可能是最完整的)
            # Use the longest match (likely the most complete)
            potential_json = max(json_matches, key=len)
        else:
            # 无法找到JSON对象
            # Unable to find JSON object
            logger.warning(f"No JSON object found in text: {text[:100]}...")
            return ""
    
    return potential_json

def fix_malformed_json(json_str: str) -> str:
    """
    尝试修复畸形的JSON字符串。
    
    Attempts to fix malformed JSON strings.
    
    Args:
        json_str: 可能不完整或畸形的JSON字符串 (Potentially incomplete or malformed JSON string)
    
    Returns:
        修复后的JSON字符串 (Fixed JSON string)
    """
    # 1. 处理缺失引号的键名
    # 1. Handle missing quotes in key names
    unquoted_key_pattern = r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:'
    json_str = re.sub(unquoted_key_pattern, r'\1"\2":', json_str)
    
    # 2. 处理缺失引号的字符串值
    # 2. Handle missing quotes in string values
    unquoted_value_pattern = r':\s*([a-zA-Z][a-zA-Z0-9_\s]*[a-zA-Z0-9_]*)([,}])'
    json_str = re.sub(unquoted_value_pattern, r': "\1"\2', json_str)
    
    # 3. 处理单引号(替换为双引号)
    # 3. Handle single quotes (replace with double quotes)
    json_str = re.sub(r"(?<!\\)'", '"', json_str)
    
    # 4. 处理尾部逗号
    # 4. Handle trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # 5. 处理尾部不完整的JSON
    # 5. Handle incomplete JSON at the end
    if not json_str.endswith('}') and json_str.startswith('{'):
        # 尝试根据花括号计数来补全
        # Try to complete based on brace count
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        
        if open_braces > close_braces:
            # 添加缺失的右花括号
            # Add missing closing braces
            json_str += '}' * (open_braces - close_braces)
    
    # 6. 处理可能的不完整属性
    # 6. Handle potentially incomplete properties
    if re.search(r':\s*$', json_str) or re.search(r',$', json_str):
        # JSON以冒号或逗号结尾, 这是畸形的
        # JSON ending with colon or comma is malformed
        json_str = re.sub(r':\s*$', ': ""}', json_str)
        json_str = re.sub(r',$', '}', json_str)
    
    return json_str

def parse_json_robust(text: str, default_value: Optional[T] = None, 
                      expected_type: Optional[Type] = None) -> Union[Any, T]:
    """
    健壮地解析可能包含JSON的文本。
    尝试提取和修复JSON，并验证结果类型。
    
    Robustly parses text that may contain JSON.
    Attempts to extract and fix JSON, and validates result type.
    
    Args:
        text: 可能包含JSON的文本 (Text that may contain JSON)
        default_value: 解析失败时返回的默认值 (Default value to return if parsing fails)
        expected_type: 预期的解析结果类型 (Expected type of the parsing result)
    
    Returns:
        解析后的Python对象，或在失败时返回default_value 
        (Parsed Python object, or default_value on failure)
    """
    if not text:
        logger.warning("Empty text provided for JSON parsing")
        return default_value
    
    # 提取可能的JSON字符串
    # Extract possible JSON string
    json_str = extract_json_str(text)
    
    if not json_str:
        logger.warning("Failed to extract JSON string from text")
        return default_value
    
    # 尝试直接解析
    # Try direct parsing
    try:
        result = json.loads(json_str)
        # 检查类型
        # Check type
        if expected_type and not isinstance(result, expected_type):
            logger.warning(f"Parsed JSON is not of expected type. Expected: {expected_type}, Got: {type(result)}")
            return default_value
        return result
    except json.JSONDecodeError as e:
        logger.info(f"Initial JSON parsing failed: {e}. Attempting to fix...")
    
    # 尝试修复并重新解析
    # Try to fix and parse again
    fixed_json = fix_malformed_json(json_str)
    try:
        result = json.loads(fixed_json)
        logger.info("Successfully parsed JSON after fixing")
        # 检查类型
        # Check type
        if expected_type and not isinstance(result, expected_type):
            logger.warning(f"Fixed JSON is not of expected type. Expected: {expected_type}, Got: {type(result)}")
            return default_value
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON even after fixing: {e}")
        return default_value

def extract_action_json(text: str) -> Dict[str, Any]:
    """
    专门用于从LLM响应中提取动作JSON。
    针对特定格式的动作说明进行优化。
    
    Specifically for extracting action JSON from LLM responses.
    Optimized for specific action specification formats.
    
    Args:
        text: LLM响应文本 (LLM response text)
    
    Returns:
        解析后的动作字典，如果解析失败则返回空字典 
        (Parsed action dictionary, or empty dict if parsing fails)
    """
    # 默认动作结构
    # Default action structure
    default_action = {}
    
    # 使用健壮解析
    # Use robust parsing
    action = parse_json_robust(text, default_value=default_action, expected_type=dict)
    
    # 验证关键字段
    # Validate key fields
    if action and not all(k in action for k in ["target_section"]):
        logger.warning(f"Parsed action missing required fields: {action}")
        # 尝试从文本推断字段
        # Try to infer fields from text
        action = infer_action_from_text(text, action)
    
    return action

def infer_action_from_text(text: str, partial_action: Dict[str, Any]) -> Dict[str, Any]:
    """
    当JSON解析不完整时，尝试从文本推断动作属性。
    
    Attempts to infer action properties from text when JSON parsing is incomplete.
    
    Args:
        text: 原始LLM响应文本 (Original LLM response text)
        partial_action: 部分解析的动作字典 (Partially parsed action dictionary)
    
    Returns:
        尽可能完整的动作字典 (Action dictionary as complete as possible)
    """
    # 复制部分动作以避免修改原对象
    # Copy partial action to avoid modifying the original
    action = partial_action.copy()
    
    # 尝试推断目标部分
    # Try to infer target section
    if "target_section" not in action:
        # 查找部分名称模式
        # Look for section name patterns
        section_match = re.search(r"section[:\s]+['\"]*([A-Z_]+)['\"]*", text, re.IGNORECASE)
        if section_match:
            action["target_section"] = section_match.group(1)
    
    # 尝试推断动作类型
    # Try to infer action type
    if "action_type" not in action:
        # 查找可能的动作类型
        # Look for possible action types
        action_types = ["REPLACE", "MODIFY", "RESTRUCTURE", "EXPAND", "SIMPLIFY"]
        for action_type in action_types:
            if action_type.lower() in text.lower():
                action["action_type"] = action_type
                break
        else:
            # 默认为MODIFY
            # Default to MODIFY
            action["action_type"] = "MODIFY"
    
    # 尝试提取内容
    # Try to extract content
    if "content" not in action:
        # 查找可能的内容块，使用启发式方法
        # Look for possible content blocks using heuristics
        content_patterns = [
            r"content[:\s]+['\"]([\s\S]*?)['\"]\s*[,}]",
            r"new content[:\s]+['\"]([\s\S]*?)['\"]\s*[,}]",
            r"content['\"]\s*:\s*['\"]([\s\S]*?)['\"]\s*[,}]"
        ]
        
        for pattern in content_patterns:
            content_match = re.search(pattern, text, re.IGNORECASE)
            if content_match:
                action["content"] = content_match.group(1)
                break
    
    return action