# prompt_forge/templates/base_template.py

import abc
import logging
import re
from typing import Any, Dict, List, Optional, Set, Type, TypeVar

logger = logging.getLogger(__name__)

# Type variable for concrete PromptTemplate implementations
T_PromptTemplate = TypeVar("T_PromptTemplate", bound="BasePromptTemplate")

class BasePromptTemplate(abc.ABC):
    """
    Prompt 模板的抽象基类。包含管理 sections 和可优化状态的通用逻辑。
    提供了一个默认的 render 方法，支持 {{variable}} 替换。
    Abstract base class for prompt templates. Includes common logic for managing
    sections and optimizable status. Provides a default render method with
    {{variable}} substitution support.

    Subclasses MUST define the following class attributes:
        _SUPPORTED_SECTIONS: Set[str]
        _REQUIRED_SECTIONS: Set[str]
        _OPTIONAL_SECTIONS: Set[str]

    Subclasses MUST implement:
        __init__(self, ...): To accept specific section content and call super().__init__().

    Subclasses CAN optionally override:
        render(self, data: Dict[str, Any]) -> str: If default rendering is not suitable.
    """
    # Subclasses must define these
    _SUPPORTED_SECTIONS: Set[str] = set()
    _REQUIRED_SECTIONS: Set[str] = set()
    _OPTIONAL_SECTIONS: Set[str] = set()

    # --- ( __init__, _validate_section_name, update_section, get_section 等方法保持不变 ) ---
    def __init__(self):
        """Initializes internal storage for sections and optimizable status."""
        self._sections: Dict[str, Any] = {}
        self._optimizable_sections: Set[str] = set()
        # Basic validation on class attributes defined by subclass
        if not isinstance(self._SUPPORTED_SECTIONS, set) or not self._SUPPORTED_SECTIONS:
             raise NotImplementedError(f"{self.__class__.__name__} must define _SUPPORTED_SECTIONS class attribute.")
        if self._REQUIRED_SECTIONS | self._OPTIONAL_SECTIONS != self._SUPPORTED_SECTIONS:
             logger.warning(f"Inconsistent section definitions in {self.__class__.__name__}: Required U Optional != Supported.")
        if self._REQUIRED_SECTIONS & self._OPTIONAL_SECTIONS:
             logger.warning(f"Inconsistent section definitions in {self.__class__.__name__}: Required and Optional sections overlap.")

    def _validate_section_name(self, section_name: str):
        """检查 Section 名称是否在此模板支持的范围内 (Check if section name is supported by this template)"""
        if section_name not in self._SUPPORTED_SECTIONS:
            raise ValueError(
                f"Invalid section name '{section_name}' for {self.__class__.__name__}. Supported sections are: {self._SUPPORTED_SECTIONS}"
            )

    # --- 默认 Render 实现 (Default Render Implementation) ---
    def render(self, data: Optional[Dict[str, Any]] = None, ignore_missing_variables=False) -> str:
        """
        将各个 Section (按字母顺序) 组合渲染成最终的 Prompt 字符串，并进行变量替换。
        使用 {{变量名}} 语法进行替换。如果 data 中缺少某个变量，会保留占位符并记录警告。
        子类可以覆盖此方法以实现自定义渲染逻辑。

        Renders the sections (in alphabetical order) into the final prompt string,
        performing variable substitution using the {{variable_name}} syntax.
        If a variable is missing in data, the placeholder is kept and a warning is logged.
        Subclasses can override this method for custom rendering logic.

        Args:
            data: (可选) 包含模板所需变量的字典。键是变量名，值是替换内容。
                  (Optional dictionary containing variables needed by the template.
                   Keys are variable names, values are the content for substitution.)
            ignore_missing_variables: (可选) 是否忽略缺失变量。
                                      (Optional boolean whether ignore missing variables or not)

        Returns:
            渲染后的 Prompt 字符串。 (The rendered prompt string.)
        """
        if data is None:
            data = {} # Use empty dict if no data provided

        rendered_parts = []
        # 按字母顺序渲染 Section (Render sections in alphabetical order)
        for section_name in self.list_sections(): # list_sections now returns sorted list
            content = self.get_section(section_name)

            # 只处理非空字符串内容 (Only process non-empty string content)
            if isinstance(content, str) and content:
                # --- 执行变量替换 (Perform variable substitution) ---
                def replace_variable(match):
                    nonlocal ignore_missing_variables
                    variable_name = match.group(1) # 获取括号内的变量名 (Get variable name inside braces)
                    if variable_name in data:
                        # Convert value to string for substitution
                        return str(data[variable_name])
                    else:
                        if not ignore_missing_variables:
                            logger.warning(f"Variable '{{{{{variable_name}}}}}' not found in provided data for template {self.__class__.__name__}. Placeholder kept.")
                        return match.group(0) # 保留原始占位符 (Keep original placeholder)

                try:
                    # 使用 re.sub 进行替换 (Use re.sub for substitution)
                    # 正则表达式匹配 {{variable_name}} (Regex matches {{variable_name}})
                    substituted_content = re.sub(r"\{\{(\w+)\}\}", replace_variable, content)
                except Exception as e:
                     logger.error(f"Error during variable substitution in section '{section_name}': {e}", exc_info=True)
                     substituted_content = content # Fallback to original content on error

                # --- 格式化 Section (Format the section) ---
                rendered_parts.append(f"<{section_name}>\n{substituted_content}\n</{section_name}>")
            elif content: # Handle non-string, non-empty content if necessary
                 logger.debug(f"Section '{section_name}' has non-string content ({type(content)}), rendering as string.")
                 # Render non-string content directly without substitution for now
                 rendered_parts.append(f"<{section_name}>\n{str(content)}\n</{section_name}>")


        return "\n\n".join(rendered_parts) # 用双换行分隔各部分 (Separate parts with double newline)

    # --- ( update_section, get_section 等通用方法保持不变 ) ---
    def update_section(self, section_name: str, content: Any) -> None:
        self._validate_section_name(section_name)
        if section_name in self._REQUIRED_SECTIONS and not content:
            logger.warning(f"Setting required section '{section_name}' to empty/None content.")
        elif section_name in self._OPTIONAL_SECTIONS and not content:
            logger.debug(f"Setting optional section '{section_name}' to empty/None.")
        self._sections[section_name] = content
        logger.debug(f"Section '{section_name}' updated in {self.__class__.__name__}.")

    def get_section(self, section_name: str) -> Any:
        self._validate_section_name(section_name)
        return self._sections.get(section_name)

    def list_sections(self) -> List[str]:
        return sorted(list(self._SUPPORTED_SECTIONS))

    def get_optimizable_sections(self) -> Dict[str, Any]:
        return {
            name: self._sections.get(name)
            for name in self._optimizable_sections
        }

    def mark_optimizable(self, section_name: str, optimizable: bool = True) -> None:
        self._validate_section_name(section_name)
        if optimizable:
            self._optimizable_sections.add(section_name)
            logger.debug(f"Section '{section_name}' marked as optimizable in {self.__class__.__name__}.")
        else:
            self._optimizable_sections.discard(section_name)
            logger.debug(f"Section '{section_name}' marked as non-optimizable in {self.__class__.__name__}.")

    def __repr__(self) -> str:
         optimizable_list = sorted(list(self._optimizable_sections))
         return f"{self.__class__.__name__}(optimizable={optimizable_list})"

