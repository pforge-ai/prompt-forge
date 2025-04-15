# prompt_forge/templates/builtins/rctcre.py

from typing import Any, Dict, List, Optional, Set
import logging

# 从上级目录导入基类
# Import base class from parent directory
from ..base_template import BasePromptTemplate

logger = logging.getLogger(__name__)

# 定义 Section 名称常量 (Define section name constants)
ROLE = "ROLE"
CONTEXT = "CONTEXT"
TASK = "TASK"
CONSTRAINTS = "CONSTRAINTS"
RESPONSE_FORMAT = "RESPONSE_FORMAT"
EXAMPLES = "EXAMPLES" # Optional section

class RCTCRETemplate(BasePromptTemplate):
    """
    实现 RCTCRE 结构的 Prompt 模板。
    (Role-Context-Task-Constraints-ResponseFormat-Examples)
    继承 BasePromptTemplate 的通用逻辑和默认 render 方法。
    Implements the RCTCRE structured prompt template.
    Inherits common logic and default render method from BasePromptTemplate.
    """

    # --- 类属性定义 (Class attribute definitions) ---
    _SUPPORTED_SECTIONS: Set[str] = {
        ROLE, CONTEXT, TASK, CONSTRAINTS, RESPONSE_FORMAT, EXAMPLES,
    }
    _REQUIRED_SECTIONS: Set[str] = { # EXAMPLES is optional
        ROLE, CONTEXT, TASK, CONSTRAINTS, RESPONSE_FORMAT,
    }
    _OPTIONAL_SECTIONS: Set[str] = {EXAMPLES}

    def __init__(
        self,
        role: str,
        context: str,
        task: str,
        constraints: str,
        response_format: str,
        examples: Optional[str] = None,
        optimizable_sections: Optional[Set[str]] = None,
    ):
        """
        初始化 RCTCRE 模板。

        Args:
            role: LLM 扮演的角色。 (The role the LLM should play.)
            context: 任务的背景信息。 (Background information for the task.)
            task: 需要执行的具体任务描述。 (Specific description of the task to be performed.)
            constraints: 任务执行的约束条件。 (Constraints for task execution.)
            response_format: 期望的输出格式描述。 (Description of the desired output format.)
            examples: (可选) 输入输出示例。 (Optional input-output examples.)
            optimizable_sections: (可选) 一个包含可被优化器修改的 Section 名称的集合。
                                        如果为 None，则所有提供了初始内容的 Section 默认为可优化。
                                        (Optional set containing names of sections that can be modified
                                         by the optimizer. If None, all sections provided with initial
                                         content default to optimizable.)
        """
        super().__init__() # 调用基类初始化 (Call base class initializer)

        # 填充 _sections 字典 (Populate the _sections dictionary)
        self._sections = {
            ROLE: role,
            CONTEXT: context,
            TASK: task,
            CONSTRAINTS: constraints,
            RESPONSE_FORMAT: response_format,
            EXAMPLES: examples,
        }

        # 设置可优化 Sections (Set optimizable sections)
        if optimizable_sections is None:
            # Default: sections with non-None initial content are optimizable
            self._optimizable_sections = {
                name for name in self._sections.items()
            }
        else:
            self._optimizable_sections = optimizable_sections
        # else:
        #     # Validate user-provided optimizable sections against supported sections
        #     invalid_sections = optimizable_sections - self._SUPPORTED_SECTIONS
        #     if invalid_sections:
        #         logger.warning(
        #             f"Ignoring invalid section names provided in optimizable_sections: {invalid_sections}"
        #         )
        #     # Use only valid and supported sections
        #     self._optimizable_sections = optimizable_sections & self._SUPPORTED_SECTIONS

        logger.debug(f"RCTCRETemplate initialized. Optimizable sections: {self._optimizable_sections}")

    # render 方法由基类提供 (render method is provided by the base class)

    # _validate_section_name, update_section, get_section, list_sections,
    # get_optimizable_sections, mark_optimizable, __repr__
    # 这些方法也由 BasePromptTemplate 提供 (These methods are also provided by BasePromptTemplate)

