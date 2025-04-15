# prompt_forge/templates/builtins/broke.py

from typing import Any, Dict, List, Optional, Set
import logging

from ..base_template import BasePromptTemplate

logger = logging.getLogger(__name__)

# 定义 Section 名称常量 (Define section name constants)
BACKGROUND = "BACKGROUND"
ROLE = "ROLE"
OBJECTIVES = "OBJECTIVES"
KEY_RESULTS = "KEY_RESULTS"
EVOLVE = "EVOLVE" # How the prompt/response should evolve

class BROKETemplate(BasePromptTemplate):
    """
    实现 BROKE 结构的 Prompt 模板。
    (Background-Role-Objectives-KeyResults-Evolve)
    继承 BasePromptTemplate 的通用逻辑和默认 render 方法。
    Implements the BROKE structured prompt template.
    Inherits common logic and default render method from BasePromptTemplate.
    """

    _SUPPORTED_SECTIONS: Set[str] = {
        BACKGROUND,
        ROLE,
        OBJECTIVES,
        KEY_RESULTS,
        EVOLVE,
    }
    _REQUIRED_SECTIONS: Set[str] = { # All sections are required
        BACKGROUND,
        ROLE,
        OBJECTIVES,
        KEY_RESULTS,
        EVOLVE,
    }
    _OPTIONAL_SECTIONS: Set[str] = set()

    def __init__(
        self,
        background: str,
        role: str,
        objectives: str,
        key_results: str,
        evolve: str,
        optimizable_sections: Optional[Set[str]] = None,
    ):
        """
        初始化 BROKE 模板。

        Args:
            background: 任务或请求的背景信息。 (Background information for the task or request.)
            role: LLM 需要扮演的角色。 (The role the LLM needs to play.)
            objectives: 需要达成的目标。 (The objectives to be achieved.)
            key_results: 衡量目标达成的关键结果或标准。 (Key results or criteria for measuring objective achievement.)
            evolve: 关于 Prompt 或响应应如何演进或改进的说明。 (Instructions on how the prompt or response should evolve or improve.)
            optimizable_sections: (可选) 可被优化的 Section 名称集合。默认为所有 Section。
                                    (Optional set of section names that can be optimized. Defaults to all sections.)
        """
        super().__init__()
        self._sections: Dict[str, Optional[str]] = {
            BACKGROUND: background,
            ROLE: role,
            OBJECTIVES: objectives,
            KEY_RESULTS: key_results,
            EVOLVE: evolve,
        }

        if optimizable_sections is None:
            self._optimizable_sections: Set[str] = set(self._SUPPORTED_SECTIONS)
        else:
            invalid_sections = optimizable_sections - self._SUPPORTED_SECTIONS
            if invalid_sections:
                logger.warning(
                    f"Ignoring invalid section names provided in optimizable_sections: {invalid_sections}"
                )
            self._optimizable_sections = optimizable_sections & self._SUPPORTED_SECTIONS

        logger.debug(f"BROKETemplate initialized. Optimizable sections: {self._optimizable_sections}")

    # render 方法由基类提供 (render method is provided by the base class)

    # 其他通用方法也由基类提供 (Other common methods also provided by base class)

