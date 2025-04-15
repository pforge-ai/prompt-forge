# prompt_forge/templates/builtins/ape.py

from typing import Any, Dict, List, Optional, Set
import logging

from ..base_template import BasePromptTemplate

logger = logging.getLogger(__name__)

# 定义 Section 名称常量 (Define section name constants)
ACTION = "ACTION"
PURPOSE = "PURPOSE"
EXPECT = "EXPECT"

class APETemplate(BasePromptTemplate):
    """
    实现 APE 结构的 Prompt 模板。
    (Action-Purpose-Expect)
    继承 BasePromptTemplate 的通用逻辑和默认 render 方法。
    Implements the APE structured prompt template.
    Inherits common logic and default render method from BasePromptTemplate.
    """

    _SUPPORTED_SECTIONS: Set[str] = {ACTION, PURPOSE, EXPECT}
    _REQUIRED_SECTIONS: Set[str] = {ACTION, PURPOSE, EXPECT} # All sections are required
    _OPTIONAL_SECTIONS: Set[str] = set() # No optional sections

    def __init__(
        self,
        action: str,
        purpose: str,
        expect: str,
        optimizable_sections: Optional[Set[str]] = None,
    ):
        """
        初始化 APE 模板。

        Args:
            action: 需要执行的动作或任务。 (The action or task to be performed.)
            purpose: 执行该动作的目的或原因。 (The purpose or reason for the action.)
            expect: 期望 LLM 达到的具体输出或结果。 (The specific output or result expected from the LLM.)
            optimizable_sections: (可选) 可被优化的 Section 名称集合。默认为所有 Section。
                                    (Optional set of section names that can be optimized. Defaults to all sections.)
        """
        super().__init__()
        self._sections: Dict[str, Optional[str]] = {
            ACTION: action,
            PURPOSE: purpose,
            EXPECT: expect,
        }

        if optimizable_sections is None:
            self._optimizable_sections: Set[str] = set(self._SUPPORTED_SECTIONS) # All sections optimizable by default
        else:
            invalid_sections = optimizable_sections - self._SUPPORTED_SECTIONS
            if invalid_sections:
                logger.warning(
                    f"Ignoring invalid section names provided in optimizable_sections: {invalid_sections}"
                )
            self._optimizable_sections = optimizable_sections & self._SUPPORTED_SECTIONS

        logger.debug(f"APETemplate initialized. Optimizable sections: {self._optimizable_sections}")

    # render 方法由基类提供 (render method is provided by the base class)

    # 其他通用方法也由基类提供 (Other common methods also provided by base class)

