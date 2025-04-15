# prompt_forge/templates/builtins/crispe.py

from typing import Any, Dict, List, Optional, Set
import logging

from ..base_template import BasePromptTemplate

logger = logging.getLogger(__name__)

# 定义 Section 名称常量 (Define section name constants)
CAPACITY_ROLE = "CAPACITY_ROLE" # Capacity and Role
INSIGHT = "INSIGHT"
STATEMENT = "STATEMENT" # Statement of what you want to be done
PERSONALITY = "PERSONALITY"
EXPERIMENT = "EXPERIMENT" # Experiment on what to change to improve

class CRISPETemplate(BasePromptTemplate):
    """
    实现 CRISPE 结构的 Prompt 模板。
    (Capacity/Role-Insight-Statement-Personality-Experiment)
    继承 BasePromptTemplate 的通用逻辑和默认 render 方法。
    Implements the CRISPE structured prompt template.
    Inherits common logic and default render method from BasePromptTemplate.
    """

    _SUPPORTED_SECTIONS: Set[str] = {
        CAPACITY_ROLE,
        INSIGHT,
        STATEMENT,
        PERSONALITY,
        EXPERIMENT,
    }
    _REQUIRED_SECTIONS: Set[str] = { # All sections are required
        CAPACITY_ROLE,
        INSIGHT,
        STATEMENT,
        PERSONALITY,
        EXPERIMENT,
    }
    _OPTIONAL_SECTIONS: Set[str] = set()

    def __init__(
        self,
        capacity_role: str,
        insight: str,
        statement: str,
        personality: str,
        experiment: str,
        optimizable_sections: Optional[Set[str]] = None,
    ):
        """
        初始化 CRISPE 模板。

        Args:
            capacity_role: LLM 的能力和角色。 (The capacity and role of the LLM.)
            insight: 背景信息或上下文洞察。 (Background information or contextual insight.)
            statement: 需要完成的具体任务陈述。 (The specific task statement.)
            personality: LLM 应表现出的个性或语气。 (The personality or tone the LLM should adopt.)
            experiment: 期望 LLM 尝试或改变以改进输出的方向。 (Direction on what the LLM should try or change to improve output.)
            optimizable_sections: (可选) 可被优化的 Section 名称集合。默认为所有 Section。
                                    (Optional set of section names that can be optimized. Defaults to all sections.)
        """
        super().__init__()
        self._sections: Dict[str, Optional[str]] = {
            CAPACITY_ROLE: capacity_role,
            INSIGHT: insight,
            STATEMENT: statement,
            PERSONALITY: personality,
            EXPERIMENT: experiment,
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

        logger.debug(f"CRISPETemplate initialized. Optimizable sections: {self._optimizable_sections}")

    # render 方法由基类提供 (render method is provided by the base class)

    # 其他通用方法也由基类提供 (Other common methods also provided by base class)

