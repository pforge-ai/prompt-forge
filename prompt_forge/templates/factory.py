# prompt_forge/templates/factory.py

import logging
from typing import Any, Dict, Optional, Set, Type

# 导入基类和内置模板类
# Import base class and built-in template classes
from .base_template import BasePromptTemplate
from .builtins.rctcre import RCTCRETemplate
from .builtins.ape import APETemplate
from .builtins.crispe import CRISPETemplate
from .builtins.broke import BROKETemplate

logger = logging.getLogger(__name__)

# 模板名称到类的映射 (Mapping from template name to class)
# 使用小写名称作为键，方便不区分大小写查找
# Use lowercase names as keys for case-insensitive lookup
TEMPLATE_REGISTRY: Dict[str, Type[BasePromptTemplate]] = {
    "rctcre": RCTCRETemplate,
    "ape": APETemplate,
    "crispe": CRISPETemplate,
    "broke": BROKETemplate,
}

def get_template(
    template_name: str,
    initial_values: Dict[str, Any],
    optimizable_sections: Optional[Set[str]] = None,
) -> BasePromptTemplate:
    """
    根据名称和初始值创建并返回一个 Prompt 模板实例。

    Args:
        template_name: 模板的名称 (例如 "RCTCRE", "APE")，不区分大小写。
                        (Name of the template (e.g., "RCTCRE", "APE"), case-insensitive.)
        initial_values: 一个字典，包含初始化模板所需的 Section 名称和对应的内容。
                        键应该是对应模板类 __init__ 方法期望的参数名 (通常是小写 section 名称)。
                        (A dictionary containing the section names and corresponding content
                            needed to initialize the template. Keys should match the expected
                            parameter names of the template class's __init__ method
                            (usually lowercase section names).)
        optimizable_sections: (可选) 指定哪些 Section 是可优化的集合。
                                如果为 None，则使用模板类的默认行为。
                                (Optional set specifying which sections are optimizable.
                                    If None, the template class's default behavior is used.)

    Returns:
        所请求的 Prompt 模板类的实例。
        (An instance of the requested prompt template class.)

    Raises:
        ValueError: 如果找不到指定的 template_name，或者 initial_values
                    缺少模板所需的必要参数。
                    (If the specified template_name is not found, or if initial_values
                        is missing required arguments for the template.)
        TypeError: 如果 initial_values 包含模板 __init__ 不接受的参数。
                    (If initial_values contains arguments not accepted by the template's __init__.)
    """
    template_name_lower = template_name.lower()
    template_class = TEMPLATE_REGISTRY.get(template_name_lower)

    if template_class is None:
        raise ValueError(
            f"Unknown template name: '{template_name}'. "
            f"Available templates: {list(TEMPLATE_REGISTRY.keys())}"
        )

    logger.info(f"Creating template instance for '{template_class.__name__}'")

    try:
        # 尝试使用 initial_values 和 optimizable_sections (如果提供) 来实例化
        # Try to instantiate using initial_values and optimizable_sections (if provided)
        if optimizable_sections is not None:
            # 假设 __init__ 接受 optimizable_sections 参数
            # Assume __init__ accepts optimizable_sections argument
            instance = template_class(**initial_values, optimizable_sections=optimizable_sections)
        else:
            instance = template_class(**initial_values)

        # (可选) 进行额外的验证，确保所有必需的 Section 都已提供
        # (Optional) Perform additional validation to ensure all required sections were provided
        # This depends on how initial_values keys map to _REQUIRED_SECTIONS
        # For now, rely on the template's __init__ signature validation (TypeError)

        return instance

    except TypeError as e:
        #捕获因参数不匹配或缺失导致的 TypeError (Catch TypeError due to argument mismatch/missing)
        logger.error(f"Failed to initialize {template_class.__name__}: {e}", exc_info=True)
        # 可以尝试提供更具体的错误信息 (Could try providing more specific error messages)
        # E.g., inspect template_class.__init__ signature vs initial_values keys
        raise TypeError(f"Error initializing {template_class.__name__}: {e}. "
                        f"Please check if 'initial_values' ({list(initial_values.keys())}) "
                        f"match the required arguments for the template.") from e
    except Exception as e:
        # 捕获其他可能的初始化错误 (Catch other potential initialization errors)
        logger.error(f"An unexpected error occurred during template initialization: {e}", exc_info=True)
        raise # Re-raise other exceptions


