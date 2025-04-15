# prompt_forge/templates/__init__.py

# 导入基类 (Import base class)
from .base_template import BasePromptTemplate

# 导入内置模板类 (Import built-in template classes)
from .builtins.rctcre import RCTCRETemplate
from .builtins.ape import APETemplate
from .builtins.crispe import CRISPETemplate
from .builtins.broke import BROKETemplate

# 导入工厂函数 (Import factory function)
from .factory import get_template

# 定义公开接口 (Define public interface)
__all__ = [
    "BasePromptTemplate",
    "RCTCRETemplate",
    "APETemplate",
    "CRISPETemplate",
    "BROKETemplate",
    "get_template",
]

