# prompt_forge/__init__.py

import logging

# --- Core Components ---
from .core.optimizer import PromptOptimizer, OptimizationResult
from .core.config import OptimizationConfig
from .core.base import UpdateGranularity, MetricResult

# --- Base Classes (for extension) ---
# Decide whether to expose base classes at the top level
# Option 1: Expose them for easier extension
from .core.base import BaseDataset as BaseDataset # Alias to avoid potential name clash if needed later
from .core.base import BaseLLMClient as BaseLLMClient
from .templates.base_template import BasePromptTemplate as BasePromptTemplate
from .core.base import BaseMetric as BaseMetric
from .updaters.base_updater import BasePromptUpdater as BasePromptUpdater
# Option 2: Don't expose base classes here, users import from submodules if needed

# --- Concrete Implementations & Factories ---

# Templates
from .templates import (
    get_template,
    RCTCRETemplate,
    APETemplate,
    CRISPETemplate,
    BROKETemplate,
)

# Datasets
from .datasets import (
    JsonlDataset,
    CsvDataset,
)

# Metrics
from .metrics import (
    ExactMatchAccuracy,
)

# Evaluators
from .evaluators import Evaluator # Assuming evaluator.py defines Evaluator class

# Updaters
from .updaters import (
    LLMBasedUpdater,
)

# LLMClients
from .llms import (
    OpenAIClient
)

# --- Logging Configuration ---
# Setup default null handler to avoid "No handler found" warnings.
# The user application should configure logging properly.
logging.getLogger(__name__).addHandler(logging.NullHandler())


# --- Define Public API (`__all__`) ---
# Controls what `from prompt_forge import *` imports
# Also useful for static analysis tools
__all__ = [
    # Core classes & config
    "PromptOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "UpdateGranularity",
    "MetricResult",

    # Base classes (if exposing - see Option 1 above)
    "BaseDataset",
    "BaseLLMClient",
    "BasePromptTemplate",
    "BaseMetric",
    "BasePromptUpdater",

    # Concrete Templates & Factory
    "get_template",
    "RCTCRETemplate",
    "APETemplate",
    "CRISPETemplate",
    "BROKETemplate",

    # Concrete Datasets
    "JsonlDataset",
    "CsvDataset",

    # Concrete Metrics
    "ExactMatchAccuracy",

    # Evaluator
    "Evaluator",

    # Concrete Updaters
    "LLMBasedUpdater",

    # LLMClient
    "OpenAIClient",
]

# Optional: Define package version
# __version__ = "0.1.0"

