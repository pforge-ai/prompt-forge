# filename: ptforge/__init__.py
# prompt_forge/__init__.py

import logging

# --- Core Components ---
# Updated import to include RL classes
from .core import (
    PromptOptimizer, OptimizationResult,
    RLPromptOptimizer, RLOptimizationResult, # <-- Added RL classes
    OptimizationConfig,
    UpdateGranularity, MetricResult,
)

# --- Base Classes (for extension) ---
# Keep exposing base classes as before
from .core.base import BaseDataset as BaseDataset
from .core.base import BaseLLMClient as BaseLLMClient
from .templates.base_template import BasePromptTemplate as BasePromptTemplate
from .core.base import BaseMetric as BaseMetric
from .updaters.base_updater import BasePromptUpdater as BasePromptUpdater
# Expose RL base classes if needed?
from .core.reward_calculator import RewardCalculator # Expose base RewardCalculator
from .core.action_memory import ActionMemory # Expose ActionMemory
from .core import create_default_reward_calculator

# --- Monitoring Components ---
# Import monitoring classes
from .monitoring import OptimizationTracker, LiveProgressMonitor

# --- Concrete Implementations & Factories ---

# Templates
from .templates import (
    get_template,
    RCTCRETemplate,
    APETemplate,
    CRISPETemplate,
    BROKETemplate,
    ActionSpace, # Expose ActionSpace?
    StructuredAction # Expose StructuredAction?
)

# Datasets
from .datasets import (
    JsonlDataset,
    CsvDataset,
)

# Metrics
from .metrics import (
    ExactMatchAccuracy,
    # Add other metrics if they exist and should be exposed
)

# Evaluators
from .evaluators import Evaluator

# Updaters
from .updaters import (
    LLMBasedUpdater,
)

# LLMClients (assuming clients.py is in ptforge/llms)
try:
    from .llms import (
        OpenAIClient
        # Add other clients here if they exist
    )
    _llm_clients_available = True
except ImportError:
    _llm_clients_available = False
    # Define dummy classes or skip if import fails? For now, just note it failed.
    # logger.warning("Could not import default LLM clients. Ensure dependencies are installed.")

# --- Logging Configuration ---
# Setup default null handler to avoid "No handler found" warnings.
# The user application should configure logging properly.
logging.getLogger(__name__).addHandler(logging.NullHandler())


# --- Define Public API (`__all__`) ---
# Controls what `from ptforge import *` imports
# Also useful for static analysis tools
__all__ = [
    # Core classes & config
    "PromptOptimizer",
    "OptimizationResult",
    "RLPromptOptimizer",     
    "RLOptimizationResult",   
    "OptimizationConfig",
    "UpdateGranularity",
    "MetricResult",
    "create_default_reward_calculator",

    # Base classes
    "BaseDataset",
    "BaseLLMClient",
    "BasePromptTemplate",
    "BaseMetric",
    "BasePromptUpdater",
    "RewardCalculator",      
    "ActionMemory",         
    # Monitoring
    "OptimizationTracker",   
    "LiveProgressMonitor",    

    # Concrete Templates & Factory
    "get_template",
    "RCTCRETemplate",
    "APETemplate",
    "CRISPETemplate",
    "BROKETemplate",
    "ActionSpace",           
    "StructuredAction",      

    # Concrete Datasets
    "JsonlDataset",
    "CsvDataset",

    # Concrete Metrics
    "ExactMatchAccuracy",

    # Evaluator
    "Evaluator",

    # Concrete Updaters
    "LLMBasedUpdater",
]

# Conditionally add LLM clients to __all__
if _llm_clients_available:
    __all__.extend([
        "OpenAIClient",
    ])