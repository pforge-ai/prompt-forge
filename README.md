# Prompt Forge

[简体中文](README.zh.md) | **English**

**Prompt Forge** is a Python framework designed for the automated optimization of prompts for large language models (LLMs). Inspired by the iterative training processes used in machine learning, Prompt Forge aims to systematically discover highly effective prompts based on evaluation metrics and datasets, particularly for scenarios involving API-only LLM access.

This framework helps users move beyond manual prompt tuning, which can be time-consuming and prone to suboptimal results, especially on large datasets. By treating prompt structures as architectures and using an LLM-based optimizer, Prompt Forge iteratively refines prompts to maximize performance according to user-defined metrics.

## Features (Planned/Included)

* **Structured Prompt Templates:** Define prompts using composable sections (e.g., Role, Task, Constraints, Examples). Includes built-in templates (RCTCRE, APE, CRISPE, BROKE).
* **Automated Optimization Loop:** Mimics ML training with epochs and batches.
* **Two-Step LLM-Based Optimizer:** Utilizes a powerful LLM (OptimizerLLM) to first analyze feedback and generate optimization directions, then apply those directions to modify the prompt, using structured JSON communication.
* **Configurable Update Granularity:** Control the scope and intensity of prompt changes (from micro-adjustments to full rewrites).
* **Extensible Components:** Easily add custom datasets, LLM clients, evaluation metrics, prompt templates, and update strategies.
* **Multi-Metric Evaluation:** Define and weight multiple metrics to guide optimization towards complex goals, returning detailed results.
* **API-Only Focus:** Designed to work with LLMs accessible only via APIs.

## Installation

```bash
pip install ptforge
```

Quick Start
```python
import os
import logging
from ptforge import (
    PromptOptimizer, OptimizationConfig, get_template, JsonlDataset,
    ExactMatchAccuracy, Evaluator, LLMBasedUpdater, OpenAIClient, UpdateGranularity
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Ensure API keys are set (e.g., via environment variables)
# os.environ["OPENAI_API_KEY"] = "your-key"
# os.environ["OPENAI_API_BASE"] = "your-base-url-if-needed" # Optional

try:
    # 1. Setup Components
    target_llm = OpenAIClient(model="gpt-3.5-turbo") # Or your preferred model/client
    optimizer_llm = OpenAIClient(model="gpt-4o")    # Powerful model recommended for optimization

    dataset = JsonlDataset("examples/dummy_data.jsonl") # Path relative to execution

    initial_template = get_template(
        "RCTCRE",
        initial_values={
            "role": "Assistant answering questions.",
            "task": "Answer concisely: {{input}}",
            "context": "General knowledge question.",
            "constraints": None,
            "response_format": "Plain text.",
            "examples": None,
        },
        optimizable_sections={"role", "task", "constraints", "context", "response_format"}
    )

    evaluator = Evaluator(metrics=[(ExactMatchAccuracy(case_sensitive=False), 1.0)])

    updater = LLMBasedUpdater(
        optimizer_llm_client=optimizer_llm,
        task_description="Answer general knowledge questions concisely and accurately."
    )

    config = OptimizationConfig(epochs=2, batch_size=4, update_granularity=UpdateGranularity.SECTION_REPHRASE)

    # 2. Initialize Optimizer
    optimizer = PromptOptimizer(target_llm, initial_template, dataset, evaluator, updater, config)

    # 3. Run Optimization
    # Use context managers if your client supports them (OpenAIClient does)
    with target_llm, optimizer_llm:
        best_template, result = optimizer.optimize()

    # 4. Use Results
    print(f"Optimization finished. Best score: {result.best_score}")
    print("Best template sections:")
    for section in best_template.list_sections():
        print(f"--- {section} ---")
        print(best_template.get_section(section) or "(Empty/None)")

except Exception as e:
    logging.error(f"Optimization failed: {e}", exc_info=True)
```

## Contributing
Contributions are welcome! Please refer to the contributing guidelines.

## License
This project is licensed under the MIT License - see the LICENSE file for details.