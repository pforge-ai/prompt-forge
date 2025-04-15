# Prompt Forge

**Prompt Forge** 是一个专为自动化优化大型语言模型（LLM）提示词（Prompt）而设计的 Python 框架。其灵感来源于机器学习中的迭代训练过程，旨在基于评估指标和数据集系统性地发现高效的提示词，尤其适用于只能通过 API 访问 LLM 的场景。

该框架帮助用户摆脱耗时且容易陷入次优的手动提示词调整过程，特别是在处理大型数据集时。通过将提示词结构视为“架构”，并利用基于 LLM 的优化器，Prompt Forge 根据用户定义的指标迭代地优化提示词，以最大化其性能。

## 特性 (已实现/计划中)

* **结构化提示词模板 (Structured Prompt Templates):** 使用可组合的 Section（如 Role, Task, Constraints, Examples）来定义提示词。包含内置模板 (RCTCRE, APE, CRISPE, BROKE)。
* **自动化优化循环 (Automated Optimization Loop):** 模拟机器学习训练中的 Epoch 和 Batch 概念。
* **两阶段基于 LLM 的优化器 (Two-Step LLM-Based Optimizer):** 利用强大的 LLM (OptimizerLLM) 先分析反馈并生成优化方向，然后应用这些方向来修改提示词，全程使用结构化的 JSON 进行通信。
* **可配置的更新粒度 (Configurable Update Granularity):** 控制提示词修改的范围和强度（从微调到完全重构）。
* **可扩展组件 (Extensible Components):** 轻松添加自定义的数据集、LLM 客户端、评估指标、提示词模板和更新策略。
* **多指标评估 (Multi-Metric Evaluation):** 定义并加权多个评估指标，以指导优化过程达成复杂目标，并返回详细评估结果。
* **聚焦 API 调用 (API-Only Focus):** 专为通过 API 访问的 LLM 设计。

## 安装

```bash
pip install ptforge
```

## 快速上手
```python
import os
import logging
from ptforge import (
    PromptOptimizer, OptimizationConfig, get_template, JsonlDataset,
    ExactMatchAccuracy, Evaluator, LLMBasedUpdater, OpenAIClient, UpdateGranularity
)

# 配置日志
logging.basicConfig(level=logging.INFO)

# 确保设置了 API 密钥（例如通过环境变量）
# os.environ["OPENAI_API_KEY"] = "your-key"
# os.environ["OPENAI_API_BASE"] = "your-base-url-if-needed" # 可选

try:
    # 1. 设置组件
    target_llm = OpenAIClient(model="gpt-4o-mini") # 或你偏好的模型/客户端
    optimizer_llm = OpenAIClient(model="gpt-4o")    # 推荐使用强力模型进行优化

    # 假设脚本从项目根目录运行，数据在 examples/ 子目录
    dataset = JsonlDataset("examples/dummy_data.jsonl")

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

    # 2. 初始化优化器
    optimizer = PromptOptimizer(target_llm, initial_template, dataset, evaluator, updater, config)

    # 3. 运行优化
    # 如果客户端支持，使用上下文管理器
    with target_llm, optimizer_llm:
        best_template, result = optimizer.optimize()

    # 4. 使用结果
    print(f"Optimization finished. Best score: {result.best_score}")
    print("Best template sections:")
    for section in best_template.list_sections():
        print(f"--- {section} ---")
        print(best_template.get_section(section) or "(Empty/None)")

except Exception as e:
    logging.error(f"Optimization failed: {e}", exc_info=True)
```

# 贡献
欢迎贡献！请参阅贡献指南。

# 许可证
本项目采用 MIT 许可证 - 详情请参阅 LICENSE 文件