# Prompt Forge

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/ptforge.svg)](https://pypi.org/project/ptforge/)
[![Python Versions](https://img.shields.io/pypi/pyversions/ptforge.svg)](https://pypi.org/project/ptforge/)
[![License](https://img.shields.io/github/license/pforge-ai/prompt-forge)](https://github.com/pforge-ai/prompt-forge/blob/main/LICENSE)

**自动化 LLM 提示词优化框架**

[English](README.md) | **简体中文**

</div>

## 📖 简介

**Prompt Forge** 是一个专为自动化优化大型语言模型（LLM）提示词而设计的 Python 框架。其灵感来源于机器学习中的迭代训练过程，旨在系统化地发现高效提示词，特别适用于只能通过 API 访问 LLM 的场景。

该框架帮助用户摆脱耗时且容易陷入次优的手动提示词调整过程。通过将提示词结构视为"架构"，并利用基于 LLM 的优化器，Prompt Forge 迭代地优化提示词以最大化其性能。

## ✨ 特性

* **结构化提示词模板:** 使用可组合的部分（如 Role, Task, Constraints, Examples）来定义提示词。包含内置模板 (RCTCRE, APE, CRISPE, BROKE)。
* **自动化优化循环:** 模拟机器学习训练中的 Epoch 和 Batch 概念。
* **两阶段基于 LLM 的优化器:** 利用强大的 LLM 先分析反馈并生成优化方向，然后应用这些方向来修改提示词，全程使用结构化的 JSON 进行通信。
* **可配置的更新粒度:** 控制提示词修改的范围和强度（从微调到完全重构）。
* **可扩展组件:** 轻松添加自定义的数据集、LLM 客户端、评估指标、提示词模板和更新策略。
* **多指标评估:** 定义并加权多个评估指标，以指导优化过程达成复杂目标，并返回详细评估结果。
* **聚焦 API 调用:** 专为通过 API 访问的 LLM 设计。

## 🔧 安装

```bash
pip install ptforge
```

## 🚀 快速上手

以下示例展示了 Prompt Forge 的基本用法：

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
    target_llm = OpenAIClient(model="gpt-3.5-turbo") # 或你偏好的模型/客户端
    optimizer_llm = OpenAIClient(model="gpt-4o")    # 推荐使用强力模型进行优化

    dataset = JsonlDataset("examples/dummy_data.jsonl") # 相对于执行路径的路径

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
    print(f"优化完成。最佳得分: {result.best_score}")
    print("最佳模板各部分:")
    for section in best_template.list_sections():
        print(f"--- {section} ---")
        print(best_template.get_section(section) or "(空/无)")

except Exception as e:
    logging.error(f"优化失败: {e}", exc_info=True)
```

## 🏗️ 架构

Prompt Forge 的设计基于以下核心组件:

- **PromptOptimizer**: 协调整个优化过程的核心类
- **PromptTemplate**: 结构化表示提示词的模板类
- **LLMClient**: 与目标语言模型进行交互的接口
- **Dataset**: 提供训练数据
- **Evaluator**: 评估生成结果的质量
- **PromptUpdater**: 基于评估结果提出模板修改建议
- **OptimizationConfig**: 控制优化过程的配置

## 🧩 内置模板

Prompt Forge 提供了多种内置的提示词模板:

- **RCTCRE**: Role-Context-Task-Constraints-ResponseFormat-Examples
- **APE**: Action-Purpose-Expect
- **CRISPE**: Capacity/Role-Insight-Statement-Personality-Experiment
- **BROKE**: Background-Role-Objectives-KeyResults-Evolve

## 📚 详细文档

更多详细文档和示例请访问我们的[GitHub 仓库](https://github.com/pforge-ai/prompt-forge)。

## 🤝 贡献

欢迎贡献! 请参阅我们的[贡献指南](CONTRIBUTING.md)了解更多信息。

## 📄 许可证

此项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/pforge-ai">pforge.ai</a></sub>
</div>
