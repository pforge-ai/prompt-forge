# examples/optimize.py
import os
import time # Import time module
import logging
import pprint # For pretty printing results
import sys
import pathlib # Import pathlib

# --- 用户添加：根据本地路径修改 ---
# --- User Added: Modify according to local path ---
# 建议：更好的方式是将 prompt_forge 安装为包，或者使用相对路径（如果结构允许）
# Recommendation: Better to install prompt_forge as a package or use relative paths if structure allows
try:
    # Adjust this path based on where you run the script from relative to the package
    package_path = pathlib.Path(__file__).absolute().parent.parent # Assumes examples is one level inside project root
    sys.path.append(str(package_path))
    logger = logging.getLogger(__name__) # Setup logger after path potentially added
    logger.info(f"Appended to sys.path: {package_path}")
except Exception as e:
    print(f"Error adjusting sys.path: {e}. Ensure prompt_forge package is accessible.")
    sys.exit(1)


# --- 用户添加：加载环境变量 ---
# --- User Added: Load environment variables ---
try:
    import dotenv
    # Loads .env file from the current working directory or parent directories
    dotenv.load_dotenv()
    logger.info(".env file loaded if found.")
except ImportError:
    logger.warning("dotenv package not found. Skipping .env file loading. Ensure environment variables are set manually.")
except Exception as e:
        logger.error(f"Error loading .env file: {e}")


# 从 prompt_forge 库导入所需组件
# Import necessary components from the prompt_forge library
from ptforge import (
    PromptOptimizer,
    OptimizationConfig,
    get_template,         # Template factory
    RCTCRETemplate,       # Example template type
    JsonlDataset,         # Dataset loader
    ExactMatchAccuracy,   # Metric implementation
    Evaluator,            # Evaluator class
    LLMBasedUpdater,      # LLM-based updater
    OpenAIClient,         # LLM client implementation
    UpdateGranularity,    # Enum for update level
)

# --- 配置日志 (Configure Logging) ---
# Setup logging configuration here, after potential sys.path changes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__) # Re-get logger after basicConfig


def run_optimization():
    """运行 Prompt 优化示例"""
    logger.info("--- Starting PromptForge Optimization Example ---")

    # --- 1. 配置组件 (Configure Components) ---

    # a) LLM 客户端 (LLM Clients)
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE") # Read from env

    if not api_key:
        logger.error("FATAL: OPENAI_API_KEY environment variable not set.")
        print("\nFATAL: OPENAI_API_KEY environment variable not set. Please set it before running.")
        return

    logger.info(f"Using API Base: {api_base or 'Default OpenAI URL'}")

    # Target model
    target_model = OpenAIClient(
        api_key=api_key,
        api_base=api_base, # Pass api_base
        model="ep-20250413172247-v4vkm", # User specified model
        max_retries=2,
        timeout=30,
    )

    # Optimizer model
    optimizer_model = OpenAIClient(
        api_key=api_key,
        api_base=api_base, # Pass api_base
        model="ep-20250413172247-v4vkm", # User specified model
        max_retries=2,
        timeout=30, # User specified timeout
    )

    # b) 数据集 (Dataset)
    # Assuming script is run from project root, and data is in examples/
    data_file_path = "examples/dummy_data.jsonl" # User specified path
    try:
        dataset = JsonlDataset(
            file_path=data_file_path,
            input_field="input",
            reference_field="reference"
        )
        logger.info(f"Loaded dataset with {len(dataset)} samples from '{data_file_path}'.")
    except FileNotFoundError:
        logger.error(f"FATAL: Data file '{data_file_path}' not found.")
        print(f"\nFATAL: Data file '{data_file_path}' not found. Please ensure it exists.")
        return
    except Exception as e:
        logger.error(f"FATAL: Failed to load dataset: {e}", exc_info=True)
        print(f"\nFATAL: Failed to load dataset: {e}")
        return

    # c) 初始 Prompt 模板 (Initial Prompt Template)
    initial_template_values = {
        "role": "You are a helpful assistant designed to answer questions accurately.",
        "task": "Answer the following question concisely: {{input}}",
        "context": "The user is asking a general knowledge question.",
        "constraints": None, # User set to None
        "response_format": "Plain text.",
        "examples": None,
    }
    optimizable = {"CONSTRAINTS", "RESPONSE_FORMAT"}

    initial_template = get_template(
        template_name="RCTCRE",
        initial_values=initial_template_values,
        optimizable_sections=optimizable
    )
    print(initial_template._optimizable_sections)
    logger.info("Initial prompt template created.")


    # d) 评估指标和评估器 (Metrics and Evaluator)
    accuracy_metric = ExactMatchAccuracy(case_sensitive=False)
    evaluator = Evaluator(metrics=[(accuracy_metric, 1.0)])
    logger.info("Evaluator configured with ExactMatchAccuracy.")

    # e) Prompt 更新器 (Prompt Updater)
    # User removed task_description
    updater = LLMBasedUpdater(optimizer_llm_client=optimizer_model)
    logger.info("LLMBasedUpdater configured (without explicit task_description).")

    # f) 优化配置 (Optimization Configuration)
    config = OptimizationConfig(
        epochs=1,
        batch_size=8,
        update_granularity=UpdateGranularity.SECTION_REPHRASE,
    )
    logger.info(f"Optimization config set: epochs={config.epochs}, batch_size={config.batch_size}, granularity={config.update_granularity.name}")

    # --- 2. 初始化并运行优化器 (Initialize and Run Optimizer) ---
    optimizer = PromptOptimizer(
        target_model_client=target_model,
        initial_template=initial_template,
        dataset=dataset,
        evaluator=evaluator,
        updater=updater,
        config=config,
    )

    logger.info("--- Starting Optimization Run ---")
    start_time = time.time()

    try:
        # 使用 with 语句管理客户端生命周期
        # Use with statement to manage client lifecycle
        with target_model, optimizer_model:
                best_template, final_result = optimizer.optimize()

        end_time = time.time()
        logger.info("--- Optimization Finished ---")
        logger.info(f"Total optimization time: {end_time - start_time:.2f} seconds")

        # --- 3. 输出结果 (Output Results) ---
        print("\n--- Optimization Results ---")
        print(f"Best Score Achieved: {final_result.best_score:.4f}")
        print("\nBest Prompt Template Found:")
        print(best_template.render(data={}, ignore_missing_variables=True))

        print("\nOptimization History (Epoch Summaries):")
        pprint.pprint(final_result.history)

    except Exception as e:
        logger.error(f"An error occurred during the optimization process: {e}", exc_info=True)
        print(f"\nAn error occurred: {e}")
    # No finally block needed as 'with' handles closing


if __name__ == "__main__":
    run_optimization()
