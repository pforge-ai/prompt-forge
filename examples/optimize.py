# filename: examples/test_optimizer.py
import os
import time
import logging
import pprint
import sys
import pathlib
import shutil # For cleaning up test checkpoint dirs

from typing import Optional

# --- Path Setup ---
try:
    package_path = pathlib.Path(__file__).absolute().parent.parent
    sys.path.append(str(package_path))
    # Setup logger *after* path potentially added
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Appended to sys.path: {package_path}")
except Exception as e:
    print(f"Error adjusting sys.path: {e}. Ensure ptforge package is accessible.")
    sys.exit(1)

# --- Environment Loading ---
try:
    # Use python-dotenv >= 1.0.0 syntax
    from dotenv import load_dotenv, find_dotenv
    # find_dotenv() searches parent directories for .env
    env_path = find_dotenv()
    if env_path:
        load_dotenv(dotenv_path=env_path, override=True)
        logger.info(f"Loaded .env file from: {env_path}")
    else:
        logger.warning(".env file not found in current or parent directories.")
except ImportError:
    logger.warning("dotenv package not found. Skipping .env file loading.")
except Exception as e:
    logger.error(f"Error loading .env file: {e}")

# --- Imports from ptforge ---
from ptforge import (
    PromptOptimizer,
    OptimizationConfig,
    OptimizationResult, # Import result dataclass
    get_template,
    JsonlDataset,
    ExactMatchAccuracy,
    Evaluator,
    LLMBasedUpdater,
    OpenAIClient,
    UpdateGranularity,
    BasePromptTemplate
    # Base classes might be needed if creating custom components
)
# Import MetricResult if analyzing details from tracker
from ptforge.core.base import MetricResult


def run_prompt_optimizer_test():
    """运行 PromptOptimizer 的测试和演示"""
    logger.info("--- Starting PromptOptimizer Test Script ---")
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_output_dir = pathlib.Path(f"./ptforge_test_optimizer_output_{run_timestamp}")
    checkpoint_dir = test_output_dir / "checkpoints"
    log_dir = test_output_dir / "logs"

    try:
        # Create output directories
        test_output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created test output directory: {test_output_dir.absolute()}")
    except Exception as e:
        logger.error(f"Failed to create output directories: {e}", exc_info=True)
        return # Cannot proceed without output dirs

    # --- 1. Configure Components ---
    logger.info("Configuring components...")
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")

    if not api_key:
        logger.error("FATAL: OPENAI_API_KEY environment variable not set.")
        return

    # Ensure API base is handled correctly (None vs empty string)
    api_base = api_base if api_base else None
    logger.info(f"Using API Base: {api_base or 'Default OpenAI URL'}")

    try:
        target_model = OpenAIClient(api_key=api_key, api_base=api_base, model="deepseek-v3-250324", max_retries=2, timeout=45)
        optimizer_model = OpenAIClient(api_key=api_key, api_base=api_base, model="deepseek-v3-250324", max_retries=2, timeout=90)
    except Exception as e:
         logger.error(f"Failed to initialize OpenAIClient: {e}", exc_info=True)
         return

    # --- Dataset ---
    # Make sure this path exists relative to where you run the script
    data_file_path = pathlib.Path("examples/dummy_data.jsonl")
    if not data_file_path.exists():
         logger.error(f"FATAL: Data file '{data_file_path}' not found.")
         # Try creating a dummy file if it doesn't exist for basic testing
         try:
              dummy_content = [
                   {"input": "What is 1+1?", "reference": "2"},
                   {"input": "Capital of France?", "reference": "Paris"},
                   {"input": "Opposite of black?", "reference": "white"},
                   {"input": "What is the earth?", "reference": "planet"},
                   {"input": "2+2=?", "reference": "4"},
                   {"input": "H2O is?", "reference": "water"},
                   {"input": "Sun rises in the?", "reference": "east"},
                   {"input": "Largest mammal?", "reference": "blue whale"},
              ]
              data_file_path.parent.mkdir(exist_ok=True) # Ensure examples dir exists
              with open(data_file_path, 'w') as f:
                   for item in dummy_content:
                        f.write(f"{item}\n".replace("'", '"')) # Basic JSONL line
              logger.info(f"Created dummy data file at '{data_file_path}'")
         except Exception as create_e:
              logger.error(f"Could not create dummy data file: {create_e}. Exiting.")
              return

    try:
        dataset = JsonlDataset(file_path=str(data_file_path), input_field="input", reference_field="reference")
        logger.info(f"Loaded dataset with {len(dataset)} samples from '{data_file_path}'.")
        # Create a smaller validation dataset from the same source for testing
        validation_dataset = JsonlDataset(file_path=str(data_file_path), input_field="input", reference_field="reference")
        # Let's assume validation_dataset uses the same data for simplicity here
        logger.info(f"Using same data for validation set ({len(validation_dataset)} samples).")

    except Exception as e:
        logger.error(f"FATAL: Failed to load dataset: {e}", exc_info=True)
        return

    # --- Initial Template ---
    initial_template_values = {
        "role": "You are an assistant.", # Using default keys from RCTCRETemplate
        "task": "Answer the question: {{input}}",
        "context": None,
        "constraints": None,
        "response_format": None,
        "examples": None
    }

    optimizable = {"CONSTRAINTS", "RESPONSE_FORMAT"}

    initial_template = get_template(
        template_name="RCTCRE",
        initial_values=initial_template_values,
        optimizable_sections=optimizable
    )
    logger.info("Initial prompt template created (RCTCRE).")
    logger.debug(f"Optimizable sections: {initial_template.get_optimizable_sections().keys()}")

    # --- Evaluator ---
    accuracy_metric = ExactMatchAccuracy(case_sensitive=False)
    evaluator = Evaluator(metrics=[(accuracy_metric, 1.0)]) # Weight must be > 0
    logger.info("Evaluator configured with ExactMatchAccuracy.")

    # --- Updater ---
    updater = LLMBasedUpdater(
        optimizer_llm_client=optimizer_model,
        task_description="Improve the prompt template to correctly answer general knowledge questions based on input." # Provide task description
    )
    logger.info("LLMBasedUpdater configured.")

    # --- Optimization Configuration ---
    # Configure monitoring options here
    config = OptimizationConfig(
        epochs=1, # Run for 2 epochs for testing
        batch_size=8, # Smaller batch size
        update_granularity=UpdateGranularity.SECTION_REPHRASE,
        validation_dataset=validation_dataset, # Add validation dataset
        early_stopping_patience=1, # Stop if validation doesn't improve for 1 epoch
        # --- Monitoring/Checkpointing Config ---
        use_live_monitor=True, # Enable tqdm progress bar
        generate_report_at_end=True, # Generate report automatically
        tracker_config={ # Pass tracker settings
            'experiment_name': f'optimizer_test_{run_timestamp}',
            'save_dir': str(log_dir), # Use pathlib object converted to string
            'save_history': True,
            'autosave_interval': 5, # Autosave every 5 mins
        }
    )
    logger.info(f"Optimization config created: {config}")
    logger.info(f"Monitoring enabled: Live={config.use_live_monitor}, Report={config.generate_report_at_end}")
    logger.info(f"Tracker output directory: {log_dir}")

    # --- 2. Initialize Optimizer ---
    try:
        optimizer = PromptOptimizer(
            target_model_client=target_model,
            initial_template=initial_template,
            dataset=dataset,
            evaluator=evaluator,
            updater=updater,
            config=config,
        )
        logger.info("PromptOptimizer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize PromptOptimizer: {e}", exc_info=True)
        return

    # --- 3. Run Optimization ---
    logger.info("--- Starting Optimization Run ---")
    opt_start_time = time.time()
    best_template: Optional[BasePromptTemplate] = None
    final_result: Optional[OptimizationResult] = None

    try:
        # Use with statement for client context management if clients support it
        # Assuming OpenAIClient doesn't inherently need 'with', just run optimize
        best_template, final_result = optimizer.optimize()

        opt_end_time = time.time()
        logger.info("--- Optimization Finished ---")
        logger.info(f"Total optimization time: {opt_end_time - opt_start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"An error occurred during the optimization process: {e}", exc_info=True)
        # Attempt to save tracker state even if optimize crashes
        try:
             optimizer.tracker.save()
             logger.info("Saved tracker state after error during optimization.")
        except Exception as save_e:
             logger.error(f"Could not save tracker state after error: {save_e}")
        return # Exit after error

    # --- 4. Output Results & Test Monitoring ---
    if final_result and best_template:
        print("\n" + "="*30)
        print("--- Optimization Results Summary ---")
        print(f"Best Score Achieved: {final_result.best_score:.4f}")
        print("\nBest Prompt Template Found:")
        # Use render method for clean output
        print(best_template.render(data={}, ignore_missing_variables=True))

        print("\nOptimization History (Epoch Summaries):")
        pprint.pprint(final_result.history)

        # --- Monitoring Output ---
        print("\n--- Monitoring & Tracking Info ---")
        # Access tracker summary (assuming optimize populates tracker correctly)
        try:
            summary = optimizer.tracker.get_summary()
            print("Tracker Summary:")
            pprint.pprint(summary)
            # Example: Access specific metric from last step in tracker history
            if optimizer.tracker.history:
                last_step_info = optimizer.tracker.history[-1]
                print(f"\nInfo from last tracked step (Step {last_step_info.get('step', 'N/A')}):")
                print(f"  Score Before Update: {last_step_info.get('before_score', 'N/A'):.4f}")
                print(f"  Template Accepted: {last_step_info.get('accepted', 'N/A')}")
                # Print detailed results scores if available
                details = last_step_info.get('detailed_results', {})
                if details:
                     print("  Detailed Metric Scores:")
                     for name, metric_data in details.items():
                          print(f"    - {name}: {metric_data.get('score', 'N/A'):.4f}")

            # Report was generated automatically if config flag was set
            if config.generate_report_at_end:
                 print(f"\nReport should have been generated in: {log_dir}")

        except Exception as e:
            logger.error(f"Error accessing tracker information: {e}", exc_info=True)
            print(f"\nError accessing tracker information: {e}")

    else:
        logger.error("Optimization did not produce a final result or best template.")
        print("\nOptimization did not complete successfully.")
        return # Exit if no result


    # --- 5. Test Checkpointing ---
    print("\n" + "="*30)
    print("--- Checkpointing Test ---")
    checkpoint_file = checkpoint_dir / "optimizer_checkpoint.pkl"

    # a) Save checkpoint
    try:
        logger.info(f"Attempting to save checkpoint to: {checkpoint_file}")
        optimizer.save_optimizer_checkpoint(str(checkpoint_file))
        if checkpoint_file.exists():
             print(f"Checkpoint saved successfully to {checkpoint_file}")
        else:
             print(f"Checkpoint file was not created at {checkpoint_file}")
             # Don't proceed with loading if save failed
             return

    except Exception as e:
        logger.error(f"Error during save_optimizer_checkpoint: {e}", exc_info=True)
        print(f"\nError saving checkpoint: {e}")
        return # Don't proceed if save fails

    # b) Load checkpoint into a *new* optimizer instance
    # Create a new config - realistically might load config from file too
    # Use same components for simplicity in this test
    config_for_load = OptimizationConfig(tracker_config={'save_dir': str(test_output_dir / "logs_loaded")}) # Use different log dir
    log_dir_loaded = test_output_dir / "logs_loaded"
    log_dir_loaded.mkdir(parents=True, exist_ok=True)


    try:
        logger.info("Initializing a new optimizer instance for loading...")
        loaded_optimizer = PromptOptimizer(
            target_model_client=target_model, # Re-use clients for test
            initial_template=get_template("RCTCRE"), # Provide a basic initial template
            dataset=dataset,                   # Re-use dataset
            evaluator=evaluator,               # Re-use evaluator
            updater=updater,                   # Re-use updater
            config=config_for_load,            # Use new config
        )

        logger.info(f"Attempting to load checkpoint from: {checkpoint_file}")
        loaded_optimizer.load_optimizer_checkpoint(str(checkpoint_file))
        print("Checkpoint loaded successfully into new optimizer instance.")

        # c) Verify loaded state (optional, basic checks)
        print("\nVerifying loaded state (basic checks):")
        print(f"  Loaded best score: {loaded_optimizer._best_score:.4f} (Original best: {optimizer._best_score:.4f})")
        # Compare template content hash for basic check
        original_best_hash = hash(optimizer._best_template.render({},ignore_missing_variables=True))
        loaded_best_hash = hash(loaded_optimizer._best_template.render({},ignore_missing_variables=True))
        print(f"  Loaded best template hash matches original: {original_best_hash == loaded_best_hash}")

        loaded_current_hash = hash(loaded_optimizer.current_template.render({},ignore_missing_variables=True))
        original_current_hash = hash(optimizer.current_template.render({},ignore_missing_variables=True))
        print(f"  Loaded current template hash matches original: {loaded_current_hash == original_current_hash}")

        # d) Optionally run a small evaluation with the loaded optimizer
        print("\nRunning one evaluation step with loaded optimizer...")
        try:
             # Evaluate the loaded 'current_template' on one batch
             first_batch = dataset.get_batches(config.batch_size)[0]
             loaded_score, _ = loaded_optimizer._evaluate_on_validation_set(loaded_optimizer.current_template) # Reuse validation func
             print(f"  Score using loaded current_template on validation set: {loaded_score:.4f}")
        except Exception as eval_e:
             logger.error(f"Error during evaluation with loaded optimizer: {eval_e}", exc_info=True)
             print(f"  Error during evaluation with loaded optimizer: {eval_e}")


    except FileNotFoundError:
         logger.error(f"Checkpoint file not found for loading: {checkpoint_file}")
         print(f"\nERROR: Checkpoint file {checkpoint_file} not found for loading.")
    except Exception as e:
        logger.error(f"Error during checkpoint loading or verification: {e}", exc_info=True)
        print(f"\nError loading/verifying checkpoint: {e}")

    # --- 6. Cleanup (Optional) ---
    # print(f"\nTest finished. Output files are in: {test_output_dir.absolute()}")
    # print("You may want to manually delete this directory after inspection.")
    # # Or uncomment to automatically clean up:
    # try:
    #     # shutil.rmtree(test_output_dir)
    #     # logger.info(f"Cleaned up test output directory: {test_output_dir}")
    # except Exception as clean_e:
    #     logger.error(f"Error cleaning up test directory: {clean_e}")


if __name__ == "__main__":
    run_prompt_optimizer_test()