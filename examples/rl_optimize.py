# filename: examples/test_rl_optimizer.py
import os
import time
import logging
import pprint
import sys
import pathlib
import shutil
import dataclasses # For MetricResult if used directly
from typing import List, Dict, Any, Tuple, Callable, Optional # For evaluator function type hint

# --- Path Setup ---
try:
    package_path = pathlib.Path(__file__).absolute().parent.parent
    sys.path.append(str(package_path))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Appended to sys.path: {package_path}")
except Exception as e:
    print(f"Error adjusting sys.path: {e}. Ensure ptforge package is accessible.")
    sys.exit(1)

# --- Environment Loading ---
try:
    from dotenv import load_dotenv, find_dotenv
    env_path = find_dotenv()
    if env_path:
        load_dotenv(dotenv_path=env_path, override=True)
        logger.info(f"Loaded .env file from: {env_path}")
    else:
        logger.warning(".env file not found.")
except ImportError:
    logger.warning("dotenv package not found. Skipping .env file loading.")
except Exception as e:
    logger.error(f"Error loading .env file: {e}")

# --- Imports from ptforge ---
from ptforge import (
    RLPromptOptimizer,         # RL Optimizer
    RLOptimizationResult,    # RL Result class
    OptimizationConfig,
    get_template,
    JsonlDataset,
    ExactMatchAccuracy,
    Evaluator,                 # Evaluator class (used *inside* the evaluator function)
    OpenAIClient,
    ActionMemory,              # RL component
    create_default_reward_calculator, # Default reward function
    # Import necessary base types if needed
    BasePromptTemplate,
    MetricResult,
    BaseLLMClient,
    UpdateGranularity # Not directly used by RL, but maybe by config?
)

def create_batch_evaluator(metrics: List[Tuple[Any, float]]) -> Callable[[BasePromptTemplate, List[Dict[str, Any]]], Tuple[float, Dict[str, MetricResult]]]:
    """
    Factory function to create the evaluator callable required by RLPromptOptimizer.
    This function encapsulates the Evaluator class instance.
    """
    # Instantiate the Evaluator *once* here
    evaluator_instance = Evaluator(metrics=metrics)
    logger.info(f"Batch evaluator function created with metrics: {[m[0].name for m in metrics]}")

    def batch_evaluate_func(template: BasePromptTemplate, batch: List[Dict[str, Any]]) -> Tuple[float, Dict[str, MetricResult]]:
        """
        The actual evaluation function passed to RLPromptOptimizer.
        It renders prompts for the batch, generates predictions, and evaluates.
        """
        if not batch:
            logger.warning("Evaluator function received an empty batch.")
            return -float('inf'), {}

        try:
            # 1. Render prompts for the batch
            prompts = [template.render(data) for data in batch]
            references = [data.get('reference') for data in batch] # Assuming 'reference' key exists
        except KeyError as e:
            logger.error(f"Evaluator function: Missing key in batch data during render: {e}", exc_info=True)
            return -float('inf'), {} # Return error score
        except Exception as e:
            logger.error(f"Evaluator function: Error rendering prompts: {e}", exc_info=True)
            return -float('inf'), {}

        try:
            # 2. Generate predictions (Needs access to target_llm_client)
            #    This is a limitation - the evaluator function doesn't have access
            #    to the optimizer's target LLM client directly.
            #    ****** REVISION NEEDED ******
            #    The RLPromptOptimizer.__init__ needs to take the target_llm_client
            #    and the _evaluate_template_on_batch method should handle prediction.
            #    The evaluator function should *only* take predictions and references.
            #
            #    Let's redefine the expected evaluator signature for RL Optimizer:
            #    evaluator: Callable[[List[str], List[Any]], Tuple[float, Dict[str, MetricResult]]]
            #    And modify RL Optimizer's _evaluate methods accordingly.
            #
            #    *** Assuming RL Optimizer is modified ***
            #    This factory should return the evaluator_instance.evaluate method directly
            #    or a lambda wrapping it if the signature matches exactly.
            #
            #    Let's revert to the original plan where RL Optimizer's evaluator call handles predictions internally.
            #    This function should encapsulate the metric computation part.

            # This function WILL BE CALLED BY RL OPTIMIZER, which should handle prediction itself.
            # The RL Optimizer's internal _evaluate method will call this function
            # by passing the generated predictions and references.

            # THEREFORE, the signature required by RL Optimizer *should* be modified to expect
            # a function like: evaluate(predictions: List[str], references: List[Any]) -> Tuple[float, Dict[str, MetricResult]]
            # For now, let's *assume* RL Optimizer expects the original signature and handles prediction internally.
            # THIS FACTORY IS THEN REDUNDANT if the user just passes an Evaluator instance.
            # Let's assume the user passes the INSTANCE of Evaluator directly to RLPromptOptimizer.

            # *** CONCLUSION: This factory is likely not needed if RLPromptOptimizer takes an Evaluator instance. ***
            # *** Let's remove this factory and pass Evaluator instance directly. ***
            # *** REQUIRES CONFIRMING/MODIFYING RLPromptOptimizer's __init__ and evaluation methods. ***

            # --- TEMPORARY WORKAROUND: Assume RL Optimizer passes preds/refs ---
            # --- This factory creates a function matching the *metric* eval signature ---
             raise NotImplementedError("RL Optimizer internal evaluation logic needs clarification.")


        except Exception as e:
            logger.error(f"Evaluator function: Error during evaluation: {e}", exc_info=True)
            return -float('inf'), {} # Return error score on failure

    # Returning the instance's evaluate method might work if signatures match
    # return evaluator_instance.evaluate # This expects evaluate(preds, refs)

    # If RL Optimizer handles prediction and expects evaluate(template, batch):
    # We need target_llm_client here which isn't feasible.
    # Let's assume the __init__ signature of RLPromptOptimizer needs adjustment.

# --- Placeholder: Define a simplified evaluator function ---
# --- This assumes RL Optimizer handles predictions internally ---
def simple_batch_evaluator(evaluator_instance: Evaluator) -> Callable[[BasePromptTemplate, List[Dict[str, Any]], BaseLLMClient], Tuple[float, Dict[str, MetricResult]]]:
    """
    Creates evaluator func that requires LLM client passed dynamically (not ideal).
    Alternative: RL Optimizer init takes Evaluator instance and target LLM client,
                 and its internal methods handle rendering, prediction, and calling evaluator.evaluate.
                 This seems the most robust design.

    Let's proceed assuming RL Optimizer expects an Evaluator INSTANCE.
    """
    pass # This function won't be used if we pass the instance.


def run_rl_optimizer_test():
    """运行 RLPromptOptimizer 的测试和演示"""
    logger.info("--- Starting RLPromptOptimizer Test Script ---")
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_output_dir = pathlib.Path(f"./ptforge_test_rl_optimizer_output_{run_timestamp}")
    checkpoint_dir = test_output_dir / "checkpoints"
    log_dir = test_output_dir / "logs"

    try:
        test_output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created test output directory: {test_output_dir.absolute()}")
    except Exception as e:
        logger.error(f"Failed to create output directories: {e}", exc_info=True)
        return

    # --- 1. Configure Components ---
    logger.info("Configuring components for RL Optimizer...")
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")

    if not api_key:
        logger.error("FATAL: OPENAI_API_KEY environment variable not set.")
        return

    api_base = api_base if api_base else None
    logger.info(f"Using API Base: {api_base or 'Default OpenAI URL'}")

    try:
        # Use distinct models for target and optimizer for clarity
        target_llm = OpenAIClient(api_key=api_key, api_base=api_base, model="deepseek-v3-250324", max_retries=2, timeout=45)
        optimizer_llm = OpenAIClient(api_key=api_key, api_base=api_base, model="deepseek-v3-250324", max_retries=2, timeout=90) # Use a powerful model for suggesting actions
    except Exception as e:
         logger.error(f"Failed to initialize OpenAIClient: {e}", exc_info=True)
         return

    # --- Dataset ---
    data_file_path = pathlib.Path("examples/dummy_data.jsonl")
    if not data_file_path.exists():
        logger.warning(f"Data file '{data_file_path}' not found. Attempting to create dummy data.")
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
                {"input": "Spell 'banana'.", "reference": "banana"},
                {"input": "Is the sky blue?", "reference": "yes"}, # Add more for batching
                {"input": "First US president?", "reference": "George Washington"},
                {"input": "What is 3*3?", "reference": "9"},
            ]
            data_file_path.parent.mkdir(exist_ok=True)
            with open(data_file_path, 'w') as f:
                for item in dummy_content:
                    f.write(f"{item}\n".replace("'", '"'))
            logger.info(f"Created dummy data file at '{data_file_path}'")
        except Exception as create_e:
            logger.error(f"Could not create dummy data file: {create_e}. Exiting.")
            return

    try:
        # Load dataset as a list of dictionaries for RL optimizer
        # JsonlDataset can be adapted or just load manually
        full_dataset = []
        with open(data_file_path, 'r') as f:
            for line in f:
                 if line.strip():
                      import json
                      try:
                           full_dataset.append(json.loads(line))
                      except json.JSONDecodeError:
                           logger.warning(f"Skipping invalid JSON line: {line.strip()}")

        if not full_dataset:
            logger.error("Loaded dataset is empty.")
            return

        logger.info(f"Loaded dataset as list with {len(full_dataset)} samples.")
        # Split for train/validation if desired
        split_index = int(len(full_dataset) * 0.8)
        train_dataset_list = full_dataset[:split_index]
        validation_dataset_list = full_dataset[split_index:]
        if not train_dataset_list: # Ensure train set is not empty
            logger.warning("Training dataset split is empty, using full dataset for training.")
            train_dataset_list = full_dataset
            validation_dataset_list = None # Cannot validate if train is full set

        logger.info(f"Using {len(train_dataset_list)} samples for training, {len(validation_dataset_list) if validation_dataset_list else 0} for validation.")

    except Exception as e:
        logger.error(f"FATAL: Failed to load dataset as list: {e}", exc_info=True)
        return

    # --- Initial Template ---
    # Use a different template perhaps? Or same RCTCRE.
    initial_template_values = {
        "role": "You are an assistant.", # Using default keys from RCTCRETemplate
        "task": "Answer the question: {{input}}",
        "context": None,
        "constraints": None,
        "response_format": None,
        "examples": None
    }
    optimizable = {"TASK", "CONSTRAINTS", "ROLE"}

    initial_template = get_template(
        template_name="RCTCRE",
        initial_values=initial_template_values,
        optimizable_sections=optimizable
    )
    logger.info("Initial prompt template created (RCTCRE).")

    # --- Evaluator (Instance) ---
    # *** This assumes RLPromptOptimizer is modified to accept an Evaluator instance ***
    # *** And that its internal _evaluate methods handle calling target_llm.generate_batch ***
    accuracy_metric = ExactMatchAccuracy(case_sensitive=False)
    evaluator_instance = Evaluator(metrics=[(accuracy_metric, 1.0)])
    logger.info("Evaluator instance configured with ExactMatchAccuracy.")

    # --- Define the evaluator *function* needed by the *current* RL Optimizer ---
    # This function needs access to the target LLM and the evaluator instance.
    # Pass the target LLM client into the function's scope.
    def rl_evaluator_func(template: BasePromptTemplate, batch: List[Dict[str, Any]]) -> Tuple[float, Dict[str, MetricResult]]:
        """Evaluator function adhering to RL Optimizer's expected signature."""
        if not batch: return -float('inf'), {}
        try:
            prompts = [template.render(data) for data in batch]
            references = [data.get('reference') for data in batch]
            # *** Prediction happens here! *** Requires target_llm in scope.
            predictions = target_llm.generate_batch(prompts) # Use the target_llm from outer scope
            if len(predictions) != len(prompts):
                 logger.error("RL Eval Fn: Prediction count mismatch.")
                 return -float('inf'), {}
            # Call the evaluator instance
            score, details = evaluator_instance.evaluate(predictions, references)
            return score, details
        except Exception as e:
            logger.error(f"Error in rl_evaluator_func: {e}", exc_info=True)
            return -float('inf'), {}


    # --- Reward Calculator & Action Memory ---
    reward_calculator = create_default_reward_calculator()
    action_memory = ActionMemory(max_entries=50, decay_factor=0.95) # Larger memory?
    logger.info("Reward calculator and action memory initialized.")

    # --- Optimization Configuration for RL ---
    config = OptimizationConfig(
        # epochs=5, # Not used by RL optimizer directly, use max_steps
        batch_size=1, # Batch size for evaluation within each step
        max_steps=8, # Total optimization steps << --- RL specific
        # RL specific params (add defaults to OptimizationConfig or handle getattr)
        initial_exploration_rate=0.4,
        min_exploration_rate=0.05,
        max_exploration_rate=0.7,
        exploration_increase_patience=3, # Increase exploration if no improvement for 3 steps
        initial_temperature=1.0,
        final_temperature=0.01,
        temperature_decay_steps=15, # Decay temperature over 15 steps
        early_stopping_patience=5, # Stop if no improvement for 5 steps
        target_score=0.95, # Stop if best score reaches 0.95
        # Monitoring/Checkpointing Config
        use_live_monitor=True,
        generate_report_at_end=True,
        tracker_config={
            'experiment_name': f'rl_optimizer_test_{run_timestamp}',
            'save_dir': str(log_dir),
            'save_history': True,
            'autosave_interval': 5,
        }
    )
    logger.info(f"RL Optimization config created: {config}")
    logger.info(f"Monitoring enabled: Live={config.use_live_monitor}, Report={config.generate_report_at_end}")
    logger.info(f"Tracker output directory: {log_dir}")


    # --- 2. Initialize RL Optimizer ---
    try:
        # *** Pass the evaluator FUNCTION here, matching the current RL Optimizer signature ***
        optimizer = RLPromptOptimizer(
            target_llm_client=target_llm,       # Pass target LLM (used internally for evaluation)
            optimizer_llm_client=optimizer_llm,  # LLM for suggesting actions
            initial_template=initial_template,
            evaluator=rl_evaluator_func,         # Pass the function defined above
            config=config,
            reward_calculator=reward_calculator,
            action_memory=action_memory
        )
        logger.info("RLPromptOptimizer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RLPromptOptimizer: {e}", exc_info=True)
        return

    # --- 3. Run Optimization ---
    logger.info("--- Starting RL Optimization Run ---")
    opt_start_time = time.time()
    final_result: Optional[RLOptimizationResult] = None

    try:
        # Run optimize with the training data list
        # Pass validation data list if available
        final_result = optimizer.optimize(train_dataset_list, validation_dataset_list)

        opt_end_time = time.time()
        logger.info("--- RL Optimization Finished ---")
        logger.info(f"Total optimization time: {opt_end_time - opt_start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"An error occurred during the RL optimization process: {e}", exc_info=True)
        try:
             optimizer.tracker.save()
             logger.info("Saved tracker state after error during RL optimization.")
        except Exception as save_e:
             logger.error(f"Could not save tracker state after error: {save_e}")
        return # Exit after error

    # --- 4. Output Results & Test Monitoring ---
    if final_result:
        print("\n" + "="*30)
        print("--- RL Optimization Results Summary ---")
        summary = final_result.get_summary()
        pprint.pprint(summary)

        print("\nBest Prompt Template Found:")
        print(final_result.best_template.render(data={}, ignore_missing_variables=True))

        print("\nAction Memory Statistics:")
        pprint.pprint(final_result.action_memory.get_statistics())

        # --- Monitoring Output ---
        print("\n--- Monitoring & Tracking Info ---")
        if final_result.tracker_summary:
             print("Tracker Summary:")
             pprint.pprint(final_result.tracker_summary)
        else:
             print("Tracker summary not available in result.")

        # Report was generated automatically if config flag was set
        if config.generate_report_at_end:
            print(f"\nReport should have been generated in: {log_dir}")

        # Example: Print info from the last *step* recorded in RL history
        if final_result.history:
             last_step_hist = final_result.history[-1]
             print(f"\nInfo from last step in RL history (Step {last_step_hist.get('step','?')})")
             print(f"  Action: {last_step_hist.get('action', {}).get('action_type', '?')} on '{last_step_hist.get('action', {}).get('target_section', '?')}'")
             print(f"  Reward: {last_step_hist.get('reward', 'N/A'):.2f}")
             print(f"  Score Before: {last_step_hist.get('before_score', 'N/A'):.4f}")
             print(f"  Score After: {last_step_hist.get('after_score', 'N/A'):.4f}")
             print(f"  Accepted: {last_step_hist.get('accepted', '?')}")
             print(f"  Best Score So Far: {last_step_hist.get('best_score', 'N/A'):.4f}")
             print(f"  Validation Score (if run): {last_step_hist.get('validation_score', 'N/A')}")


    else:
        logger.error("RL Optimization did not produce a final result.")
        print("\nRL Optimization did not complete successfully.")
        return


    # --- 5. Test Checkpointing ---
    print("\n" + "="*30)
    print("--- Checkpointing Test (RL Optimizer) ---")
    checkpoint_file = checkpoint_dir / "rl_optimizer_checkpoint.pkl"

    # a) Save checkpoint
    try:
        logger.info(f"Attempting to save RL checkpoint to: {checkpoint_file}")
        optimizer.save_optimizer_checkpoint(str(checkpoint_file)) # Use the inherited method
        if checkpoint_file.exists():
             print(f"RL Checkpoint saved successfully to {checkpoint_file}")
        else:
             print(f"RL Checkpoint file was not created at {checkpoint_file}")
             return
    except Exception as e:
        logger.error(f"Error during RL save_optimizer_checkpoint: {e}", exc_info=True)
        print(f"\nError saving RL checkpoint: {e}")
        return

    # b) Load checkpoint into a *new* RL optimizer instance
    config_for_load = OptimizationConfig(tracker_config={'save_dir': str(test_output_dir / "logs_loaded_rl")})
    log_dir_loaded = test_output_dir / "logs_loaded_rl"
    log_dir_loaded.mkdir(parents=True, exist_ok=True)

    # Create new instances of components needed for init
    new_action_memory = ActionMemory() # Create fresh memory for loaded instance

    try:
        logger.info("Initializing a new RL optimizer instance for loading...")
        loaded_optimizer = RLPromptOptimizer(
            target_llm_client=target_llm,       # Re-use client
            optimizer_llm_client=optimizer_llm,  # Re-use client
            initial_template=get_template("RCTCRE"), # Basic initial template
            evaluator=rl_evaluator_func,         # Re-use evaluator function wrapper
            config=config_for_load,              # New config
            reward_calculator=create_default_reward_calculator(), # New reward calc
            action_memory=new_action_memory      # New action memory
        )

        logger.info(f"Attempting to load RL checkpoint from: {checkpoint_file}")
        loaded_optimizer.load_optimizer_checkpoint(str(checkpoint_file))
        print("RL Checkpoint loaded successfully into new optimizer instance.")

        # c) Verify loaded state (optional, basic checks)
        print("\nVerifying loaded RL state (basic checks):")
        print(f"  Loaded best score: {loaded_optimizer._best_score:.4f} (Original best: {optimizer._best_score:.4f})")
        # Compare template hash
        original_best_hash = hash(optimizer._best_template.render({},ignore_missing_variables=True))
        loaded_best_hash = hash(loaded_optimizer._best_template.render({},ignore_missing_variables=True))
        print(f"  Loaded best template hash matches original: {original_best_hash == loaded_best_hash}")
        # Check if ActionMemory was loaded (basic size check)
        print(f"  Loaded action memory size: {len(loaded_optimizer.action_memory.memory)} (Original: {len(optimizer.action_memory.memory)})")
        # Check exploration context (e.g., rate)
        print(f"  Loaded exploration rate: {loaded_optimizer.exploration_context.get('exploration_rate'):.2f}")

        # d) Optionally run a small evaluation with the loaded optimizer
        print("\nRunning one evaluation step with loaded RL optimizer...")
        try:
             # Evaluate the loaded 'current_template' on one batch
             first_batch = train_dataset_list[:config.batch_size] # Get first train batch
             loaded_score, _ = loaded_optimizer._evaluate_template_on_batch(loaded_optimizer.current_template, first_batch)
             print(f"  Score using loaded current_template on first training batch: {loaded_score:.4f}")
        except Exception as eval_e:
             logger.error(f"Error during evaluation with loaded RL optimizer: {eval_e}", exc_info=True)
             print(f"  Error during evaluation with loaded RL optimizer: {eval_e}")


    except FileNotFoundError:
         logger.error(f"RL Checkpoint file not found for loading: {checkpoint_file}")
         print(f"\nERROR: RL Checkpoint file {checkpoint_file} not found for loading.")
    except Exception as e:
        logger.error(f"Error during RL checkpoint loading or verification: {e}", exc_info=True)
        print(f"\nError loading/verifying RL checkpoint: {e}")

    # --- 6. Cleanup (Optional) ---
    # print(f"\nRL Test finished. Output files are in: {test_output_dir.absolute()}")
    # print("You may want to manually delete this directory after inspection.")


if __name__ == "__main__":
    run_rl_optimizer_test()