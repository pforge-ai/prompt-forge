# prompt_forge/updaters/llm_updater.py

import json
import logging
import copy
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

# 导入核心基础类型和基类
from ptforge.core.base import (
    BaseLLMClient,
    BasePromptTemplate,
    MetricResult,
    UpdateGranularity,
)
from .base_updater import BasePromptUpdater

logger = logging.getLogger(__name__)

# --- (TypedDicts remain the same) ---
class ModificationSuggestion(TypedDict):
    target_section: str
    change_description: str
    priority_weight: Optional[float]

class OptimizationDirections(TypedDict):
    analysis_summary: Optional[str]
    suggested_modifications: List[ModificationSuggestion]

# --- (Analyzer Meta Prompt Template remains the same as v5) ---
DEFAULT_ANALYZER_META_PROMPT = """
You are an expert Prompt Analyzer. Your goal is to analyze the performance of a prompt template and suggest specific, actionable directions for improvement based on feedback.

**Task Goal:**
{task_description}

**Current Prompt Template Structure:**
(Sections: {section_list})
```
{current_prompt_rendered}
```

**Recent Performance Feedback:**
* Overall Score (Higher is better): {overall_score:.4f}
* Detailed Metrics:
{detailed_metrics_formatted}
* (Optional) Examples of recent poor performance (Input -> Prediction -> Reference):
{bad_examples_formatted}

**Analysis Instructions (Based on Update Level: {update_level_name}):**
1. Analyze the feedback and the current prompt structure in relation to the task goal.
2. Identify potential reasons for suboptimal performance.
3. Based on the update level '{update_level_name}', propose specific modifications. {level_specific_instructions}
4. **Constraint:** Your suggested modifications should focus on improving task performance based on input. **Do NOT suggest explicitly matching or copying the 'reference' data format or content.**

**Output Format Constraint:**
You MUST output your analysis and suggestions as a **single JSON object** matching this structure:
```json
{{
  "analysis_summary": "Brief analysis of the key issues found.",
  "suggested_modifications": [
    {{
      "target_section": "SECTION_NAME_TO_MODIFY",
      "change_description": "Clear instruction on how the Editor should modify this section (e.g., 'Make the instruction more concise', 'Add a constraint about output length', 'Rephrase to sound more authoritative').",
      "priority_weight": 0.8
    }},
    // ... more modification suggestions if needed
  ]
}}
```
Provide ONLY the JSON object. Do not include explanations outside the JSON. Priority weight is optional (0.0-1.0).
Include suggestions only for sections that need changes based on your analysis and the allowed update level. **Only suggest modifications for sections listed as optimizable earlier in the instructions.**

**Your Analysis and Modification Directions (JSON object only):**
"""

# --- 修改 Editor Meta-Prompt ---
# --- Modified Editor Meta-Prompt ---
DEFAULT_EDITOR_META_PROMPT = """
You are an expert Prompt Editor. Your task is to revise the provided prompt template sections based on specific modification directions.

**Current Prompt Template Sections:**
```json
{current_sections_json}
```

**Modification Directions:**
```json
{modification_directions_json}
```

**Context: Originally Optimizable Sections:**
The sections initially marked as available for optimization were: [{optimizable_sections_list}]

**Editing Instructions:**
1. Carefully review each modification direction provided above.
2. Apply the changes described in `change_description` to the corresponding `target_section`.
3. Ensure the revised content aligns with the instruction's intent.
4. **Important:** While following the directions, primarily focus your edits on the sections listed above as originally optimizable. If a direction targets a non-optimizable section, apply the change cautiously or minimally if it seems essential based on the direction.

**Output Format Constraint:**
You MUST output the results as a **single JSON object**.
The keys of the JSON object should be the names of ONLY the sections you actually modified.
The values should be the **complete new string content** for those modified sections.
Do NOT output anything else besides the JSON object.

**Your Revised Sections (JSON object only):**
"""


class LLMBasedUpdater(BasePromptUpdater):
    """
    (重构为两阶段) 使用 LLM 通过分析反馈生成优化方向，然后应用这些方向来修改 Prompt。
    在应用最终修改前会校验 Section 是否可优化。Editor Prompt 现在也包含可优化列表信息。
    (Refactored to Two-Step) Uses an LLM to first generate optimization directions
    by analyzing feedback, and then applies those directions to modify the prompt.
    Verifies section optimizability before applying final changes. Editor prompt now
    also includes the list of optimizable sections.
    """

    def __init__(
        self,
        optimizer_llm_client: BaseLLMClient,
        analyzer_meta_prompt: Optional[str] = None,
        editor_meta_prompt: Optional[str] = None,
        num_bad_examples_to_show: int = 3,
        task_description: Optional[str] = None,
    ):
        # ... (__init__ remains the same) ...
        self.optimizer_llm_client = optimizer_llm_client
        self.analyzer_meta_prompt_template = analyzer_meta_prompt or DEFAULT_ANALYZER_META_PROMPT
        self.editor_meta_prompt_template = editor_meta_prompt or DEFAULT_EDITOR_META_PROMPT
        self.num_bad_examples_to_show = num_bad_examples_to_show
        self.task_description = task_description or "No specific task description provided."
        logger.info(f"LLMBasedUpdater (Two-Step) initialized with Optimizer LLM: {type(optimizer_llm_client).__name__}")
        logger.info(f"Task Description: {self.task_description}")


    # --- (Helper methods _get_level_specific_instructions_v2, _format_detailed_metrics, _format_bad_examples, _parse_json_from_llm remain the same as v5) ---
    def _get_level_specific_instructions_v2(
        self, update_level: UpdateGranularity, template: BasePromptTemplate
    ) -> str:
        # ... (same as v5) ...
        optimizable_sections_list = sorted(list(template.get_optimizable_sections().keys()))
        optimizable_sections_str = ", ".join(optimizable_sections_list) if optimizable_sections_list else "None"
        base_instruction = f"Only suggest modifications for the following optimizable sections: [{optimizable_sections_str}]. "
        if update_level == UpdateGranularity.MICRO:
            return base_instruction + ("Suggest only minimal, targeted text changes (like word choice, minor rephrasing). "
                                       "Do not make structural changes.")
        elif update_level == UpdateGranularity.SECTION_REPHRASE:
            return base_instruction + ("You can suggest rephrasing sentences or paragraphs "
                                       "to improve performance, but maintain the original purpose "
                                       "of each section. Do not suggest adding/removing sections.")
        elif update_level == UpdateGranularity.STRUCTURE_ADAPT:
            examples_section = getattr(template, 'EXAMPLES', None) # Use template specific constant if available
            can_edit_examples = examples_section and examples_section in optimizable_sections_list
            instr = base_instruction + "You can suggest significant content rewriting. "
            if can_edit_examples:
                 instr += f"You may also suggest adding, removing, or modifying examples in the '{examples_section}' section. "
            return instr
        elif update_level == UpdateGranularity.FULL_REWRITE:
             return base_instruction + ("You may suggest complete reformulation of these sections "
                                        "or changes to the prompting strategy to maximize the score.")
        else:
            return "No changes should be suggested (Update Level FIXED)."

    def _format_detailed_metrics(self, detailed_results: Dict[str, MetricResult]) -> str:
        # ... (same as v5) ...
        lines = []
        for name, result in detailed_results.items():
            details_str = f" (Details: {result.details})" if result.details else ""
            lines.append(f"  - {name}: {result.score:.4f}{details_str}")
        return "\n".join(lines) if lines else "N/A"

    def _format_bad_examples(self, batch_data: List[Dict[str, Any]], predictions: List[str]) -> str:
        # ... (same as v5) ...
        num_to_show = min(self.num_bad_examples_to_show, len(batch_data))
        if num_to_show == 0: return "N/A"
        lines = []
        for i in range(num_to_show):
             inp = batch_data[i].get('input', 'N/A')
             pred = predictions[i] if i < len(predictions) else 'N/A'
             ref = batch_data[i].get('reference', 'N/A')
             lines.append(f"  Example {i+1}:\n    Input: {inp}\n    Prediction: {pred}\n    Reference: {ref}")
        return "\n".join(lines)

    def _parse_json_from_llm(self, llm_response: str, expected_type: type = dict) -> Optional[Any]:
        try:
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[len("```json"):].strip()
            elif cleaned_response.startswith("```"):
                 cleaned_response = cleaned_response[len("```"):].strip()
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-len("```")].strip()
            if not cleaned_response:
                 logger.warning("LLM returned an empty response after cleaning.")
                 return None
            parsed_json = json.loads(cleaned_response)
            if not isinstance(parsed_json, expected_type):
                 logger.warning(f"LLM did not return expected type {expected_type}, got {type(parsed_json)}.")
                 return None
            return parsed_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM: {e}\nResponse was: {llm_response}")
            return None
        except Exception as e:
            logger.error(f"Error processing LLM response: {e}\nResponse was: {llm_response}", exc_info=True)
            return None


    # --- Stage 1: Analyze Feedback ---
    def _analyze_feedback(
        self,
        current_template: BasePromptTemplate,
        batch_data: List[Dict[str, Any]],
        predictions: List[str],
        detailed_results: Dict[str, MetricResult],
        overall_score: float,
        update_level: UpdateGranularity,
        **kwargs,
        ) -> Optional[OptimizationDirections]:
        """Calls the AnalyzerLLM to get optimization directions."""
        # ... (logic remains the same as v5) ...
        current_prompt_rendered = current_template.render({}, ignore_missing_variables=True)
        section_list = ", ".join(current_template.list_sections())
        level_specific_instructions_with_context = self._get_level_specific_instructions_v2(update_level, current_template)
        detailed_metrics_formatted = self._format_detailed_metrics(detailed_results)
        bad_examples_formatted = self._format_bad_examples(batch_data, predictions)

        analyzer_prompt = self.analyzer_meta_prompt_template.format(
            current_prompt_rendered=current_prompt_rendered,
            section_list=section_list,
            task_description=self.task_description,
            overall_score=overall_score,
            detailed_metrics_formatted=detailed_metrics_formatted,
            bad_examples_formatted=bad_examples_formatted,
            update_level_name=update_level.name,
            level_specific_instructions=level_specific_instructions_with_context,
        )
        try:
            logger.info("Calling OptimizerLLM (Analyzer Stage)...")
            response_text = self.optimizer_llm_client.generate(analyzer_prompt, **kwargs)
            logger.debug(f"AnalyzerLLM raw response: {response_text}")
            parsed_directions = self._parse_json_from_llm(response_text, expected_type=dict)
            if parsed_directions and "suggested_modifications" in parsed_directions and isinstance(parsed_directions["suggested_modifications"], list):
                 return cast(OptimizationDirections, parsed_directions)
            else:
                 logger.warning("AnalyzerLLM response did not contain valid 'suggested_modifications' list.")
                 return None
        except Exception as e:
            logger.error(f"Error during AnalyzerLLM call or parsing: {e}", exc_info=True)
            return None


    # --- Stage 2: Apply Directions (Modified) ---
    def _apply_directions(
        self,
        current_template: BasePromptTemplate,
        directions: OptimizationDirections,
        **kwargs,
        ) -> Optional[Dict[str, str]]:
        """Calls the EditorLLM to apply modification directions."""

        # Prepare current sections
        current_sections: Dict[str, Any] = {}
        for section_name in current_template.list_sections():
             current_sections[section_name] = current_template.get_section(section_name)

        # --- 新增：获取可优化列表字符串 ---
        # --- New: Get optimizable list string ---
        optimizable_sections_list = sorted(list(current_template.get_optimizable_sections().keys()))
        optimizable_list_str = ", ".join(optimizable_sections_list) if optimizable_sections_list else "None"


        # Format inputs for the editor prompt
        try:
            current_sections_json = json.dumps(current_sections, indent=2)
            modification_directions_json = json.dumps(directions, indent=2)
        except Exception as e:
             logger.error(f"Failed to format data for EditorLLM prompt: {e}", exc_info=True)
             return None

        # --- 修改这里：格式化 Editor Prompt 时加入可优化列表 ---
        # --- Modify here: Add optimizable list when formatting Editor Prompt ---
        editor_prompt = self.editor_meta_prompt_template.format(
             current_sections_json=current_sections_json,
             modification_directions_json=modification_directions_json,
             optimizable_sections_list=optimizable_list_str # <--- 传入列表字符串 (Pass the list string)
        )
        print(editor_prompt)

        try:
            logger.info("Calling OptimizerLLM (Editor Stage)...")
            response_text = self.optimizer_llm_client.generate(editor_prompt, **kwargs)
            logger.debug(f"EditorLLM raw response: {response_text}")

            # Parse the final modified sections JSON
            modified_sections = self._parse_json_from_llm(response_text, expected_type=dict)

            # Validate keys are valid section names and values are strings
            if modified_sections:
                 valid_section_names = set(current_template.list_sections())
                 validated_modifications: Dict[str, str] = {}
                 for section_name, new_content in modified_sections.items():
                      if section_name not in valid_section_names:
                           logger.warning(f"EditorLLM returned invalid section '{section_name}'. Ignoring.")
                           continue
                      if not isinstance(new_content, str):
                           logger.warning(f"EditorLLM provided non-string content for '{section_name}'. Converting.")
                           validated_modifications[section_name] = str(new_content)
                      else:
                           validated_modifications[section_name] = new_content
                 return validated_modifications if validated_modifications else None
            else:
                 return None

        except Exception as e:
            logger.error(f"Error during EditorLLM call or parsing: {e}", exc_info=True)
            return None


    # --- Main Propose Update Method (remains the same as v5) ---
    def propose_update(
        self,
        current_template: BasePromptTemplate,
        batch_data: List[Dict[str, Any]],
        predictions: List[str],
        detailed_results: Dict[str, MetricResult],
        overall_score: float,
        update_level: UpdateGranularity,
        **kwargs,
    ) -> BasePromptTemplate:
        """
        执行两阶段的 Prompt 更新：先分析反馈生成方向，然后根据方向修改 Prompt。
        """
        # 1. Handle FIXED level
        if update_level == UpdateGranularity.FIXED:
            logger.debug("Update level is FIXED, returning original template copy.")
            return copy.deepcopy(current_template)

        # 2. Stage 1: Analyze feedback, get directions
        optimization_directions = self._analyze_feedback(
            current_template, batch_data, predictions, detailed_results, overall_score, update_level, **kwargs
        )

        if not optimization_directions or not optimization_directions.get("suggested_modifications"):
            logger.warning("Failed to get valid optimization directions from AnalyzerLLM. No update applied.")
            return copy.deepcopy(current_template)

        logger.info(f"AnalyzerLLM suggested {len(optimization_directions['suggested_modifications'])} modifications.")
        logger.debug(f"Optimization directions: {optimization_directions}")


        # 3. Stage 2: Modify prompt based on directions
        final_modifications = self._apply_directions(
            current_template, optimization_directions, **kwargs
        )

        # 4. Apply Modifications (with final check)
        if final_modifications:
            print("**"*100, final_modifications)
            new_template = copy.deepcopy(current_template)
            applied_count = 0
            original_optimizable_sections = set(current_template.get_optimizable_sections().keys())
            logger.info(f"Applying validated changes suggested by EditorLLM.")
            for section_name, new_content in final_modifications.items():
                # Final check: Only apply if section was originally optimizable
                if section_name not in original_optimizable_sections:
                    logger.warning(f"EditorLLM suggested change for section '{section_name}' which was NOT originally marked as optimizable. Skipping this change.")
                    continue

                try:
                    new_template.update_section(section_name, new_content)
                    logger.debug(f"Applied update to optimizable section '{section_name}'.")
                    applied_count += 1
                except Exception as e:
                     logger.error(f"Error applying update to section '{section_name}': {e}", exc_info=True)

            if applied_count > 0:
                 logger.info(f"Successfully applied {applied_count} changes to optimizable sections.")
            else:
                 logger.info("No valid changes were applied to optimizable sections.")

            return new_template
        else:
            logger.warning("EditorLLM did not provide valid modifications based on directions. No update applied.")
            return copy.deepcopy(current_template)