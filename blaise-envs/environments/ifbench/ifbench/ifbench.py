"""IFBench RL environment for precise instruction following.

This environment implements the IFBench benchmark for evaluating and training
language models on verifiable instruction following constraints.

The reward function is based on Section 5 of the IFBench paper (arXiv:2507.02833):
- Verifiable reward: Binary signal for whether each constraint is satisfied
- Combined reward: Combines verifiable reward with a judge model signal
  to balance constraint following and response quality.

Reference implementation:
https://github.com/PrimeIntellect-ai/research-environments/tree/main/environments/ifbench

Reward Hacking Mitigation (Appendix E):
Uses verifiers' JudgeRubric with Gemini to evaluate response quality independent
of constraint satisfaction. The combined reward formula (Equation 2):
    F_i = V_i + bonus     if V_i > 0 and S_i > α
    F_i = V_i - penalty   if V_i > 0 and S_i ≤ α
    F_i = V_i             if V_i ≤ 0

Where:
    - V_i is the verifiable reward (constraint satisfaction)
    - S_i is the judge model score (scaled to 0.1-1.0)
    - α is the threshold (default 0.7 on 0.1-1.0 scale)
"""

import logging
import os
import re
from typing import Any

import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI

from .utils import (
    InputExample,
    test_instruction_following_strict,
    test_instruction_following_loose,
)

logger = logging.getLogger(__name__)


# LLM-as-judge prompt from Appendix C of the IFBench paper
# https://arxiv.org/html/2507.02833v3#AC
# Modified to work with JudgeRubric's template format
IFBENCH_JUDGE_PROMPT = """Evaluate the response provided below to determine if it meets the specified constraints related to the following prompt. Provide an integer score from 1 to 10, taking into account its helpfulness, relevance, accuracy, depth, creativity, and how well it conforms to the constraints. Here are the criteria that you should score:
1. Helpfulness: Does the response address the user's needs and questions effectively?
2. Relevance: Is the response directly related to the context of the dialog?
3. Accuracy: Are the facts and information presented in the response correct?
4. Depth: Does the response cover the topic thoroughly, with sufficient detail?
5. Creativity: Is the response original and engaging?

Question:
```
{question}
```

Response:
```
{response}
```

Respond with only a single integer from 1 to 10."""


class IFBenchJudgeRubric(vf.JudgeRubric):
    """Judge rubric for IFBench that evaluates response quality.
    
    Extends verifiers' JudgeRubric to:
    1. Use the LLM-as-judge prompt from Appendix C of the IFBench paper
    2. Scale scores to 0.1-1.0 range
    3. Support Gemini as the judge model via OpenAI-compatible API
    
    This implements the S_i (judge score) component from Section 5/Appendix E
    of the IFBench paper for mitigating reward hacking.
    """
    
    def __init__(
        self,
        judge_model: str = "gemini-2.5-flash",
        judge_api_key_env: str = "GEMINI_API_KEY",
        judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        judge_threshold: float = 0.7,
        judge_bonus: float = 0.1,
        judge_penalty: float = 0.05,
        **kwargs,
    ):
        """Initialize IFBench judge rubric.
        
        Args:
            judge_model: Model to use for judging (default: gemini-2.0-flash).
            judge_api_key_env: Environment variable for judge API key.
            judge_base_url: Base URL for judge API (Gemini OpenAI-compatible).
            judge_threshold: Threshold α for judge score (0.1-1.0 scale).
            judge_bonus: Bonus when judge score exceeds threshold.
            judge_penalty: Penalty when judge score below threshold.
        """
        # Create Gemini client via OpenAI-compatible API
        api_key = os.getenv(judge_api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable {judge_api_key_env} must be set for judge model. "
                f"Get your API key from https://aistudio.google.com/app/apikey"
            )
        
        judge_client = AsyncOpenAI(
            api_key=api_key,
            base_url=judge_base_url,
        )
        
        super().__init__(
            judge_client=judge_client,
            judge_model=judge_model,
            judge_prompt=IFBENCH_JUDGE_PROMPT,
            judge_sampling_args={"temperature": 0, "max_tokens": 10},
            **kwargs,
        )
        
        self.judge_threshold = judge_threshold
        self.judge_bonus = judge_bonus
        self.judge_penalty = judge_penalty
        
        # Add the combined reward function
        self.add_reward_func(self._judge_quality_reward, weight=1.0)
    
    async def _judge_quality_reward(
        self,
        prompt: vf.Messages,
        completion: vf.Messages,
        answer: str,
        state: vf.State,
        **kwargs,
    ) -> float:
        """Compute judge-based quality reward using JudgeRubric's judge method.
        
        Calls the parent JudgeRubric's judge() method to get the quality score,
        then normalizes it to 0.1-1.0 scale.
        
        Args:
            prompt: The input messages.
            completion: The model's output messages.
            answer: Ground truth answer (unused, but required by interface).
            state: Rollout state for caching judge responses.
            
        Returns:
            Normalized judge score in range [0.1, 1.0].
        """
        # Call parent's judge method
        judge_response = await self.judge(
            prompt=prompt,
            completion=completion,
            answer=answer,
            state=state,
        )
        
        # Parse score from response (expecting integer 1-10)
        match = re.search(r'\b(\d+)\b', judge_response or "")
        if match:
            score = int(match.group(1))
            score = max(1, min(10, score))  # Clamp to 1-10
        else:
            score = 5  # Default
        
        # Normalize to 0.1-1.0 scale
        normalized_score = score / 10.0
        
        # Store in state for use by combined reward logic
        state["judge_score"] = normalized_score
        
        return normalized_score


def create_verifiable_rubric(
    strict: bool = True,
    reward_weight: float = 1.0,
    reward_multiplier: float = 1.0,
) -> vf.Rubric:
    """Create a rubric for IFBench verifiable constraint checking.
    
    This implements the V_i (verifiable reward) component from Section 5
    of the IFBench paper.
    
    Args:
        strict: Whether to use strict verification.
        reward_weight: Weight applied to constraint rewards.
        reward_multiplier: Multiplier applied to rewards.
        
    Returns:
        A Rubric instance for verifiable constraint checking.
    """
    
    @vf.reward_function
    async def verifiable_reward(
        completion: vf.Messages,
        prompt: vf.Messages,
        info: dict[str, Any],
        state: vf.State,
        **kwargs,
    ) -> float:
        """Compute verifiable reward for constraint satisfaction.
        
        Args:
            completion: The model's output messages.
            prompt: The input messages.
            info: Dataset info containing constraint metadata.
            state: Rollout state for storing metrics.
            
        Returns:
            Proportion of constraints satisfied (0.0 to 1.0).
        """
        # Extract completion text
        if isinstance(completion, list) and completion:
            completion_text = completion[-1].get("content", "")
        else:
            completion_text = str(completion)
        
        # Extract prompt text
        if isinstance(prompt, list) and prompt:
            prompt_text = prompt[-1].get("content", "")
        else:
            prompt_text = str(prompt)
        
        # Get constraint info from dataset
        instruction_id_list = info.get("instruction_id_list", [])
        constraint_kwargs = info.get("kwargs", [])
        key = info.get("key", 0)
        
        # Create input example for verification
        inp = InputExample(
            key=key,
            instruction_id_list=instruction_id_list,
            prompt=prompt_text,
            kwargs=constraint_kwargs,
        )
        
        # Test instruction following using strict or loose verification
        if strict:
            output = test_instruction_following_strict(inp, completion_text)
        else:
            output = test_instruction_following_loose(inp, completion_text)
        
        # Calculate verifiable reward
        num_constraints = len(output.follow_instruction_list)
        num_satisfied = sum(output.follow_instruction_list)
        
        if num_constraints > 0:
            base_reward = num_satisfied / num_constraints
        else:
            base_reward = 1.0
        
        verifiable_reward_value = base_reward * reward_weight * reward_multiplier
        
        # Store metrics in state
        state["verifiable_reward"] = verifiable_reward_value
        state["is_correct"] = output.follow_all_instructions
        state["follow_instruction_list"] = output.follow_instruction_list
        state["num_constraints"] = num_constraints
        state["num_satisfied"] = num_satisfied
        state["constraint_accuracy"] = base_reward
        
        return verifiable_reward_value
    
    return vf.Rubric(funcs=[verifiable_reward])


def create_combined_rubric(
    strict: bool = True,
    reward_weight: float = 1.0,
    reward_multiplier: float = 1.0,
    judge_model: str = "gemini-2.5-flash",
    judge_api_key_env: str = "GEMINI_API_KEY",
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
    judge_threshold: float = 0.7,
    judge_bonus: float = 0.1,
    judge_penalty: float = 0.05,
) -> vf.RubricGroup:
    """Create a combined rubric using verifiers' RubricGroup.
    
    Combines the verifiable reward (V_i) with the judge score (S_i)
    using the formula from Appendix E (Equation 2) of the IFBench paper.
    
    Args:
        strict: Whether to use strict verification.
        reward_weight: Weight for verifiable reward.
        reward_multiplier: Multiplier for verifiable reward.
        judge_model: Model for judging (default: gemini-2.5-flash).
        judge_api_key_env: Environment variable for judge API key.
        judge_base_url: Base URL for judge API.
        judge_threshold: Threshold α for judge score (0.1-1.0 scale).
        judge_bonus: Bonus when judge score exceeds threshold.
        judge_penalty: Penalty when judge score below threshold.
        
    Returns:
        A RubricGroup combining verifiable and judge rubrics.
    """
    # Create verifiable reward rubric
    verifiable_rubric = create_verifiable_rubric(
        strict=strict,
        reward_weight=reward_weight,
        reward_multiplier=reward_multiplier,
    )
    
    # Create judge rubric using IFBenchJudgeRubric
    judge_rubric = IFBenchJudgeRubric(
        judge_model=judge_model,
        judge_api_key_env=judge_api_key_env,
        judge_base_url=judge_base_url,
        judge_threshold=judge_threshold,
        judge_bonus=judge_bonus,
        judge_penalty=judge_penalty,
    )
    
    # Add the combined reward adjustment function to verifiable rubric
    @vf.reward_function
    async def combined_reward_adjustment(state: vf.State, **kwargs) -> float:
        """Apply Equation 2 adjustment based on judge score.
        
        F_i = V_i + bonus     if V_i > 0 and S_i > α
        F_i = V_i - penalty   if V_i > 0 and S_i ≤ α
        F_i = V_i             if V_i ≤ 0
        
        This runs after both verifiable and judge rewards are computed,
        and adjusts the final reward based on the combined formula.
        """
        verifiable_reward = state.get("verifiable_reward", 0.0)
        judge_score = state.get("judge_score", 0.5)
        
        if verifiable_reward > 0:
            if judge_score > judge_threshold:
                adjustment = judge_bonus
                state["reward_adjustment"] = "bonus"
            else:
                adjustment = -judge_penalty
                state["reward_adjustment"] = "penalty"
        else:
            adjustment = 0.0
            state["reward_adjustment"] = "none"
        
        state["combined_adjustment"] = adjustment
        return adjustment
    
    verifiable_rubric.add_reward_func(combined_reward_adjustment, weight=1.0)
    
    # Combine using RubricGroup - rewards are summed
    # verifiable_reward + judge_score + adjustment
    return vf.RubricGroup(rubrics=[verifiable_rubric, judge_rubric])


def load_environment(
    dataset_name: str = "allenai/IFBench",
    split: str = "test",
    strict: bool = True,
    reward_weight: float = 1.0,
    reward_multiplier: float = 1.0,
    judge_model: str = "gemini-2.5-flash",
    judge_api_key_env: str = "GEMINI_API_KEY",
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
    judge_threshold: float = 0.7,
    judge_bonus: float = 0.1,
    judge_penalty: float = 0.05,
    max_examples: int = -1,
    multi_turn: bool = False,
    **kwargs,
) -> vf.Environment:
    """Load the IFBench environment for instruction following evaluation.

    This environment implements the IFBench benchmark from the paper
    "Generalizing Verifiable Instruction Following" (arXiv:2507.02833).

    The benchmark evaluates language models on 58 new, diverse, and challenging
    verifiable constraints including counting, formatting, sentence/word/character
    manipulations, and copying.

    The reward function uses a RubricGroup combining:
    1. Verifiable reward: Constraint satisfaction proportion (V_i)
    2. Judge reward: Gemini-based quality score (S_i)
    3. Combined adjustment: Bonus/penalty based on Equation 2 from Appendix E

    This implements the "Mitigating Reward Hacking" setup from Section 5/Appendix E,
    using verifiers' JudgeRubric and RubricGroup.

    Args:
        dataset_name: HuggingFace dataset name for IFBench data.
        split: Dataset split to use ("test", "train", etc.).
        strict: Whether to use strict verification.
        reward_weight: Weight applied to constraint rewards.
        reward_multiplier: Multiplier applied to rewards.
        judge_model: Model for judging (default: gemini-2.5-flash).
        judge_api_key_env: Environment variable for judge API key.
        judge_base_url: Base URL for judge API.
        judge_threshold: Threshold α for judge score (0.1-1.0 scale).
        judge_bonus: Bonus when judge score exceeds threshold.
        judge_penalty: Penalty when judge score below threshold.
        max_examples: Maximum examples to load (-1 for all).
        multi_turn: Whether to use multi-turn format.
        **kwargs: Additional arguments for Environment constructor.

    Returns:
        A verifiers Environment configured for IFBench evaluation.

    Example:
        ```python
        import verifiers as vf
        from ifbench import load_environment

        # Load environment (requires GEMINI_API_KEY)
        env = load_environment()

        # Custom judge settings
        env = load_environment(
            judge_model="gemini-2.5-flash",
            judge_threshold=0.7,
        )

        # Run evaluation
        results = vf.eval(env, model="gpt-4")
        ```

    References:
        - Paper: https://arxiv.org/html/2507.02833v3
        - Section 5: Reward Hacking and the Instruction Hierarchy
        - Appendix C: LLM-as-judge Prompt
        - Appendix E: Mitigating Reward Hacking (Equation 2)
        - Verifiers docs: https://docs.primeintellect.ai/verifiers/environments
    """
    # Load dataset from HuggingFace
    dataset = load_dataset(dataset_name, split=split)

    # Limit examples if specified
    if max_examples > 0 and len(dataset) > max_examples:
        dataset = dataset.select(range(max_examples))

    # Create combined rubric with verifiable + judge rewards
    rubric = create_combined_rubric(
        strict=strict,
        reward_weight=reward_weight,
        reward_multiplier=reward_multiplier,
        judge_model=judge_model,
        judge_api_key_env=judge_api_key_env,
        judge_base_url=judge_base_url,
        judge_threshold=judge_threshold,
        judge_bonus=judge_bonus,
        judge_penalty=judge_penalty,
    )

    # Define prompt formatter
    def format_prompt(example: dict[str, Any]) -> list[dict[str, str]]:
        """Format an IFBench example into chat messages."""
        if multi_turn and "response" in example:
            messages = [
                {"role": "user", "content": example["task"]},
                {"role": "assistant", "content": example["response"]},
                {"role": "user", "content": example["constraint"]},
            ]
        else:
            messages = [{"role": "user", "content": example["prompt"]}]
        return messages

    # Define info extractor that passes constraint metadata
    def extract_info(example: dict[str, Any]) -> dict[str, Any]:
        """Extract constraint metadata for reward computation."""
        return {
            "instruction_id_list": example["instruction_id_list"],
            "kwargs": example["kwargs"],
            "prompt": example["prompt"],
            "key": example.get("key", 0),
        }

    # Create and return the environment
    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        prompt_formatter=format_prompt,
        info_extractor=extract_info,
        **kwargs,
    )
