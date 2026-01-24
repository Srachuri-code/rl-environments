# IFBench

### Overview
- **Environment ID**: `ifbench`
- **Short description**: Verifiable instruction following benchmark for evaluating and training language models on precise constraint adherence
- **Tags**: `instruction-following`, `verifiable`, `ifeval`, `train`, `eval`
- **Paper**: [Generalizing Verifiable Instruction Following (arXiv:2507.02833)](https://arxiv.org/abs/2507.02833)

### Datasets
- **Primary dataset(s)**: [allenai/IFBench](https://huggingface.co/datasets/allenai/IFBench) - 58 new verifiable constraints covering counting, formatting, sentence/word/character manipulations, and copying
- **Source links**: [GitHub](https://github.com/PrimeIntellect-ai/research-environments/tree/main/environments/ifbench)
- **Split sizes**: 300 test prompts (single-turn and multi-turn variants)

### Task
- **Type**: Single-turn (with multi-turn variant available)
- **Parser**: Standard chat completion
- **Rubric overview**: Uses verifiers' `RubricGroup` combining verifiable constraints with `JudgeRubric` (Gemini) for the combined reward from Section 5 of the IFBench paper

### Reward Function

This environment implements the combined reward function from Section 5/Appendix E of the IFBench paper (arXiv:2507.02833), which balances constraint following with response quality using Gemini as an LLM-as-judge:

```
F_i = V_i + bonus    if V_i > 0 and S_i > α    (constraint satisfied, good quality)
F_i = V_i - penalty  if V_i > 0 and S_i ≤ α    (constraint satisfied, poor quality)
F_i = V_i            if V_i ≤ 0                 (constraint not satisfied)
```

Where:
- `V_i` = verifiable reward (constraint satisfaction score, 0.0 to 1.0)
- `S_i` = judge model score (response quality, scaled to 0.1-1.0)
- `α` = threshold (default 0.7, corresponding to 7 on original 1-10 scale)
- `bonus` = reward bonus for high-quality responses (default 0.1)
- `penalty` = reward penalty for low-quality responses (default 0.05)

The judge uses the LLM-as-judge prompt from Appendix C of the paper, evaluating:
1. **Helpfulness**: Does the response address the user's needs?
2. **Relevance**: Is the response related to the dialog context?
3. **Accuracy**: Are the facts and information correct?
4. **Depth**: Does the response cover the topic thoroughly?
5. **Creativity**: Is the response original and engaging?

This combined reward helps mitigate reward hacking where models over-optimize for constraints
at the expense of response quality.

#### Implementation using Verifiers Framework

The combined reward uses verifiers' built-in components:

- **`IFBenchJudgeRubric`**: Extends `vf.JudgeRubric` with the LLM-as-judge prompt from Appendix C and Gemini integration
- **`vf.RubricGroup`**: Combines the verifiable rubric with the judge rubric, aggregating rewards
- **`vf.Rubric`**: Base rubric for the verifiable constraint checking reward function

```python
# The combined rubric architecture:
RubricGroup([
    Rubric(funcs=[verifiable_reward, combined_adjustment]),  # V_i + adjustment
    IFBenchJudgeRubric(judge_model="gemini-2.5-flash")       # S_i
])
```

### Eval

```bash
export GEMINI_API_KEY="your-gemini-api-key"

# Run eval and save results as JSONL
uv run vf-eval ifbench -s

# With custom model
uv run vf-eval ifbench -s -m gpt-4.1-mini
```

Results saved to `./outputs/` as `results.jsonl`.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"allenai/IFBench"` | HuggingFace dataset name |
| `split` | str | `"test"` | Dataset split to use |
| `strict` | bool | `True` | Use strict verification (False allows minor formatting variations) |
| `reward_weight` | float | `1.0` | Weight applied to constraint rewards |
| `reward_multiplier` | float | `1.0` | Multiplier applied to rewards |
| `judge_model` | str | `"gemini-2.5-flash"` | Gemini model for judging response quality |
| `judge_api_key_env` | str | `"GEMINI_API_KEY"` | Environment variable for Gemini API key |
| `judge_threshold` | float | `0.7` | Threshold α for judge score (0.1-1.0 scale) |
| `judge_bonus` | float | `0.1` | Bonus when judge score exceeds threshold |
| `judge_penalty` | float | `0.05` | Penalty when judge score below threshold |
| `max_examples` | int | `-1` | Limit dataset size (-1 for all) |
| `multi_turn` | bool | `False` | Use multi-turn conversation format |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (V_i + S_i + adjustment) |
| `verifiable_reward` | Constraint satisfaction score V_i (0.0 to 1.0) |
| `judge_score` | Gemini judge quality score S_i (0.1 to 1.0) |
| `reward_adjustment` | Adjustment type: "bonus", "penalty", or "none" |
| `is_correct` | Boolean - all constraints satisfied |
| `constraint_accuracy` | Proportion of satisfied constraints (0.0 to 1.0) |
| `num_constraints` | Total number of constraints in the instruction |
| `num_satisfied` | Number of constraints satisfied |
| `follow_instruction_list` | Per-constraint satisfaction list |

### Constraint Categories

IFBench includes 58 constraint types across 7 categories:

| Category | Example Constraints |
| -------- | ------------------- |
| **count** | Word count range, unique words, conjunctions, numbers, pronouns |
| **ratio** | Stop word percentage, sentence type ratios, trigram overlap |
| **words** | Alphabet cycling, palindromes, prime-length words, alliteration |
| **sentence** | Keyword placement, word count increment, sentence linking |
| **format** | Bullet points, quotes, parentheses nesting, emoji usage |
| **copy** | Request repetition, span copying, verbatim copying |
| **custom** | CSV generation, date formatting, reverse alphabetical lists |

### Verification Modes

- **Strict**: Exact constraint verification
- **Loose**: Allows minor variations (removes first/last lines, strips markdown formatting)

Both modes are available to match the evaluation setup in the original IFBench paper.

