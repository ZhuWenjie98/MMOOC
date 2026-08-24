# Default prompts for different judge types
BINARY_JUDGE_PROMPT = """You are an evaluator assessing answer correctness for vision-language QA.

# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{prediction}
```

# Evaluation Rules
1. **Extract the final answer** from both ground truth and prediction (ignore reasoning/process text).

2. **For NEGATED answers** (ground truth starts with "The X is not", "There is no X", "X is not", "No X", etc.):
   - If the prediction ALSO denies/negates the same entity or premise, score {positive}.
   - Example: GT="The man is not wearing a baseball cap", Prediction="The man is not wearing a cap" → {positive}
   - Example: GT="There is no smartphone in his hand", Prediction="He is not holding a phone" → {positive}

3. **For AFFIRMATIVE answers** (ground truth states what IS present):
   - Score {positive} if prediction semantically matches, even with different wording.
   - Score {negative} if prediction contradicts or states the opposite.

4. **General rules**:
   - Ignore minor formatting differences, capitalization, or spacing.
   - For multiple-choice: match if option letter or content matches.
   - Treat numerical answers as correct if within reasonable precision.
   - For units: both value AND unit must be correct.
   - "I can't answer" / "refuse" should match if GT is also a refusal.

# Decision
Output ONLY {positive} or {negative} with no explanation."""


COMPARATIVE_JUDGE_PROMPT = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of {min_score} to {max_score}, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

[Question]
{question}

{context_section}

[Assistant 1]
{response1}
[End of Assistant 1]

[Assistant 2]
{response2}
[End of Assistant 2]

[System]
{evaluation_instruction}"""


CORRECTNESS_JUDGE_PROMPT = """You are given a question, the solution and the correct answer. Please determine if the solution matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \\boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the solution and the correct answer.
The process or reasoning leading to the Solution is irrelevant, ONLY the correctness of the result matters.
Return only "{positive}" if the solution is correct or "{negative}" if it is incorrect.
Only return "{positive}" or "{negative}" with no additional text or formatting.

Question: 
{question}
--------------------------------
Correct Answer:
{answer}
--------------------------------
Solution: 
{prediction}
--------------------------------"""
