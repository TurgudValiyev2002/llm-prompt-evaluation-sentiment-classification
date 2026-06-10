# Prompt Evaluation for Sentiment Classification

## Motivation

Prompt design changes model behavior, but prompt quality should not be judged by intuition alone. In this project, we built a small evaluation workflow for sentiment classification so that different prompt styles can be compared with the same examples and the same metrics.

## Project Goal

We compared three prompt-style classification policies:

1. A short label prompt.
2. A cautious prompt.
3. A balanced reasoning prompt.

The goal was to practice prompt evaluation: define labeled examples, run each prompt policy, measure accuracy, and inspect mistakes.

## Dataset

We used a small controlled set of 15 sentences: 5 positive, 5 negative, and 5 neutral. The examples are simple on purpose, because the focus is the evaluation method rather than dataset size.

## Tools

Python, pandas, scikit-learn metrics, and matplotlib.

## Method

Each prompt style was represented by a transparent keyword-based classifier. This does not replace a real LLM evaluation, but it makes the behavior inspectable and reproducible without external API access.

## Hyperparameters

No model was trained. The fixed settings were the three prompt policies and the three output labels: positive, neutral, and negative.

## Results

| Prompt Policy | Accuracy |
|---|---:|
| short_label_prompt | 1.0000 |
| clinical_cautious_prompt | 0.9333 |
| balanced_reasoning_prompt | 0.9333 |

The result files include the labeled examples, all predictions, prompt-level metrics, classification reports, confusion matrices, and an accuracy figure.

## Interpretation

The short label prompt reached the best accuracy on this small set. This does not prove it is generally better. It means the examples matched the vocabulary of the short prompt very well. The cautious and balanced prompts made fewer direct assumptions, but they also missed some examples.

The important lesson is methodological: prompt evaluation needs more than a few examples. A small set is useful for debugging, but a serious conclusion requires larger and more diverse data.

## Conclusion

We built a simple but clear prompt-evaluation pipeline. The next step would be to replace the transparent keyword policies with real LLM outputs and test on a public sentiment benchmark.

## How To Run

```bash
pip install -r requirements.txt
python 1_prompt_evaluation.py
```
