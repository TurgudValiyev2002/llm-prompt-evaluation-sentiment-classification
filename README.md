# Prompt Evaluation for Sentiment Classification

## Motivation

Prompt design is often discussed by intuition: one prompt "feels" more careful, another "feels" more direct, and another asks for reasoning. That is not enough for research work. Even for a small task, prompts should be compared on the same labeled examples, with the same metrics, and with visible error analysis.

This project builds a small offline lab for that idea. It does not call a real LLM API. Instead, each prompt style is represented by a transparent rule-based policy so the full evaluation workflow can run locally and be inspected line by line.

## Project Goal

The goal is to compare three prompt styles for sentiment classification:

1. `short_label_prompt`: direct classification with a simple label-only instruction.
2. `clinical_cautious_prompt`: conservative classification that prefers neutral when evidence is weak.
3. `balanced_reasoning_prompt`: a policy that weighs positive and negative evidence before deciding.

The scientific question is not "which prompt is universally best?" The question is narrower: how do different prompt behaviors change accuracy, class balance, and errors on the same sentiment examples?

## Dataset

The dataset contains 30 short labeled sentences:

- 10 positive examples.
- 10 negative examples.
- 10 neutral examples.

The examples include plain sentiment, AI-system comments, engineering comments, academic statements, ambiguous wording, and mixed clauses. The dataset is intentionally small, so the results should be read as a controlled demonstration rather than a general benchmark.

## Tools

- Python
- pandas
- scikit-learn metrics
- matplotlib

## Method

The script stores the dataset, prompt templates, predictions, rationales, aggregate metrics, per-slice metrics, classification reports, confusion matrices, and error analysis files in `results/`.

Each prompt policy uses different positive/negative evidence words and a different neutral-bias setting. This approximates how prompt wording can change model behavior:

- A direct prompt may classify more aggressively.
- A cautious prompt may overuse neutral.
- A balanced prompt may be sensitive to mixed positive and negative words.

This is a useful teaching proxy, but it is not a substitute for evaluating a real LLM on a larger public dataset.

## Hyperparameters

No model is trained. The fixed experimental settings are:

- `num_examples = 30`
- labels: `positive`, `neutral`, `negative`
- prompt policies: 3
- neutral-bias values:
  - `short_label_prompt`: 0
  - `clinical_cautious_prompt`: 1
  - `balanced_reasoning_prompt`: 0

## Results

| Prompt Policy | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| short_label_prompt | 0.9667 | 0.9666 | 0.9666 |
| balanced_reasoning_prompt | 0.9333 | 0.9332 | 0.9332 |
| clinical_cautious_prompt | 0.3667 | 0.2315 | 0.2315 |

![Prompt evaluation scores](results/prompt_accuracy_macro_f1.png)

The best policy on this small dataset is the direct label prompt. The balanced policy is close behind but misclassifies some mixed or limited-evidence sentences. The cautious policy performs badly because it collapses many positive and negative examples into neutral.

Important result files:

- `results/prompt_metrics.csv`
- `results/prompt_templates.csv`
- `results/all_prompt_predictions.csv`
- `results/error_analysis.csv`
- `results/slice_metrics.csv`
- `results/*_classification_report.csv`
- `results/*_confusion_matrix.csv`
- `results/*_confusion_matrix.png`

## Interpretation

The main lesson is that cautious prompting is not automatically better. A conservative instruction can reduce overconfident mistakes, but it can also destroy recall for positive and negative classes if the model becomes too willing to answer neutral.

The direct prompt wins here because the examples match its evidence vocabulary well. That does not prove it would win on real user data. A stronger conclusion would require more diverse examples, repeated runs, and real LLM outputs.

## Conclusion

This lab demonstrates a clean prompt-evaluation workflow: define a dataset, define prompt policies, run the same examples through each policy, save metrics, inspect errors, and explain the limits of the result. The next research step is to replace the transparent rule policies with outputs from a real LLM and evaluate on a public sentiment benchmark such as SST-2, IMDb, or TweetEval.

## How To Run

```bash
pip install -r requirements.txt
python 1_prompt_evaluation.py
```
