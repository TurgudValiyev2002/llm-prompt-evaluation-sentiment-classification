# One-Page Report: Prompt Evaluation for Sentiment Classification

## Motivation

A small prompt test can look convincing while hiding failures on mixed or neutral examples. This benchmark uses a larger labeled set so prompt comparison is more meaningful.

## Dataset

The dataset has 247 labeled examples across clear positive, clear negative, neutral method/reporting, and mixed-risk slices. Labels are positive, neutral, and negative.

## Method

We evaluated three deterministic prompt-policy simulators: a short label prompt, a cautious neutral-biased prompt, and a balanced reasoning prompt. We measured accuracy, macro F1, weighted F1, confusion matrices, slice metrics, and errors.

## Results

The balanced reasoning prompt achieved 0.9271 accuracy and 0.9168 macro F1. The short label prompt achieved 0.8543 accuracy and 0.8389 macro F1. The cautious prompt achieved only 0.2227 accuracy and 0.1727 macro F1.

## Interpretation

The cautious prompt failed because it predicted neutral too often. This shows why a prompt that sounds careful can still perform poorly if the task requires clear positive or negative labels. The balanced prompt handled mixed evidence better.

## Conclusion

The project shows how prompt policies can be compared with repeatable metrics and error analysis. The next improvement should use a real LLM and a public sentiment dataset.
