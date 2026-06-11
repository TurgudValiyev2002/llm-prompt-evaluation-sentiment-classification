# One-Page Report: Prompt Evaluation for Sentiment Classification

## Motivation

The first version used only 30 examples, which made slice-level results fragile. We expanded the benchmark to make prompt comparison more meaningful.

## Dataset

The dataset now has 247 labeled examples across clear positive, clear negative, neutral method/reporting, and mixed-risk slices. Labels are positive, neutral, and negative.

## Method

We evaluated three deterministic prompt-policy simulators: a short label prompt, a cautious neutral-biased prompt, and a balanced reasoning prompt. We measured accuracy, macro F1, weighted F1, confusion matrices, slice metrics, and errors.

## Results

The balanced reasoning prompt achieved 0.9271 accuracy and 0.9168 macro F1. The short label prompt achieved 0.8543 accuracy and 0.8389 macro F1. The cautious prompt achieved only 0.2227 accuracy and 0.1727 macro F1.

## Interpretation

The cautious prompt failed because it predicted neutral too often. This shows why a prompt that sounds careful can still perform poorly if the task requires clear positive or negative labels. The balanced prompt handled mixed evidence better.

## Conclusion

The project is now a stronger prompt-evaluation demo. The next improvement should use a real LLM and a public sentiment dataset.
