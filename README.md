# LLM Prompt Evaluation for Sentiment Classification

## 1. Motivation

Prompt quality changes the behavior of an LLM. Dear Turgud, the critical point is that we should not say "this prompt is better" by feeling. We need a small evaluation set, clear labels, and measurable results.

## 2. Project Goal

The goal is to compare three prompt-style decision rules for sentiment classification: a short label prompt, a cautious prompt, and a balanced reasoning prompt.

## 3. Dataset Or Problem Description

This is an offline prompt-evaluation simulation with 15 manually written sentences: 5 positive, 5 negative, and 5 neutral. It is not a substitute for a real LLM benchmark, but it teaches the evaluation logic.

## 4. Tools

Python, pandas, scikit-learn metrics, and matplotlib.

## 5. Models Or Methods

Each prompt template is represented by a transparent keyword policy. This lets us test prompt behavior without internet access or paid API calls.

## 6. Hyperparameters

No model training is used. The fixed settings are three prompt templates and the label set: positive, neutral, negative.

## 7. Results

The best template on this tiny evaluation set was the short label prompt. Results are saved in `results/prompt_metrics.csv`, `results/all_prompt_predictions.csv`, and `results/prompt_accuracy.png`.

## 8. Interpretation Of Results

The cautious and balanced prompts avoid some overconfident decisions, but in this small synthetic set the short label prompt performs best because its keyword coverage exactly matches the examples. This is a warning: a tiny benchmark can over-reward a simple rule.

## 9. Conclusion

Prompt evaluation must be empirical. Even a simple test shows that different instructions can change classification behavior.

## 10. How To Run

```bash
pip install -r requirements.txt
python 1_prompt_evaluation.py
```
