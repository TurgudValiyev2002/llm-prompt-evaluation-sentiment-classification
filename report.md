# One-Page Report: Prompt Evaluation for Sentiment Classification

## Motivation

This lab shows how to evaluate prompt behavior instead of trusting intuition. The research lesson is simple: a prompt is a method, and methods need measurement.

## Tools

Python, pandas, scikit-learn metrics, and matplotlib.

## Dataset Or Problem

The dataset is a small offline evaluation set with 15 labeled sentences across positive, neutral, and negative sentiment.

## Method

Three prompt templates are simulated as transparent keyword classifiers: short label, cautious, and balanced reasoning.

## Hyperparameters

The label set is fixed to three classes. No training or tuning is performed.

## Results

The script saves prompt-level accuracy, prediction tables, classification reports, and an accuracy bar chart in `results/`.

## Interpretation

The short label template performs best on this tiny set because its keyword coverage matches the examples. That does not prove it is generally better; it shows why prompt evaluation needs larger and more diverse test data.

## Conclusion

Prompt engineering should be treated as experimental design: define labels, run comparisons, and inspect errors.
