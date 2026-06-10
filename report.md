# Report: Prompt Evaluation for Sentiment Classification

## Motivation

We evaluated prompt styles for sentiment classification because prompts should be compared with evidence, not only with personal preference.

## Dataset

The evaluation set contains 15 short sentences: 5 positive, 5 negative, and 5 neutral.

## Method

Three prompt styles were simulated as transparent keyword policies: short label, cautious, and balanced reasoning. For each policy, we saved predictions, metrics, reports, and confusion matrices.

## Hyperparameters

No model training was used. The labels were positive, neutral, and negative.

## Results

The short label prompt achieved 1.0000 accuracy. The cautious and balanced prompts both achieved 0.9333 accuracy.

## Interpretation

The short prompt worked best because its keyword coverage matched this small evaluation set. This is not enough to claim that it is the best prompt in general. The result shows why prompt evaluation must use diverse examples.

## Conclusion

The project demonstrates a basic evaluation workflow for prompts. A stronger version should use a real LLM and a larger public sentiment dataset.
