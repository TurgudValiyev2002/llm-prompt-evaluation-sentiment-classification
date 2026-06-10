from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

RESULTS = Path("results")
LABELS = ["positive", "neutral", "negative"]

DATA = [
    ("The model is fast and accurate, I really like it.", "positive", "plain_positive"),
    ("This update solved my problem and saved time.", "positive", "plain_positive"),
    ("The interface feels clean, stable, and useful.", "positive", "plain_positive"),
    ("The battery life is excellent for edge deployment.", "positive", "plain_positive"),
    ("I am happy with the clear documentation.", "positive", "plain_positive"),
    ("The new dashboard is reliable and easy to understand.", "positive", "plain_positive"),
    ("The agent gave correct answers after the prompt was shortened.", "positive", "llm_context"),
    ("The detector improves recall without increasing latency.", "positive", "ai_system"),
    ("The deployment is simple and the logs are helpful.", "positive", "engineering"),
    ("The baseline is modest, but the improvement is meaningful.", "positive", "mixed_clause"),
    ("The agent failed twice and produced wrong answers.", "negative", "plain_negative"),
    ("The tool is slow, confusing, and unreliable.", "negative", "plain_negative"),
    ("The results are noisy and the setup is painful.", "negative", "plain_negative"),
    ("The detector missed important objects in the image.", "negative", "ai_system"),
    ("The latency is unacceptable for real-time use.", "negative", "edge_context"),
    ("The output looks polished, but the factual errors are serious.", "negative", "mixed_clause"),
    ("The model is accurate on easy cases but unsafe on hard cases.", "negative", "mixed_clause"),
    ("The prompt sounds confident while giving unsupported claims.", "negative", "llm_context"),
    ("The benchmark score dropped after quantization.", "negative", "ai_system"),
    ("The system crashes when the input is longer than expected.", "negative", "engineering"),
    ("The paper is interesting but the evidence is limited.", "neutral", "academic"),
    ("The dataset contains 1000 samples and 12 features.", "neutral", "metadata"),
    ("The method uses logistic regression as a baseline.", "neutral", "method_description"),
    ("The report describes training and evaluation steps.", "neutral", "method_description"),
    ("The system runs on CPU and stores logs locally.", "neutral", "engineering"),
    ("The experiment compares three prompt templates.", "neutral", "llm_context"),
    ("The model was evaluated on validation and test splits.", "neutral", "method_description"),
    ("The paper reports accuracy, F1 score, and confusion matrices.", "neutral", "academic"),
    ("The pipeline stores predictions in a CSV file.", "neutral", "engineering"),
    ("The result is not bad, but it is not clearly good either.", "neutral", "ambiguous"),
]

PROMPT_POLICIES = {
    "short_label_prompt": {
        "template": "Classify the sentence as positive, neutral, or negative. Return only the label.",
        "positive": {"fast", "accurate", "like", "solved", "saved", "clean", "stable", "useful", "excellent", "happy", "clear", "reliable", "easy", "correct", "improves", "helpful", "meaningful"},
        "negative": {"failed", "wrong", "slow", "confusing", "unreliable", "noisy", "painful", "missed", "unacceptable", "errors", "unsafe", "unsupported", "dropped", "crashes"},
        "neutral_bias": 0,
    },
    "clinical_cautious_prompt": {
        "template": "Classify only when the evidence is strong. If sentiment is mixed or weak, return neutral.",
        "positive": {"accurate", "stable", "excellent", "reliable", "correct", "improves"},
        "negative": {"failed", "wrong", "unreliable", "missed", "unacceptable", "unsafe", "unsupported", "crashes"},
        "neutral_bias": 1,
    },
    "balanced_reasoning_prompt": {
        "template": "Decide the overall sentiment after weighing positive and negative evidence in the sentence.",
        "positive": {"fast", "accurate", "like", "solved", "saved", "clean", "stable", "useful", "excellent", "happy", "clear", "reliable", "easy", "correct", "improves", "helpful", "meaningful", "polished"},
        "negative": {"failed", "wrong", "slow", "confusing", "unreliable", "noisy", "painful", "missed", "unacceptable", "limited", "errors", "serious", "unsafe", "unsupported", "dropped", "crashes"},
        "neutral_bias": 0,
    },
}


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def classify(text: str, prompt_name: str) -> tuple[str, str]:
    words = tokenize(text)
    policy = PROMPT_POLICIES[prompt_name]
    pos_hits = sorted(words & policy["positive"])
    neg_hits = sorted(words & policy["negative"])
    pos = len(pos_hits)
    neg = len(neg_hits)
    if abs(pos - neg) <= policy["neutral_bias"]:
        return "neutral", f"neutral tie/bias; positive={pos_hits}; negative={neg_hits}"
    if pos > neg:
        return "positive", f"more positive evidence; positive={pos_hits}; negative={neg_hits}"
    return "negative", f"more negative evidence; positive={pos_hits}; negative={neg_hits}"


def save_confusion_matrix_plot(matrix: pd.DataFrame, prompt_name: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(matrix.values, cmap="Blues")
    ax.set_title(f"Confusion Matrix: {prompt_name}")
    ax.set_xticks(range(len(LABELS)), labels=[f"pred {x}" for x in LABELS], rotation=25, ha="right")
    ax.set_yticks(range(len(LABELS)), labels=[f"true {x}" for x in LABELS])
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, matrix.iloc[row_idx, col_idx], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{prompt_name}_confusion_matrix.png", dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(DATA, columns=["text", "label", "slice"])
    df.to_csv(RESULTS / "sentiment_examples.csv", index=False)

    prompt_rows = [
        {"prompt_template": name, "template_text": config["template"]}
        for name, config in PROMPT_POLICIES.items()
    ]
    pd.DataFrame(prompt_rows).to_csv(RESULTS / "prompt_templates.csv", index=False)

    metric_rows = []
    slice_rows = []
    prediction_rows = []

    for prompt_name in PROMPT_POLICIES:
        predictions = []
        rationales = []
        for text in df["text"]:
            pred, rationale = classify(text, prompt_name)
            predictions.append(pred)
            rationales.append(rationale)

        accuracy = accuracy_score(df["label"], predictions)
        macro_f1 = f1_score(df["label"], predictions, labels=LABELS, average="macro")
        weighted_f1 = f1_score(df["label"], predictions, labels=LABELS, average="weighted")
        metric_rows.append({
            "prompt_template": prompt_name,
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "num_examples": len(df),
        })

        report = classification_report(df["label"], predictions, labels=LABELS, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(RESULTS / f"{prompt_name}_classification_report.csv")

        cm = confusion_matrix(df["label"], predictions, labels=LABELS)
        cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS])
        cm_df.to_csv(RESULTS / f"{prompt_name}_confusion_matrix.csv")
        save_confusion_matrix_plot(cm_df, prompt_name)

        for row, pred, rationale in zip(df.itertuples(index=False), predictions, rationales):
            correct = row.label == pred
            prediction_rows.append({
                "prompt_template": prompt_name,
                "text": row.text,
                "slice": row.slice,
                "true_label": row.label,
                "predicted_label": pred,
                "correct": correct,
                "rationale": rationale,
            })

        scored = df.copy()
        scored["prediction"] = predictions
        scored["correct"] = scored["label"] == scored["prediction"]
        for slice_name, slice_df in scored.groupby("slice"):
            slice_rows.append({
                "prompt_template": prompt_name,
                "slice": slice_name,
                "num_examples": len(slice_df),
                "accuracy": round(slice_df["correct"].mean(), 4),
            })

    metrics = pd.DataFrame(metric_rows).sort_values(["accuracy", "macro_f1"], ascending=False)
    metrics.to_csv(RESULTS / "prompt_metrics.csv", index=False)

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(RESULTS / "all_prompt_predictions.csv", index=False)
    predictions_df[predictions_df["correct"] == False].to_csv(RESULTS / "error_analysis.csv", index=False)
    pd.DataFrame(slice_rows).sort_values(["slice", "prompt_template"]).to_csv(RESULTS / "slice_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(metrics))
    ax.bar(x, metrics["accuracy"], label="Accuracy", color="#3d6fb6")
    ax.plot(x, metrics["macro_f1"], marker="o", label="Macro F1", color="#b35d2e")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Prompt Policy Evaluation")
    ax.set_xticks(x, labels=metrics["prompt_template"], rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "prompt_accuracy_macro_f1.png", dpi=160)
    plt.close(fig)

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
