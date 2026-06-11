from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


RESULTS = Path("results")
LABELS = ["positive", "neutral", "negative"]

POSITIVE_TEMPLATES = [
    "The {system} is fast, stable, and useful for {context}.",
    "The {system} improved {metric} without increasing {cost}.",
    "The {system} gave correct answers and clear explanations for {context}.",
    "The {system} solved the problem and saved time during {context}.",
    "The {system} is reliable, easy to use, and well documented.",
]
NEGATIVE_TEMPLATES = [
    "The {system} failed during {context} and produced wrong answers.",
    "The {system} is slow, confusing, and unreliable for {context}.",
    "The {system} missed important cases and reduced {metric}.",
    "The {system} looks polished but the factual errors are serious.",
    "The {system} crashed when {context} became more complex.",
]
NEUTRAL_TEMPLATES = [
    "The {system} was evaluated on validation and test splits.",
    "The {system} uses {method} as the main method.",
    "The report describes {metric}, data preprocessing, and evaluation.",
    "The experiment stores predictions and metrics in CSV files.",
    "The paper compares {method} with a baseline model.",
]
MIXED_TEMPLATES = [
    "The {system} is accurate on easy cases but unsafe on hard cases.",
    "The {system} improves {metric}, but the evidence is limited.",
    "The {system} is fast, but the errors are serious for {context}.",
    "The {system} is useful in simple demos but unreliable in deployment.",
    "The {system} gives clear explanations, but some claims are unsupported.",
]

SYSTEMS = ["model", "agent", "detector", "retriever", "pipeline", "dashboard"]
CONTEXTS = ["edge deployment", "medical triage", "scientific retrieval", "object detection", "prompt evaluation", "graph analysis"]
METRICS = ["accuracy", "recall", "latency", "macro F1", "calibration", "retrieval precision"]
METHODS = ["logistic regression", "federated averaging", "TF-IDF retrieval", "quantization", "graph smoothing", "early stopping"]

PROMPT_POLICIES = {
    "short_label_prompt": {
        "template": "Classify the sentence as positive, neutral, or negative. Return only the label.",
        "positive": {"fast", "stable", "useful", "improved", "correct", "clear", "solved", "saved", "reliable", "easy", "well", "documented", "accurate"},
        "negative": {"failed", "wrong", "slow", "confusing", "unreliable", "missed", "reduced", "errors", "serious", "crashed", "unsafe", "unsupported"},
        "neutral_bias": 0,
    },
    "clinical_cautious_prompt": {
        "template": "Classify only when the evidence is strong. If sentiment is mixed or weak, return neutral.",
        "positive": {"stable", "improved", "correct", "solved", "reliable"},
        "negative": {"failed", "wrong", "unreliable", "missed", "errors", "crashed", "unsafe"},
        "neutral_bias": 1,
    },
    "balanced_reasoning_prompt": {
        "template": "Decide the overall sentiment after weighing positive and negative evidence in the sentence.",
        "positive": {"fast", "stable", "useful", "improved", "correct", "clear", "solved", "saved", "reliable", "easy", "documented", "accurate", "polished"},
        "negative": {"failed", "wrong", "slow", "confusing", "unreliable", "missed", "reduced", "limited", "errors", "serious", "crashed", "unsafe", "unsupported"},
        "neutral_bias": 0,
    },
}


def build_dataset() -> pd.DataFrame:
    rows = []
    specs = [
        ("positive", "positive_clear", POSITIVE_TEMPLATES),
        ("negative", "negative_clear", NEGATIVE_TEMPLATES),
        ("neutral", "neutral_method", NEUTRAL_TEMPLATES),
        ("negative", "mixed_risk", MIXED_TEMPLATES),
    ]
    for label, slice_name, templates in specs:
        for template in templates:
            for system in SYSTEMS:
                for context in CONTEXTS[:3]:
                    text = template.format(
                        system=system,
                        context=context,
                        metric=METRICS[(len(rows) + 2) % len(METRICS)],
                        method=METHODS[(len(rows) + 1) % len(METHODS)],
                        cost="latency",
                    )
                    rows.append({"text": text, "label": label, "slice": slice_name})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


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
    fig.savefig(RESULTS / f"{prompt_name}_confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = build_dataset()
    df.to_csv(RESULTS / "sentiment_examples.csv", index=False)

    pd.DataFrame(
        [{"prompt_template": name, "template_text": config["template"]} for name, config in PROMPT_POLICIES.items()]
    ).to_csv(RESULTS / "prompt_templates.csv", index=False)

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

        metric_rows.append(
            {
                "prompt_template": prompt_name,
                "accuracy": round(accuracy_score(df["label"], predictions), 4),
                "macro_f1": round(f1_score(df["label"], predictions, labels=LABELS, average="macro"), 4),
                "weighted_f1": round(f1_score(df["label"], predictions, labels=LABELS, average="weighted"), 4),
                "num_examples": len(df),
            }
        )

        report = classification_report(df["label"], predictions, labels=LABELS, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(RESULTS / f"{prompt_name}_classification_report.csv")
        cm = confusion_matrix(df["label"], predictions, labels=LABELS)
        cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS])
        cm_df.to_csv(RESULTS / f"{prompt_name}_confusion_matrix.csv")
        save_confusion_matrix_plot(cm_df, prompt_name)

        for row, pred, rationale in zip(df.itertuples(index=False), predictions, rationales):
            prediction_rows.append(
                {
                    "prompt_template": prompt_name,
                    "text": row.text,
                    "slice": row.slice,
                    "true_label": row.label,
                    "predicted_label": pred,
                    "correct": row.label == pred,
                    "rationale": rationale,
                }
            )

        scored = df.copy()
        scored["prediction"] = predictions
        scored["correct"] = scored["label"] == scored["prediction"]
        for slice_name, slice_df in scored.groupby("slice"):
            slice_rows.append(
                {
                    "prompt_template": prompt_name,
                    "slice": slice_name,
                    "num_examples": len(slice_df),
                    "accuracy": round(slice_df["correct"].mean(), 4),
                }
            )

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
    ax.set_title("Prompt Policy Evaluation on Expanded Dataset")
    ax.set_xticks(x, labels=metrics["prompt_template"], rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "prompt_accuracy_macro_f1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = [
        (f"{len(df)} examples", 0.16),
        ("3 prompt\npolicies", 0.40),
        ("Slice + error\nanalysis", 0.64),
        ("Accuracy and\nmacro F1", 0.86),
    ]
    for text, xpos in boxes:
        ax.text(xpos, 0.55, text, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef6ff", edgecolor="#336699"))
    for start, end in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(end[1] - 0.11, 0.55), xytext=(start[1] + 0.11, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Expanded prompt-evaluation workflow", fontsize=15)
    fig.tight_layout()
    fig.savefig("assets/readme_project_overview.png", dpi=180)
    plt.close(fig)

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
