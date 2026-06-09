from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RESULTS = Path("results")

DATA = [
    ("The model is fast and accurate, I really like it.", "positive"),
    ("This update solved my problem and saved time.", "positive"),
    ("The interface feels clean, stable, and useful.", "positive"),
    ("The battery life is excellent for edge deployment.", "positive"),
    ("I am happy with the clear documentation.", "positive"),
    ("The agent failed twice and produced wrong answers.", "negative"),
    ("The tool is slow, confusing, and unreliable.", "negative"),
    ("The results are noisy and the setup is painful.", "negative"),
    ("The detector missed important objects in the image.", "negative"),
    ("The latency is unacceptable for real-time use.", "negative"),
    ("The paper is interesting but the evidence is limited.", "neutral"),
    ("The dataset contains 1000 samples and 12 features.", "neutral"),
    ("The method uses logistic regression as a baseline.", "neutral"),
    ("The report describes training and evaluation steps.", "neutral"),
    ("The system runs on CPU and stores logs locally.", "neutral"),
]

PROMPTS = {
    "short_label_prompt": {
        "positive": {"good", "fast", "accurate", "like", "solved", "saved", "clean", "stable", "useful", "excellent", "happy", "clear"},
        "negative": {"failed", "wrong", "slow", "confusing", "unreliable", "noisy", "painful", "missed", "unacceptable"},
    },
    "clinical_cautious_prompt": {
        "positive": {"accurate", "stable", "excellent", "clear"},
        "negative": {"failed", "wrong", "unreliable", "missed", "unacceptable", "noisy"},
    },
    "balanced_reasoning_prompt": {
        "positive": {"good", "fast", "accurate", "like", "solved", "saved", "clean", "stable", "useful", "excellent", "happy", "clear"},
        "negative": {"failed", "wrong", "slow", "confusing", "unreliable", "noisy", "painful", "missed", "unacceptable", "limited"},
    },
}

def classify(text: str, prompt_name: str) -> str:
    words = {w.strip(".,").lower() for w in text.split()}
    prompt = PROMPTS[prompt_name]
    pos = len(words & prompt["positive"])
    neg = len(words & prompt["negative"])
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"

def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(DATA, columns=["text", "label"])
    df.to_csv(RESULTS / "sentiment_examples.csv", index=False)
    rows = []
    all_predictions = []
    for prompt_name in PROMPTS:
        preds = [classify(t, prompt_name) for t in df["text"]]
        acc = accuracy_score(df["label"], preds)
        rows.append({"prompt_template": prompt_name, "accuracy": round(acc, 4)})
        report = classification_report(df["label"], preds, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(RESULTS / f"{prompt_name}_classification_report.csv")
        cm = confusion_matrix(df["label"], preds, labels=["positive", "neutral", "negative"])
        pd.DataFrame(cm, index=["true_positive", "true_neutral", "true_negative"], columns=["pred_positive", "pred_neutral", "pred_negative"]).to_csv(RESULTS / f"{prompt_name}_confusion_matrix.csv")
        for text, true, pred in zip(df["text"], df["label"], preds):
            all_predictions.append({"prompt_template": prompt_name, "text": text, "true_label": true, "predicted_label": pred})
    metrics = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    metrics.to_csv(RESULTS / "prompt_metrics.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(RESULTS / "all_prompt_predictions.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.bar(metrics["prompt_template"], metrics["accuracy"], color=["#3d6fb6", "#4a8f5a", "#b26a3b"])
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Prompt Template Accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "prompt_accuracy.png", dpi=160)
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
