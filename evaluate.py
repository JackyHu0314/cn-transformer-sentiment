import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_ALIASES = {
    "0": 0,
    "negative": 0,
    "neg": 0,
    "负面": 0,
    "消极": 0,
    "1": 1,
    "positive": 1,
    "pos": 1,
    "正面": 1,
    "积极": 1,
}

ID2LABEL = {0: "负面", 1: "正面"}


class SentimentDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        texts = [item["text"] for item in examples]
        labels = [item["label"] for item in examples]
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.texts = texts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        item["index"] = torch.tensor(idx, dtype=torch.long)
        return item


def parse_label(raw_label):
    label = str(raw_label).strip().lower()
    if label not in LABEL_ALIASES:
        allowed = ", ".join(sorted(LABEL_ALIASES))
        raise ValueError(f"Unsupported label '{raw_label}'. Allowed labels: {allowed}")
    return LABEL_ALIASES[label]


def read_examples(path):
    examples = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain 'text' and 'label' columns.")

        for row_number, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            if not text:
                raise ValueError(f"Row {row_number}: text is empty.")
            examples.append({"text": text, "label": parse_label(row.get("label"))})
    return examples


def compute_metrics(y_true, y_pred, num_labels=2):
    confusion = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for true_label, pred_label in zip(y_true, y_pred):
        confusion[true_label][pred_label] += 1

    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(num_labels))
    accuracy = correct / max(total, 1)

    per_label = {}
    precision_values = []
    recall_values = []
    f1_values = []
    for label_id in range(num_labels):
        tp = confusion[label_id][label_id]
        fp = sum(confusion[row][label_id] for row in range(num_labels) if row != label_id)
        fn = sum(confusion[label_id][col] for col in range(num_labels) if col != label_id)
        support = sum(confusion[label_id])

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_label[label_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "accuracy": accuracy,
        "macro_precision": sum(precision_values) / num_labels,
        "macro_recall": sum(recall_values) / num_labels,
        "macro_f1": sum(f1_values) / num_labels,
        "confusion": confusion,
        "per_label": per_label,
    }


def print_metrics(metrics):
    print(f"accuracy:        {metrics['accuracy']:.4f}")
    print(f"macro_precision: {metrics['macro_precision']:.4f}")
    print(f"macro_recall:    {metrics['macro_recall']:.4f}")
    print(f"macro_f1:        {metrics['macro_f1']:.4f}")
    print()
    print("per-label metrics:")
    print("label  precision  recall  f1      support")
    for label_id, values in metrics["per_label"].items():
        print(
            f"{ID2LABEL[label_id]:<4}  "
            f"{values['precision']:.4f}     "
            f"{values['recall']:.4f}  "
            f"{values['f1']:.4f}  "
            f"{values['support']}"
        )

    print()
    print("confusion matrix:")
    print("rows=true, cols=pred")
    print("          pred_负面  pred_正面")
    print(f"true_负面  {metrics['confusion'][0][0]:>8}  {metrics['confusion'][0][1]:>8}")
    print(f"true_正面  {metrics['confusion'][1][0]:>8}  {metrics['confusion'][1][1]:>8}")


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    mistakes = []

    for batch in dataloader:
        labels = batch.pop("labels")
        indices = batch.pop("index")
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        probabilities = torch.softmax(logits, dim=-1).cpu()
        predictions = torch.argmax(probabilities, dim=-1)

        y_true.extend(labels.tolist())
        y_pred.extend(predictions.tolist())

        for index, true_label, pred_label, probs in zip(indices.tolist(), labels.tolist(), predictions.tolist(), probabilities):
            if true_label != pred_label:
                mistakes.append(
                    {
                        "index": index,
                        "true_label": ID2LABEL[true_label],
                        "pred_label": ID2LABEL[pred_label],
                        "confidence": float(probs[pred_label].item()),
                    }
                )

    return compute_metrics(y_true, y_pred), mistakes


def enrich_mistakes(mistakes, examples):
    enriched = []
    for item in mistakes:
        example = examples[item["index"]]
        enriched.append(
            {
                **item,
                "text": example["text"],
            }
        )
    return enriched


def write_mistakes(path, mistakes):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "true_label", "pred_label", "confidence", "text"])
        writer.writeheader()
        writer.writerows(mistakes)


def print_mistakes(mistakes, limit):
    if limit <= 0:
        return

    print()
    print(f"mistakes preview: {min(limit, len(mistakes))}/{len(mistakes)}")
    for item in mistakes[:limit]:
        print(
            f"[{item['index']}] true={item['true_label']} "
            f"pred={item['pred_label']} conf={item['confidence']:.4f} | {item['text']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned sentiment model on a CSV dataset.")
    parser.add_argument("--model_dir", default="outputs/sentiment-roberta-medium")
    parser.add_argument("--test_file", default="data/waimai_10k_test.csv")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mistakes_file", default="outputs/sentiment-roberta-medium/mistakes.csv")
    parser.add_argument("--show_mistakes", type=int, default=20)
    args = parser.parse_args()

    examples = read_examples(args.test_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    dataset = SentimentDataset(examples, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Examples: {len(examples)}")
    print(f"Device: {device}")

    metrics, mistakes = evaluate(model, dataloader, device)
    mistakes = enrich_mistakes(mistakes, examples)
    print_metrics(metrics)
    write_mistakes(args.mistakes_file, mistakes)
    if args.mistakes_file:
        print(f"\nSaved mistakes to {args.mistakes_file}")
    print_mistakes(mistakes, args.show_mistakes)


if __name__ == "__main__":
    main()
