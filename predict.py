import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_label_name(config, label_id):
    id2label = getattr(config, "id2label", {}) or {}
    return id2label.get(label_id) or id2label.get(str(label_id)) or str(label_id)


@torch.no_grad()
def predict(texts, model_dir, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)

    results = []
    for text, probs in zip(texts, probabilities):
        label_id = int(torch.argmax(probs).item())
        label_name = get_label_name(model.config, label_id)
        confidence = float(probs[label_id].item())
        results.append({"text": text, "label": label_name, "confidence": confidence})
    return results


def read_interactive_texts():
    print("Enter text, one sentence per line. Submit an empty line to finish.")
    texts = []
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        texts.append(line)
    return texts


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment with a fine-tuned Transformer model.")
    parser.add_argument("--model_dir", default="outputs/sentiment-roberta-small")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--text", nargs="*", help="Text to classify. Pass multiple strings to batch predict.")
    args = parser.parse_args()

    if args.text:
        texts = args.text
    else:
        texts = read_interactive_texts()

    if not texts:
        raise SystemExit("No text provided.")

    for item in predict(texts, args.model_dir, args.max_length):
        print(f"{item['label']}\t{item['confidence']:.4f}\t{item['text']}")


if __name__ == "__main__":
    main()
