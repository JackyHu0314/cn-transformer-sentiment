import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


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
LABEL2ID = {"负面": 0, "正面": 1}


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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
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
            raw_label = row.get("label")
            if not text:
                raise ValueError(f"Row {row_number}: text is empty.")
            examples.append({"text": text, "label": parse_label(raw_label)})

    if len(examples) < 2:
        raise ValueError("Need at least 2 examples to train.")
    return examples


def stratified_split(examples, val_ratio, seed):
    if val_ratio <= 0:
        return examples, []

    rng = random.Random(seed)
    grouped = defaultdict(list)
    for item in examples:
        grouped[item["label"]].append(item)

    train_examples = []
    val_examples = []
    for label_examples in grouped.values():
        rng.shuffle(label_examples)
        if len(label_examples) == 1:
            train_examples.extend(label_examples)
            continue
        val_size = max(1, round(len(label_examples) * val_ratio))
        val_size = min(val_size, len(label_examples) - 1)
        val_examples.extend(label_examples[:val_size])
        train_examples.extend(label_examples[val_size:])

    rng.shuffle(train_examples)
    rng.shuffle(val_examples)
    return train_examples, val_examples


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    non_blocking = device.type == "cuda"
    return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}


def accuracy_from_logits(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct, total


def format_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def format_gib(num_bytes):
    return f"{num_bytes / 1024 ** 3:.2f} GiB"


def print_device_info(device):
    print(f"Device: {device}")
    if device.type != "cuda":
        return

    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print(f"GPU: {props.name}")
    print(f"GPU capability: {props.major}.{props.minor}")
    print(f"GPU memory: {format_gib(total_bytes - free_bytes)} used / {format_gib(total_bytes)} total")


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, epochs, use_fp16):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        dataloader,
        desc=f"Train {epoch}/{epochs}",
        dynamic_ncols=True,
        leave=False,
    )
    for batch in progress:
        batch = move_batch_to_device(batch, device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            outputs = model(**batch)
            loss = outputs.loss

        if use_fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        correct, count = accuracy_from_logits(outputs.logits.detach(), batch["labels"])
        total_loss += loss.item() * count
        total_correct += correct
        total_count += count
        progress.set_postfix(
            loss=f"{total_loss / max(total_count, 1):.4f}",
            acc=f"{total_correct / max(total_count, 1):.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


@torch.no_grad()
def evaluate(model, dataloader, device, epoch, epochs, use_fp16):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    progress = tqdm(
        dataloader,
        desc=f"Eval  {epoch}/{epochs}",
        dynamic_ncols=True,
        leave=False,
    )
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            outputs = model(**batch)
        correct, count = accuracy_from_logits(outputs.logits, batch["labels"])
        total_loss += outputs.loss.item() * count
        total_correct += correct
        total_count += count
        progress.set_postfix(
            loss=f"{total_loss / max(total_count, 1):.4f}",
            acc=f"{total_correct / max(total_count, 1):.4f}",
        )

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


def save_metadata(output_dir, args, train_count, val_count, best_metrics):
    metadata = {
        "model_name": args.model_name,
        "train_examples": train_count,
        "val_examples": val_count,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "fp16": args.fp16,
        "best_metrics": best_metrics,
        "labels": ID2LABEL,
    }
    with (Path(output_dir) / "training_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a small Chinese Transformer for sentiment analysis.")
    parser.add_argument("--train_file", default="data/sentiment_sample.csv", help="CSV file with text,label columns.")
    parser.add_argument("--val_file", default=None, help="Optional validation CSV file with text,label columns.")
    parser.add_argument("--output_dir", default="outputs/sentiment-roberta-small", help="Directory to save the model.")
    parser.add_argument("--model_name", default="uer/chinese_roberta_L-2_H-128", help="Hugging Face model name or local path.")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Used only when --val_file is not provided.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers. Keep 0 on Windows if multiprocessing is unstable.")
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable pinned CPU memory for CUDA transfers.")
    parser.add_argument("--fp16", action="store_true", help="Use CUDA mixed precision training.")
    parser.add_argument("--local_files_only", action="store_true", help="Only load models from local cache or a local directory.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_train_examples = read_examples(args.train_file)
    if args.val_file:
        train_examples = all_train_examples
        val_examples = read_examples(args.val_file)
    else:
        train_examples, val_examples = stratified_split(all_train_examples, args.val_ratio, args.seed)

    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    pretrained_kwargs = {"local_files_only": args.local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **pretrained_kwargs)
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        **pretrained_kwargs,
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config, **pretrained_kwargs)

    train_dataset = SentimentDataset(train_examples, tokenizer, args.max_length)
    pin_memory = torch.cuda.is_available() and not args.no_pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = None
    if val_examples:
        val_dataset = SentimentDataset(val_examples, tokenizer, args.max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_fp16 = args.fp16 and device.type == "cuda"
    print_device_info(device)
    print(f"Mixed precision fp16: {use_fp16}")
    print(f"DataLoader workers: {args.num_workers}")
    print(f"Pin memory: {pin_memory}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_training_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_accuracy = -1.0
    best_metrics = {}
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, args.epochs, use_fp16)
        elapsed = time.perf_counter() - epoch_start
        peak_memory = None
        if device.type == "cuda":
            peak_memory = format_gib(torch.cuda.max_memory_allocated(device))

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"time={format_duration(elapsed)}"
            + (f" | peak_vram={peak_memory}" if peak_memory else "")
        )

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, epoch, args.epochs, use_fp16)
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )
            current_accuracy = val_metrics["accuracy"]
            current_metrics = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        else:
            current_accuracy = train_metrics["accuracy"]
            current_metrics = {"epoch": epoch, "train": train_metrics, "validation": None}

        if current_accuracy >= best_accuracy:
            best_accuracy = current_accuracy
            best_metrics = current_metrics
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            save_metadata(output_dir, args, len(train_examples), len(val_examples), best_metrics)
            print(f"Saved best model to {output_dir}")

    print("Training finished.")
    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
