import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download


DATASETS = {
    "waimai_10k": {
        "hub_name": "dirtycomputer/waimai_10k",
        "split": "train",
        "direct_file": "waimai_10k.csv",
        "text_columns": ["text", "review", "comment", "content"],
        "label_columns": ["label", "labels"],
    },
    "chnsenticorp": {
        "hub_name": "lansinuote/ChnSentiCorp",
        "split": "train",
        "direct_file": "data/train-00000-of-00001-02f200ca5f2a7868.parquet",
        "text_columns": ["text", "sentence", "review", "content"],
        "label_columns": ["label", "labels"],
    },
}


def choose_column(column_names, candidates, kind):
    for name in candidates:
        if name in column_names:
            return name
    raise ValueError(f"Could not find a {kind} column. Available columns: {', '.join(column_names)}")


def normalize_label(value):
    label = str(value).strip().lower()
    if label in {"1", "positive", "pos", "正面", "积极"}:
        return 1
    if label in {"0", "negative", "neg", "负面", "消极"}:
        return 0
    raise ValueError(f"Unsupported label value: {value}")


def stratified_split(examples, val_ratio, test_ratio, seed):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for item in examples:
        grouped[item["label"]].append(item)

    train_examples = []
    val_examples = []
    test_examples = []
    for label_examples in grouped.values():
        rng.shuffle(label_examples)
        total = len(label_examples)
        test_size = int(total * test_ratio)
        val_size = int(total * val_ratio)

        if total >= 3:
            if test_ratio > 0:
                test_size = max(1, test_size)
            if val_ratio > 0:
                val_size = max(1, val_size)

        test_size = min(test_size, max(total - 2, 0))
        val_size = min(val_size, max(total - test_size - 1, 0))

        test_examples.extend(label_examples[:test_size])
        val_examples.extend(label_examples[test_size : test_size + val_size])
        train_examples.extend(label_examples[test_size + val_size :])

    rng.shuffle(train_examples)
    rng.shuffle(val_examples)
    rng.shuffle(test_examples)
    return train_examples, val_examples, test_examples


def write_csv(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(examples)


def read_source_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))

    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()

    raise ValueError(f"Unsupported source file type: {path.suffix}")


def load_examples(dataset_key, limit):
    spec = DATASETS[dataset_key]
    if spec.get("direct_file"):
        source_path = hf_hub_download(
            repo_id=spec["hub_name"],
            filename=spec["direct_file"],
            repo_type="dataset",
        )
        rows = read_source_rows(source_path)
    else:
        from datasets import load_dataset

        dataset = load_dataset(spec["hub_name"], split=spec["split"])
        rows = list(dataset)

    columns = list(rows[0].keys()) if rows else []
    text_column = choose_column(columns, spec["text_columns"], "text")
    label_column = choose_column(columns, spec["label_columns"], "label")

    examples = []
    seen_texts = set()
    for row in rows:
        text = str(row[text_column]).strip()
        if not text or text in seen_texts:
            continue
        examples.append({"text": text, "label": normalize_label(row[label_column])})
        seen_texts.add(text)
        if limit and len(examples) >= limit:
            break

    return examples, spec["hub_name"], text_column, label_column


def main():
    parser = argparse.ArgumentParser(description="Prepare public Chinese sentiment datasets as text,label CSV files.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="waimai_10k")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of examples. 0 means no limit.")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples, hub_name, text_column, label_column = load_examples(args.dataset, args.limit)
    if len(examples) < 10:
        raise ValueError("Too few examples after loading and cleaning.")

    train_examples, val_examples, test_examples = stratified_split(
        examples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    prefix = args.dataset
    write_csv(output_dir / f"{prefix}_train.csv", train_examples)
    write_csv(output_dir / f"{prefix}_val.csv", val_examples)
    write_csv(output_dir / f"{prefix}_test.csv", test_examples)

    print(f"Source dataset: {hub_name}")
    print(f"Text column: {text_column}")
    print(f"Label column: {label_column}")
    print(f"Total examples: {len(examples)}")
    print(f"Label counts: {dict(Counter(item['label'] for item in examples))}")
    print(f"Train: {len(train_examples)} -> {output_dir / f'{prefix}_train.csv'}")
    print(f"Validation: {len(val_examples)} -> {output_dir / f'{prefix}_val.csv'}")
    print(f"Test: {len(test_examples)} -> {output_dir / f'{prefix}_test.csv'}")


if __name__ == "__main__":
    main()
