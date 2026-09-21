"""Fine-tune DistilBERT with class weighting and held-out evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
from activity_data import LABELS, LABEL2ID, load_records, split_records, class_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="activity_model_outputs")
    parser.add_argument("--model", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    splits = split_records(load_records(args.data), args.seed)
    counts = {name: class_counts(rows) for name, rows in splits.items()}
    print(json.dumps(counts, indent=2))
    if args.validate_only:
        return
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    import numpy as np
    import torch
    from datasets import Dataset, DatasetDict
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = DatasetDict({name: Dataset.from_list(rows) for name, rows in splits.items()})
    dataset = dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=512),
                          batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID, problem_type="single_label_classification")
    weights = torch.tensor([len(splits["train"]) / (len(LABELS) * counts["train"][label])
                            for label in LABELS], dtype=torch.float)

    class BalancedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels,
                                                     weight=weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    def metrics(prediction):
        logits, labels = prediction
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions),
                "macro_f1": f1_score(labels, predictions, labels=list(range(7)),
                                     average="macro", zero_division=0)}

    output = Path(args.output)
    trainer = BalancedTrainer(
        model=model, processing_class=tokenizer,
        args=TrainingArguments(output_dir=str(output / "checkpoints"),
            learning_rate=2e-5, per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size, num_train_epochs=args.epochs,
            weight_decay=.01, eval_strategy="epoch", save_strategy="epoch",
            save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="macro_f1",
            greater_is_better=True, report_to="none", seed=args.seed),
        train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    prediction = trainer.predict(dataset["test"])
    predicted = prediction.predictions.argmax(axis=-1)
    report = {"metrics": prediction.metrics, "labels": LABELS,
              "per_class": classification_report(prediction.label_ids, predicted,
                  labels=list(range(7)), target_names=LABELS, output_dict=True, zero_division=0),
              "confusion_matrix": confusion_matrix(prediction.label_ids, predicted,
                  labels=list(range(7))).tolist(),
              "split_counts": counts, "seed": args.seed, "base_model": args.model,
              "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest()}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
