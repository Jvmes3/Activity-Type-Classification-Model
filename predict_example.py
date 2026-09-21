"""Reusable classifier and command-line prediction."""
import argparse
import json
import os
from pathlib import Path
from activity_data import LABELS, activity_text


HF_REPO_ID = "jvmes3/ActivityTypeClassifierModel"


class ActivityClassifier:
    def __init__(self, model_path=None):
        model_path = model_path or os.getenv("MODEL_PATH") or (
            "activity_model_outputs" if Path("activity_model_outputs/config.json").is_file()
            else HF_REPO_ID)
        local = Path(model_path).is_dir()
        if not local and model_path != HF_REPO_ID:
            raise ValueError(f"Trained model directory not found: {model_path}. Run training first.")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=local)
        if self.model.config.id2label != dict(enumerate(LABELS)):
            raise ValueError("Model label mapping does not match the seven activity types")
        self.model.eval()

    def predict(self, activity):
        text = activity_text(activity)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with self.torch.inference_mode():
            probabilities = self.model(**inputs).logits.softmax(dim=-1)[0].tolist()
        index = max(range(len(LABELS)), key=probabilities.__getitem__)
        return {"activityType": LABELS[index], "confidence": probabilities[index],
                "scores": dict(zip(LABELS, probabilities))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Local model directory; defaults to local weights or the Hugging Face repo")
    parser.add_argument("--title", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--instructions", default="")
    args = parser.parse_args()
    activity = {key: getattr(args, key) for key in ("title", "description", "instructions")}
    activity_text(activity)
    print(json.dumps(ActivityClassifier(args.model).predict(activity), indent=2))


if __name__ == "__main__":
    main()
