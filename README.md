# MyVillage Activity Type Classification Model — A2

Phase 1 classifier that takes an activity's title, description, and instructions and
returns exactly one `activityType`:

`REFLECTION`, `RESEARCH`, `COLLABORATE`, `CREATE`, `PRACTICE`, `EXPERIENCE`, or `TEACH`.

Adapted from the workflow in [your theme classifier](https://github.com/Jvmes3/theme_modelAPI):
train, evaluate, save, predict. This model uses DistilBERT, integer class targets,
cross-entropy loss, and softmax predictions. The theme classifier's multi-label
sigmoid thresholds do not apply here.

**Status:** implementation and synthetic pipeline fixtures. No production-trained
weights or real-data accuracy claims are included. Approved MyVillage records were
not available during implementation. Training downloads the base model on first use.

## Setup

Python 3.11 recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data and training

Export labeled activities through approved MCP `activity_list` access or the AI-team
datasets, then normalize them to the schema below. This repository reads local exports;
it does not assume a private MCP endpoint, credential format, or response envelope.

```json
{"title":"Learning journal","description":"Review your learning.","instructions":"Explain what changed in your understanding.","activityType":"REFLECTION"}
```

Use JSONL or a JSON array. Text fields may be omitted, but at least one must be
nonempty. Labels must match exactly. See [data guidance](data/README.md).

```bash
# Validate the included synthetic fixture without installing ML dependencies:
python train_activity_model.py --data data/examples.jsonl --validate-only

# Train using your approved, reviewed data:
python train_activity_model.py --data data/activities.jsonl --epochs 3

# Optional pipeline demonstration only; scores are not real-world evidence:
python train_activity_model.py --data data/examples.jsonl --epochs 1
```

The loader rejects unknown labels and conflicting duplicates, removes exact duplicates,
and splits each class approximately 70/15/15 with a fixed seed. Inverse-frequency
training weights address class imbalance. Validation macro F1 selects the best checkpoint;
the separate test partition produces accuracy, macro F1, per-class precision/recall/F1,
and a confusion matrix. `activity_model_outputs/evaluation.json` also records class counts,
the seed, and a source-data hash. Checkpoints and private datasets are excluded from Git.
Near duplicates and related activities require manual grouping before final evaluation.

Inputs are truncated to 512 tokens; important instructions should fit within that limit.

## Prediction

```bash
python predict_example.py --title "Learning journal" \
  --description "Review your learning" \
  --instructions "Explain what you learned and what you would change"
```

Python integration:

```python
from predict_example import ActivityClassifier
classifier = ActivityClassifier("activity_model_outputs")
result = classifier.predict({"title": "Learning journal", "instructions": "Reflect on your progress"})
print(result["activityType"])
```

Results contain `activityType`, `confidence`, and `scores` for all seven classes.
Confidence is an uncalibrated softmax score. Validate any review threshold on real data.
Inference loads trained local weights and checks the label mapping.

## HTTP API

After training:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"title":"Learning journal","instructions":"Reflect on your progress"}'
```

`POST /predict` accepts the three text fields. `GET /health` reports readiness after
model loading; `/docs` provides the interactive API schema. Set `MODEL_PATH` for a
custom local model directory. Startup fails clearly when trained weights are missing.
The API supports A1 Activity Generation routing and downstream analytics. Authentication
and deployment should be supplied by the host application before exposing it externally.

## Labeling guide

| Type | Primary objective |
| --- | --- |
| REFLECTION | Examine one's learning, decisions, or experiences |
| RESEARCH | Gather, compare, and assess evidence |
| COLLABORATE | Work with others toward a shared decision or outcome |
| CREATE | Produce an original artifact or solution |
| PRACTICE | Improve a skill through repeated performance |
| EXPERIENCE | Learn through firsthand exposure or participation |
| TEACH | Help someone else learn through explanation or guidance |

These are working annotation guidelines derived from the roadmap; confirm them with
the AI team. For mixed activities, label the primary learning objective.

## Checks

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs data-contract tests and fixture validation without downloading a model.

References: [MyVillage MCP reference](https://www.myvillageproject.ai/developers/mcp-reference)
and [Hugging Face sequence classification guide](https://huggingface.co/docs/transformers/tasks/sequence_classification).

For an end-to-end offline smoke test after installing dependencies:

```bash
pip install httpx
HF_HUB_OFFLINE=1 python tests/smoke_pipeline.py
```

This builds a tiny random DistilBERT in a temporary directory, runs one training
epoch, reloads the checkpoint, and checks evaluation, prediction, and API validation.
It verifies the pipeline, not model quality.
