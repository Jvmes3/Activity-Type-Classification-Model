# V3 synthetic training experiment — 2026-09-26

Dataset: 350 AI-generated activities; 252 training, 49 validation, 49 test.
Dataset commit: `087c6d4`. No real MyVillage records were used.

Command:
```bash
python train_activity_model.py --data data/activities.jsonl --epochs 5 --seed 42 --output activity_model_outputs/v3
```

Started from pretrained DistilBERT. This run used CPU; V2 used Apple MPS.
The best validation checkpoint was step 96 (epoch 3), with validation macro F1 1.0.
The five-epoch schedule differs from the earlier three-epoch schedule, including
its learning-rate decay. This is not a controlled comparison of epoch count alone.

| Metric | V2 | V3 |
| --- | --- | --- |
| Synthetic test accuracy | 43/49 (87.8%) | 47/49 (95.9%) |
| Synthetic test macro F1 | 0.8683 | 0.9571 |

Data hash and split seed match V2. Repeated inspection of this test set makes it
an exploratory benchmark; a fresh independent evaluation set is still needed.
These results do not establish real-world accuracy. Remaining two errors are
RESEARCH examples classified as REFLECTION and EXPERIENCE.

Weights are saved locally in `activity_model_outputs/v3`; the existing V2 weights
are preserved. This run was not uploaded to Hugging Face.
