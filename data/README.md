`examples.jsonl` contains 21 manually authored synthetic examples, three per type.
It is a schema and pipeline fixture, not a representative training or evaluation dataset.

Put approved activity exports in this directory (ignored by Git). Convert exports to
JSONL or a JSON array with string fields `title`, `description`, `instructions`, and
one exact `activityType`. At least one text field must be nonempty. Keep provenance
such as `source` and activity IDs for auditing; these are not classifier inputs.

Prioritize real labeled activities, review ambiguous cases by their primary learning
objective, and supplement scarce classes with reviewed synthetic examples only.
Keep related activities and synthetic variants together before creating a final
external evaluation set. The automatic split removes exact normalized duplicates,
but cannot detect paraphrases or activity families. Three examples per class is only
a technical minimum. Evaluate on a larger, independently labeled real dataset before use.

## Synthetic starter dataset (synthetic-v1)

`activities.jsonl` contains 350 AI-authored synthetic activities: 50 for each of the
seven labels. `activities.synthetic.json` contains the exact same records as a
formatted JSON array. Use either file, not both combined. These synthetic files are versioned in Git; other activity exports remain ignored.
They are not uploaded by `push_model.py`.

Every record includes `title`, `description`, `instructions`, `activityType`, a
unique `id`, `source: ai_generated_synthetic`, `dataset_version: synthetic-v1`,
and `human_reviewed: false`. The existing loader uses only the three text fields
and the target, excluding provenance from model inputs. No MyVillage records were
used to create these examples. Labels reflect the roadmap's working definitions,
not independently confirmed MyVillage annotation policy.

Topics recur across labels to illustrate differences in primary learning objective.
Each activity was authored separately rather than expanded by replacing topic words
in a template. The prose still has synthetic stylistic regularities. Review label
choices, especially mixed activities, and revise wording before treating this as a
training resource. Some guided participation examples could also involve practice;
these examples emphasize firsthand exposure as their principal objective.

The current split produces 252 training, 49 validation, and 49 test records, with
36/7/7 examples per class. These are all synthetic: reported metrics do not measure
real MyVillage performance. The splitter does not group semantic similarities or
related topics. Obtain a separately held-out, human-labeled real evaluation set
when approved access becomes available.

```bash
python train_activity_model.py --data data/activities.jsonl --validate-only
python train_activity_model.py --data data/activities.jsonl --epochs 3 --output activity_model_outputs/v2
python predict_example.py --model activity_model_outputs/v2 --title "Help a friend understand fractions" --instructions "Explain the method and check their understanding."
```

Training and uploading have not been run as part of dataset creation. If you later
upload a checkpoint trained on this file, explicitly describe its synthetic-only
training data in the model card.
