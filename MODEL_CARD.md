---
language:
- en
library_name: transformers
pipeline_tag: text-classification
tags:
- distilbert
- myvillage
- activity-classification
---

# MyVillage Activity Type Classifier

Classifies an activity into one of REFLECTION, RESEARCH, COLLABORATE, CREATE,
PRACTICE, EXPERIENCE, or TEACH. This is single-label classification with softmax.

Source code, training instructions, and API:
https://github.com/Jvmes3/Activity-Type-Classification-Model

## Input format

Use the same formatting as training: normalize whitespace within each field and
join these three lines, retaining empty fields when omitted:

```text
title: Learning journal
description: Review your learning.
instructions: Explain what changed in your understanding.
```

Load this repository with `AutoTokenizer.from_pretrained` and
`AutoModelForSequenceClassification.from_pretrained`, tokenize with truncation and
`max_length=512`, and apply softmax to the logits. Map the highest-scoring index
through `model.config.id2label`. The GitHub project's `ActivityClassifier` does this.

## Evaluation and limitations

See `evaluation.json`, when present, for metrics, class counts, the source-data hash,
and base model. Metrics depend on the uploaded checkpoint and dataset; the repository
name alone does not establish training quality. Scores from the included synthetic
fixture are pipeline demonstrations, not evidence of real-world accuracy.

Confidence scores are uncalibrated. Review mixed-objective activities and evaluate
on independent, approved MyVillage data before using predictions for routing.
Inputs longer than 512 tokens are truncated. Confirm training-data provenance and
usage rights before distributing a checkpoint.
