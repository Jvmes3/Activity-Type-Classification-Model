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
