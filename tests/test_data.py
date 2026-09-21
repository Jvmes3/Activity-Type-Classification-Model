import json
import tempfile
import unittest
from pathlib import Path
from activity_data import LABELS, activity_text, load_records, split_records


class DataTests(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.json'
            path.write_text(json.dumps(rows))
            return load_records(path)

    def test_all_labels_and_disjoint_reproducible_splits(self):
        rows = self.load([{'title': f'{label} example {n}', 'activityType': label}
                          for label in LABELS for n in range(10)])
        splits = split_records(rows)
        self.assertEqual(splits, split_records(rows))
        texts = [row['text'] for group in splits.values() for row in group]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertEqual(len(texts), len(rows))
        for group in splits.values():
            self.assertEqual({r['labels'] for r in group}, set(range(7)))

    def test_duplicate_and_conflicting_labels(self):
        row = {'title': '  A journal ', 'activityType': 'REFLECTION'}
        self.assertEqual(len(self.load([row, dict(row, title='a   journal')])), 1)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            self.load([row, dict(row, activityType='CREATE')])

    def test_reject_bad_input(self):
        for row in ({}, {'title': 123}, {'title': 'Hello', 'activityType': 'UNKNOWN'},
                    {'title': 'Hello', 'activityType': ['TEACH']}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.load([row])

    def test_no_target_leakage(self):
        text = activity_text({'title': 'Look back', 'instructions': 'Write notes',
                              'activityType': 'REFLECTION', 'source': 'synthetic'})
        self.assertNotIn('REFLECTION', text)
        self.assertNotIn('synthetic', text)
        self.assertIn('Write notes', text)

    def test_insufficient_data(self):
        with self.assertRaisesRegex(ValueError, 'at least 3'):
            split_records([])


if __name__ == '__main__':
    unittest.main()
