"""Offline end-to-end check using a tiny random model; no accuracy claim."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification
    from fastapi.testclient import TestClient
    from activity_data import LABELS
    from predict_example import ActivityClassifier
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory) / 'base'
        base.mkdir()
        vocab = base / 'vocab.txt'
        vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
                                    'title', 'description', 'instructions', ':', 'learning']))
        BertTokenizerFast(vocab_file=str(vocab), model_input_names=['input_ids', 'attention_mask']).save_pretrained(base)
        DistilBertForSequenceClassification(DistilBertConfig(
            vocab_size=10, n_layers=1, n_heads=2, dim=16, hidden_dim=32,
            num_labels=7, id2label=dict(enumerate(LABELS)),
            label2id={label: i for i, label in enumerate(LABELS)})).save_pretrained(base)
        output = Path(directory) / 'trained'
        subprocess.run([sys.executable, str(ROOT / 'train_activity_model.py'),
                        '--data', str(ROOT / 'data/examples.jsonl'), '--model', str(base),
                        '--output', str(output), '--epochs', '1'], check=True)
        subprocess.run([sys.executable, str(ROOT / 'push_model.py'),
                        '--model', str(output), '--dry-run'], check=True)
        result = ActivityClassifier(str(output)).predict({'title': 'Learning'})
        assert result['activityType'] in LABELS
        assert abs(sum(result['scores'].values()) - 1) < 1e-5
        report = json.loads((output / 'evaluation.json').read_text())
        assert len(report['confusion_matrix']) == 7
        os.environ['MODEL_PATH'] = str(output)
        from app import app
        with TestClient(app) as client:
            assert client.get('/health').status_code == 200
            response = client.post('/predict', json={'title': 'Learning'})
            assert response.status_code == 200, response.text
            assert response.json()['activityType'] in LABELS
            assert client.post('/predict', json={}).status_code == 422
            assert client.post('/predict', json={'title': 123}).status_code == 422
        print('Training, checkpoint reload, evaluation, prediction, and API smoke checks passed.')


if __name__ == '__main__':
    main()
