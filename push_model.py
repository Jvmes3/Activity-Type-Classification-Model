"""Upload validated trained artifacts to the project's Hugging Face model repo."""
import argparse
from pathlib import Path
import shutil
import tempfile
from predict_example import ActivityClassifier, HF_REPO_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='activity_model_outputs')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    folder = Path(args.model)
    if not (folder / 'config.json').is_file() or not (folder / 'model.safetensors').is_file():
        parser.error('No trained model found. Train first with train_activity_model.py.')
    # Verify that both tokenizer and weights load and the seven labels match.
    ActivityClassifier(str(folder))
    names = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
             'special_tokens_map.json', 'vocab.txt', 'added_tokens.json', 'evaluation.json']
    with tempfile.TemporaryDirectory() as directory:
        staging = Path(directory)
        for name in names:
            if (folder / name).is_file():
                shutil.copy2(folder / name, staging / name)
        shutil.copy2(Path(__file__).with_name('MODEL_CARD.md'), staging / 'README.md')
        print(f'Destination: https://huggingface.co/{HF_REPO_ID}')
        print('Files: ' + ', '.join(sorted(p.name for p in staging.iterdir())))
        if args.dry_run:
            return
        from huggingface_hub import HfApi, get_token
        if not get_token():
            parser.error('Log in first with hf auth login, or set HF_TOKEN in your environment.')
        result = HfApi().upload_folder(
            repo_id=HF_REPO_ID, repo_type='model', folder_path=staging,
            commit_message='Upload activity classifier model and evaluation')
        print(result.commit_url)


if __name__ == '__main__':
    main()
