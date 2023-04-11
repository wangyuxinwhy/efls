from pathlib import Path

import typer

from efls.predictors import EflsEmbeddingPredictor, evaluate_spearman
from efls.data import read_jsonl


def main(
    json_file: Path = typer.Argument(
        ...,
        help='jsonl format file, must contain "sentence1", "sentence2", "score" fields',
    ),
    model_path: Path = typer.Argument(..., help='efls model path'),
    whitening: bool = True,
):
    efls_predictor = EflsEmbeddingPredictor.from_pretrained(model_path)
    records = read_jsonl(json_file)
    spearman_score = evaluate_spearman(efls_predictor, records, whitening)
    print(f'Evaluation on {json_file.name}, model: {model_path}, whitening: {whitening}')
    print(f'Spearman score: {spearman_score:.4f}')


if __name__ == '__main__':
    typer.run(main)
