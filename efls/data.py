from __future__ import annotations

import json
from typing import Sequence, TypedDict
from pathlib import Path

from torch.utils.data import Dataset


class TextPairSimilarityRecord(TypedDict):
    sentence1: str
    sentence2: str
    score: float


class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index: int):
        return self.texts[index]

    @classmethod
    def from_text_file(cls, file_path: str):
        with open(file_path, 'r') as f:
            texts = f.readlines()
        texts = [text.strip() for text in texts if text.strip()]
        return cls(texts)

    @classmethod
    def from_text_pair_similarity_records(cls, records: list[TextPairSimilarityRecord]):
        texts = []
        for record in records:
            texts.append(record['sentence1'])
            texts.append(record['sentence2'])
        return cls(texts)


class EflsCollator:
    def __init__(self, encoder_tokenzier, decoder_tokenzier, max_length=64):
        self.encoder_tokenzier = encoder_tokenzier
        self.decoder_tokenzier = decoder_tokenzier
        self.max_length = max_length

    def __call__(self, batch: list[str]):
        encoder_encodings = self.encoder_tokenzier(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_token_type_ids=False,
        )
        decoder_encodings = self.decoder_tokenzier(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'encoder_input_ids': encoder_encodings['input_ids'],
            'decoder_input_ids': decoder_encodings['input_ids'],
            'encoder_attention_mask': encoder_encodings['attention_mask'],
            'decoder_attention_mask': decoder_encodings['attention_mask'],
        }


def read_from_tsv(tsv_file: Path, header: bool = True):
    with open(tsv_file, 'r') as f:
        lines = f.readlines()

    if header:
        keys = lines[0].strip().split('\t')
        lines = lines[1:]
    else:
        num_columns = len(lines[0].strip().split('\t'))
        keys = [f'column_{i}' for i in range(num_columns)]

    records: list[dict[str, str]] = []
    for line in lines:
        values = line.strip().split('\t')
        if len(values) != len(keys):
            print(line)
            continue
        records.append(dict(zip(keys, values)))
    return records


def read_jsonl(json_file: Path | str):
    json_file = Path(json_file)
    records = []
    with open(json_file, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records
