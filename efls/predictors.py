from __future__ import annotations
from enum import Enum

from pathlib import Path
from typing import Protocol, cast

import torch
from transformers import AutoTokenizer, BertModel

from efls.model import Efls


class EmbbedingPredictor(Protocol):
    def __call__(self, texts: list[str], **kwargs) -> torch.Tensor:
        ...


def compute_kernel_bias(vecs: torch.Tensor):
    mu = vecs.mean(dim=0, keepdim=True)
    cov = torch.cov(vecs.T)
    svd_result = torch.linalg.svd(cov)
    W = torch.mm(svd_result.U, torch.diag(1 / torch.sqrt(svd_result.S)))
    return W, -mu


def evaluate_spearman(efls_predictor: EmbbedingPredictor, records: list[dict], whitening: bool = True, **kwargs):
    from torchmetrics.regression.spearman import SpearmanCorrCoef

    sentences1 = [record['sentence1'] for record in records]
    sentences2 = [record['sentence2'] for record in records]
    embeddings1 = efls_predictor(sentences1, **kwargs)
    embeddings2 = efls_predictor(sentences2, **kwargs)
    if whitening:
        all_embeddings = torch.cat([embeddings1, embeddings2], dim=0)
        kernel, bias = compute_kernel_bias(all_embeddings)
        embeddings1 = torch.mm(embeddings1 + bias, kernel)
        embeddings2 = torch.mm(embeddings2 + bias, kernel)
    preds = torch.cosine_similarity(embeddings1, embeddings2, dim=1)
    targets = torch.tensor([float(record['score']) for record in records])
    spearman_score = SpearmanCorrCoef()(preds, targets)
    return spearman_score


class EflsEmbeddingPredictor(EmbbedingPredictor):
    def __init__(self, efsl_model: Efls, tokenzier, max_length: int = 64):
        self.efls_model = efsl_model
        self.efls_model.eval()
        self.tokenizer = tokenzier
        self.device = next(self.efls_model.parameters()).device
        self.max_length = max_length

    def __call__(self, texts: list[str]) -> torch.Tensor:
        embeddings = []
        for batch in self._generate_batch(texts):
            embeddings.extend(self._batch_embedding(batch))
        embeddings = torch.stack(embeddings, dim=0)
        return embeddings
    
    def _batch_embedding(self, texts: list[str]) -> torch.Tensor:
        encodes = self.tokenizer(texts, return_tensors='pt', padding=True, return_token_type_ids=False, truncation=True, max_length=self.max_length).to(device=self.device)
        with torch.no_grad():
            batch_embeddings = self.efls_model(**encodes)['embeddings'].mean(dim=1)
        return batch_embeddings.cpu().detach()
    
    @staticmethod
    def _generate_batch(texts: list[str], batch_size: int = 32):
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Path | str):
        pretrained_model_path = Path(pretrained_model_path)
        efsl_model = Efls.from_pretrained(pretrained_model_path)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        efsl_model.to(device)
        return cls(efsl_model, tokenizer)


class BertEmbeddingMode(str, Enum):
    static = 'static'
    first_last = 'first_last'
    last = 'last'


class BertEmbeddingPredictor(EmbbedingPredictor):
    def __init__(self, bert_model: BertModel, tokenzier, mode: BertEmbeddingMode | str = BertEmbeddingMode.static, max_length: int = 512) -> None:
        self.bert_model = bert_model
        self.bert_model.eval()
        self.tokenizer = tokenzier
        self.device = next(self.bert_model.parameters()).device
        self.max_length = max_length
        self.mode = BertEmbeddingMode(mode)
    
    def __call__(self, texts: list[str], mode: BertEmbeddingMode | str | None = None) -> torch.Tensor:
        model = BertEmbeddingMode(mode) if mode else self.mode
        embeddings = []
        for batch in self._generate_batch(texts):
            if model == BertEmbeddingMode.static:
                embeddings.extend(self._batch_static_embedding(batch))
            elif model == BertEmbeddingMode.first_last:
                embeddings.extend(self._batch_first_last_embedding(batch))
            elif model == BertEmbeddingMode.last:
                embeddings.extend(self._batch_last_embedding(batch))
        embeddings = torch.stack(embeddings, dim=0)
        return embeddings
    
    def _batch_first_last_embedding(self, texts: list[str]) -> torch.Tensor:
        encodes = self.tokenizer(texts, return_tensors='pt', padding=True, return_token_type_ids=False, truncation=True, max_length=self.max_length).to(device=self.device)
        with torch.no_grad():
            attention_mask = encodes['attention_mask']
            all_hidden_states = self.bert_model(**encodes, output_hidden_states=True)['hidden_states']
            first_layer_hidden_states = all_hidden_states[0]
            first_layer_hidden_states[~attention_mask.bool()] = 0
            first_layer_mean_embedding = first_layer_hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            last_layer_hidden_states = all_hidden_states[-1]
            last_layer_hidden_states[~attention_mask.bool()] = 0
            last_layer_mean_embedding = last_layer_hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            batch_embeddings = (first_layer_mean_embedding + last_layer_mean_embedding) / 2
        return batch_embeddings.cpu().detach()

    def _batch_last_embedding(self, texts: list[str]) -> torch.Tensor:
        encodes = self.tokenizer(texts, return_tensors='pt', padding=True, return_token_type_ids=False, truncation=True, max_length=self.max_length).to(device=self.device)
        with torch.no_grad():
            attention_mask = encodes['attention_mask']
            hidden_states = self.bert_model(**encodes)['last_hidden_state']
            hidden_states[~attention_mask.bool()] = 0
            batch_embeddings = hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return batch_embeddings.cpu().detach()

    def _batch_static_embedding(self, texts: list[str]) -> torch.Tensor:
        encodes = self.tokenizer(texts, return_tensors='pt', padding=True, return_token_type_ids=False, truncation=True, max_length=self.max_length).to(device=self.device)
        with torch.no_grad():
            attention_mask = encodes['attention_mask']
            hidden_states = self.bert_model.embeddings(input_ids=encodes['input_ids'])
            hidden_states[~attention_mask.bool()] = 0
            batch_embeddings = hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return batch_embeddings.cpu().detach()

    @staticmethod
    def _generate_batch(texts: list[str], batch_size: int = 32):
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Path | str):
        pretrained_model_path = Path(pretrained_model_path)
        bert_model = BertModel.from_pretrained(pretrained_model_path)
        bert_model = cast(BertModel, bert_model)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        bert_model.to(device)
        return cls(bert_model, tokenizer)
