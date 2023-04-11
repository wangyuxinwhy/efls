from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel


def frozen_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


class Efls(nn.Module):
    EFLS_PROJECTOR_STATE_NAME = 'efls_projector_model.bin'
    efls_dummy_token_ids: torch.Tensor

    def __init__(
        self,
        encoder: PreTrainedModel,
        hidden_size: int = 300,
        num_semantic_tokens: int = 10,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_semantic_tokens = num_semantic_tokens
        self.hidden_size = hidden_size
        self.projector = nn.Linear(encoder.config.hidden_size, hidden_size)  # type: ignore
        self.register_buffer(
            'efls_dummy_token_ids',
            torch.tensor(list(range(99 - num_semantic_tokens, 99)), dtype=torch.long),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        input_ids = torch.cat(
            [
                self.efls_dummy_token_ids.unsqueeze(0).expand(input_ids.shape[0], -1),
                input_ids,
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                torch.full_like(self.efls_dummy_token_ids, 1).unsqueeze(0).expand(attention_mask.shape[0], -1),
                attention_mask,
            ],
            dim=1,
        )
        last_hidden_state = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        cls_embeddings = last_hidden_state[:, : self.num_semantic_tokens, :]
        cls_embeddings = self.projector(cls_embeddings)
        output_dict = {
            'embeddings': cls_embeddings,
        }
        return output_dict

    def save_pretrained(self, output_dir: Path | str):
        output_dir = Path(output_dir)
        self.encoder.config.efls_hidden_size = self.hidden_size
        self.encoder.config.efls_num_semantic_tokens = self.num_semantic_tokens
        self.encoder.save_pretrained(output_dir)
        torch.save(
            self.projector.state_dict(),
            output_dir / self.EFLS_PROJECTOR_STATE_NAME,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Path | str):
        pretrained_model_path = Path(pretrained_model_path)
        encoder = AutoModel.from_pretrained(pretrained_model_path)
        hidden_size = encoder.config.efls_hidden_size
        num_semantic_tokens = encoder.config.efls_num_semantic_tokens
        model = cls(encoder, hidden_size, num_semantic_tokens)
        state_dict = torch.load(pretrained_model_path / cls.EFLS_PROJECTOR_STATE_NAME)
        model.projector.load_state_dict(state_dict)
        return model


class EmbeddingFromLanguageModel(nn.Module):
    efls_dummy_token_ids: torch.Tensor

    def __init__(
        self,
        encoder: PreTrainedModel,
        decoder: PreTrainedModel,
        projection_size: int = 300,
        num_efls_tokens: int = 10,
    ):
        super().__init__()
        self.efls = Efls(encoder, projection_size, num_efls_tokens)
        self.num_tokens = num_efls_tokens
        self.register_buffer(
            'efls_dummy_token_ids',
            torch.tensor([1] * num_efls_tokens, dtype=torch.long),
        )
        self.decoder = decoder
        self.decoder_projector = nn.Linear(projection_size, decoder.config.hidden_size, bias=False)   # type: ignore
        self.decoder_embeddings = self.decoder.get_input_embeddings()   # type: ignore
        frozen_model(self.decoder)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            frozen_model(self.decoder)
        return self

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ):
        efls_embeddings = self.efls(encoder_input_ids, encoder_attention_mask)['embeddings']
        projected_efls_embeddings = self.decoder_projector(efls_embeddings)

        labels = decoder_input_ids.clone()
        if decoder_attention_mask is not None:
            labels[~decoder_attention_mask.bool()] = -100
        decoder_input_ids = torch.cat(
            [
                self.efls_dummy_token_ids.unsqueeze(0).expand(decoder_input_ids.shape[0], -1),
                decoder_input_ids,
            ],
            dim=1,
        )
        labels = torch.cat(
            [
                torch.full_like(self.efls_dummy_token_ids, -100).unsqueeze(0).expand(labels.shape[0], -1),
                labels,
            ],
            dim=1,
        )
        decoder_input_embeddings = self.decoder_embeddings(decoder_input_ids)
        decoder_input_embeddings[:, : self.num_tokens, :] = projected_efls_embeddings

        output_dict = self.decoder(
            labels=labels,
            inputs_embeds=decoder_input_embeddings,
            return_dict=True,
        )
        return output_dict
