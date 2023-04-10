from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler
import typer
from typer import Argument
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from efls.data import EflsCollator, TextDataset, read_jsonl
from efls.model import EmbeddingFromLanguageModel
from efls.trainer import Trainer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.basicConfig(
    level="INFO",
    datefmt="[%X]",
    format="%(message)s",
    handlers=[RichHandler()],
)
logger = logging.getLogger("rich")


def create_dataloader(
    encoder_tokenzier,
    decoder_tokenzier,
    json_file: Path,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 0,
    shuffle: bool = True,
):
    records = read_jsonl(json_file)
    train_dataset = TextDataset.from_text_pair_similarity_records(records)
    collator = EflsCollator(encoder_tokenzier, decoder_tokenzier, max_length=max_length)
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator, num_workers=num_workers
    )
    return dataloader


def create_adamw_optimizer(model: EmbeddingFromLanguageModel, lr: float, project_lr=1e-3, weight_decay: float = 0.01):
    parameters = list(model.named_parameters())
    # todo: add no decay param group
    optimizer_grouped_parameters = [
        {'params': [p for n, p in parameters if 'project' not in n]},
        {'params': [p for n, p in parameters if 'project' in n], 'lr': project_lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def main(
    encoder_name_or_path: str = Argument(..., help='Encoder name or path, eg: bert-base-uncased'),
    decoder_name_or_path: str = Argument(..., help='Decoder name or path, eg: gpt2-large'),
    train_json_file: Path = Argument(..., help='jsonl format file, must contain "sentence1", "sentence2" fields'),
    embedding_size: int = 300,
    num_efls_tokens: int = 10,
    output_dir: Optional[Path] = None,
    epochs: int = 3,
    seed: int = 24,
    max_length: int = 64,
    batch_size: int = 32,
    lr: float = 5e-5,
    mixed_precision: str = 'bf16',
    gradient_accumulation_steps: int = 1,
    use_tensorboard: bool = False,
):
    set_seed(seed)
    logger.info(f'Start with seed: {seed}')
    output_dir = output_dir or Path('experiments') / 'exp01'
    logger.info(f'Output dir: {output_dir}')
    logger.info(f'Final Efls Model dir: {output_dir / "efls"}')

    project_config = ProjectConfiguration(project_dir=str(output_dir))
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        project_config=project_config,
        log_with=['tensorboard'] if use_tensorboard else None,
    )
    accelerator.init_trackers('efls')
    logger.info(f'Using {accelerator.device} device')
    
    encoder_tokenzier = AutoTokenizer.from_pretrained(encoder_name_or_path, fast=True)
    decoder_tokenzier = AutoTokenizer.from_pretrained(decoder_name_or_path)
    if decoder_tokenzier.pad_token is None:
        decoder_tokenzier.pad_token = decoder_tokenzier.eos_token
    logger.info(f'Creat dataloader from {train_json_file}')
    train_dataloader = create_dataloader(encoder_tokenzier, decoder_tokenzier, train_json_file, batch_size=batch_size, max_length=max_length)
    train_dataloader = accelerator.prepare(train_dataloader)
    
    logger.info(f'Creat model from {encoder_name_or_path} and {decoder_name_or_path}')
    encoder = AutoModel.from_pretrained(encoder_name_or_path)
    decoder = AutoModelForCausalLM.from_pretrained(decoder_name_or_path)
    model = EmbeddingFromLanguageModel(encoder, decoder, projection_size=embedding_size, num_efls_tokens=num_efls_tokens)
    optimizer = create_adamw_optimizer(model, lr, weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps
    )
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=None,
        accelerator=accelerator,
        epochs=epochs,
        core_metric_name='-loss',
        lr_scheduler=lr_scheduler,
        log_interval=10,
        save_on_epoch_end=False
    )
    logger.info('all set, start training')
    trainer.train()
    logger.info(f'Save model to {output_dir}')

    model.efls.save_pretrained(output_dir / 'efls')
    encoder_tokenzier.save_pretrained(output_dir / 'efls')

if __name__ == '__main__':
    typer.run(main)
