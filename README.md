# efls
Embedding From Language model Sufficently

# Usage

1. install requirements

```bash
pip install -r requirements.txt
```

2. train efls model

```bash
# use help: python train_efls.py --help
python train_efls.py \
    --output-dir experiments/stsb01 \
    /data/pretrained_models/bert-base-uncased \
    /data/pretrained_models/gpt2-large \
    datasets/processed/stsb.all.jsonl
```

3. evaluate efls model

```bash
# use help: python eval_efls.py --help
python eval_efls.py datasets/processed/stsb.dev.jsonl experiments/stsb01/efls/
```
