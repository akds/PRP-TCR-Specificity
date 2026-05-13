# PRP-TCR-Specificity
Repository for "Deep peptide recognition profiling decodes TCR specificity and enables disease-associated antigen discovery"

<p align="center">
  <img src="https://github.com/akds/PRP-TCR-Specificity/blob/main/PRP-TCR-Specificity.png" alt="Logo">
</p>

## Installation
```
git clone https://github.com/akds/PRP-TCR-Specificity.git
cd PRP-TCR-Specificity/

# install environment
conda env create -f environment.yml
conda activate prp

# download ESM2-650M
mkdir esm
cd esm
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
```

## Download Data
```
cd data/
gdown --fuzzy https://drive.google.com/file/d/1F_aYA7fvd-P46uOBeGZWOQsRS7IMQYJ8/view?usp=sharing
unzip ASdata-all.zip
```

## Download Pretrained Weights

### VDJdb pretrained checkpoint
The single-TCR configs in `configs/single_tcr/` initialize from a VDJdb-pretrained
checkpoint for faster convergence. Download this if you would like to use it, otherwise, modify configs by removing path assigned to `pretrained_weights`.

```
gdown --fuzzy https://drive.google.com/file/d/17KhZvZSm-XGDcOqZTlcjdpexeXNxuqCK/view?usp=sharing
tar -xzvf vdjdb-pretrained-ckpt.tar.gz
rm vdjdb-pretrained-ckpt.tar.gz
```

### Single-TCR checkpoints (`single_tcr.zip`)
All 22 single-TCR checkpoints + cached test/proteome predictions, packaged as
`single_tcr.zip`. Drop the contents into `model_outputs/` so the layout matches
what the scripts and notebooks expect:

```
mkdir model_outputs/
gdown --fuzzy https://drive.google.com/file/d/1UV9AKkajvnUpnSqDxU7jXJeQPHzMuAit/view?usp=sharing
unzip single_tcr.zip -d model_outputs/
rm single_tcr.zip
```

This produces:
```
model_outputs/single_tcr/lightning_logs/<tcr>_finetune_esm/
    checkpoints/best-checkpoint.ckpt
    outputs/y_pred_test.npy
    outputs/y_pred_proteome.npy
```
The configs under `configs/single_tcr/` point at `model_outputs/single_tcr/...`,
so no further changes are needed.

## Repository layout
```
configs/
  single_tcr/<tcr>_model.yml      training/inference config per TCR
  joint_19.2_tcr/                 joint model configs
scripts/                          training + inference entry points (see below)
source/                           model, dataset, and trainer code
notebooks/                        analysis notebooks (see below)
data/                             directory for data, CDR3b sequences, and netMHC panels
```

## Scripts

### `scripts/train.py`
Train a single-TCR model from a config. Use the VDJdb pretrained
checkpoint for faster convergence. Best checkpoint is
saved as `best-checkpoint.ckpt` under
`model_outputs/<folder>/lightning_logs/<version>/checkpoints/`.
```
python scripts/train.py <config>

# example: train TCR 19.2 (download data first)
python scripts/train.py configs/single_tcr/19.2_model.yml
```

### `scripts/inference_test.py`
Run inference on the held-out **test split** defined by the config's
`data_path`. Saves `y_pred_test.npy` (logits) to the model's `outputs/` dir.
```
python scripts/inference_test.py --config <config> [--device cuda:0] [--batch_size 128] \
    [--save_path <dir>] [--save_filename <name>] [--save_y_true]

# example: TCR 019.1
python scripts/inference_test.py --config configs/single_tcr/019.1_model.yml
```

### `scripts/inference_proteome.py`
Score a CDR3β across a peptide panel (netMHC SB/WB 9mers, or any CSV with an
`Epitope` column). Saves `y_pred_proteome.npy` to `outputs/`.
```
python scripts/inference_proteome.py --config <config> \
    (--cdr <CDR3b> | --tcr_id <id from data/tcr_cdr3b.csv>) \
    --panel {SB|WB|SBWB|path/to/peptides.csv} \
    [--device cuda:0]

# example: pass the CDR3β sequence directly
python scripts/inference_proteome.py --config configs/single_tcr/19.2_model.yml \
    --cdr CASSPATYSTDTQYF --panel SBWB --device cuda:0

# example: look up the CDR3β by TCR id from data/tcr_cdr3b.csv
python scripts/inference_proteome.py --config configs/single_tcr/19.2_model.yml \
    --tcr_id 19.2 --panel SBWB --device cuda:0
```

### Batch shell helpers
Iterate every config in `configs/single_tcr/`:
```
bash scripts/inference_all_single_tcr.sh                   # test-set inference, all TCRs
bash scripts/inference_proteome_all_single_tcr.sh          # proteome inference, all TCRs
```

## Notebooks

### `notebooks/01_SingleTCRModel_TestSet_Predictions.ipynb`
Loads test-set logits from each TCR's
`model_outputs/single_tcr/lightning_logs/<tcr>_finetune_esm/outputs/y_pred_test.npy`,
applies the same `Epitope` filtering as the dataset class, and produces ROC and
PR curves per TCR against the labels in `data/ASdata-all/`. Requires
`scripts/inference_test.py` (or `inference_all_single_tcr.sh`) to have been run
first — or use the cached predictions shipped in `single_tcr.zip`.

### `notebooks/02_SingleTCRModel_Proteome_Predictions.ipynb`
Loads proteome logits from each TCR's
`outputs/y_pred_proteome.npy`, attaches them to
`data/netmhc_WBSB_9mers_clean.csv`, scores against
`data/activation_binary.csv`, and filters for 9mers with proline at P8.
Requires `scripts/inference_proteome.py` (or
`inference_proteome_all_single_tcr.sh`) to have been run — or use the cached
predictions in `single_tcr.zip`.

---

Please email Hugh (hughy@uchicago.edu), Ben (ben.lai@czbiohub.org), Jason (jason.perera@czbiohub.org), or Aly (aakhan@uchicago.edu) if you have any questions.
