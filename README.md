# PRP-TCR-Specificity
Repository for ["Deep peptide recognition profiling decodes TCR specificity and enables disease-associated antigen discovery"](https://www.nature.com/articles/s41587-026-03128-x) (*Nature Biotechnology*, 2026).

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
gdown --fuzzy https://drive.google.com/file/d/1shbyIZRiS-ZWDhqsLz2cY9RDVivuLVig/view?usp=sharing
unzip ASdata-all.zip
```

## Download Pretrained Weights

### VDJdb pretrained checkpoint
The single-TCR configs in `configs/single_tcr/` initialize from a VDJdb-pretrained
checkpoint for faster convergence. Download this if you would like to use it, otherwise, modify configs by removing path assigned to `pretrained_weights`.

```
gdown --fuzzy https://drive.google.com/file/d/1SyGIQ3UpUE6ovzmFnA_dgbuZbcpECoDq/view?usp=sharing
unzip vdjdb-pretrained-ckpt.zip
rm vdjdb-pretrained-ckpt.zip
```

### Single-TCR checkpoints
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

If you only need a subset of the 22 checkpoints, each one is also mirrored as
its own zip on Hugging Face at
[hyeh/PRP-TCR-Specificity](https://huggingface.co/hyeh/PRP-TCR-Specificity),
under `lightning_logs/<tcr>_finetune_esm.zip`.

```bash
pip install huggingface_hub
mkdir -p model_outputs/single_tcr/lightning_logs/
huggingface-cli download hyeh/PRP-TCR-Specificity \
    lightning_logs/19.2_finetune_esm.zip \
    --repo-type model --local-dir .
unzip lightning_logs/19.2_finetune_esm.zip \
    -d model_outputs/single_tcr/lightning_logs/
rm lightning_logs/19.2_finetune_esm.zip
```

Each zip expands to a single <tcr>_finetune_esm/ directory containing checkpoints/ and outputs/. The
premade configs under configs/single_tcr/ reference paths of the form
model_outputs/single_tcr/lightning_logs/<tcr>_finetune_esm/..., so the
contents must end up at exactly that location for inference scripts and
notebooks to find them. 

## Repository layout
```
configs/
  single_tcr/<tcr>_model.yml      training/inference config per TCR
  joint_19.2_tcr/                 joint model configs
scripts/                          training + inference entry points (see below)
source/                           model, dataset, and trainer code
notebooks/                        analysis notebooks (see below)
data/                             contains CDR3b sequences, activation data, and netMHC panels and also store deep sequencing data here
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

To train on your own data, format the input CSV to match the files under
`data/ASdata-all/` — required columns are `CDR3_b` (CDR3β sequence), `Epitope`
(peptide), `Score` (0/1 label), and `Split` (one of `train`, `valid`, `test`).
Point your config's `data.data_path` at the new CSV.

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

## Citations

If you use this code or data, please cite:

```bibtex
@article{wang2026deep,
  title={Deep peptide recognition profiling decodes TCR specificity and enables disease-associated antigen discovery},
  author={Wang, Nan and Yeh, Hugh and Lai, Ben and Perera, Jason and Jude, Kevin M and Risch, Isabel and Um, Joy and Chen, Xiaojing and Xiang, Xinyu and Wang, Chunyu and others},
  journal={Nature Biotechnology},
  pages={1--11},
  year={2026},
  publisher={Nature Publishing Group US New York}
}
```

---

Please email Hugh (hughy@uchicago.edu), Ben (ben.lai@czbiohub.org), Jason (jason.perera@czbiohub.org), or Aly (aakhan@uchicago.edu) if you have any questions.
