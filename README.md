# PRP-TCR-Specificity
Repository for "Deep peptide recognition profiling decodes TCR specificity and enables autoantigen discovery"

<p align="center">
  <img src="https://github.com/akds/PRP-TCR-Specificity/blob/main/PRP-TCR-Specificity.png" alt="Logo">
</p>

Repo currently under construction, please email Hugh (hughy@uchicago.edu), Ben (ben.lai@czbiohub.org), Jason (jason.perera@czbiohub.org), and Aly (aakhan@uchicago.edu) if you have any questions

## Installation
```
git clone https://github.com/akds/PRP-TCR-Specificity.git
cd PRP-TCR-Specificity/

# install environment
conda env create -f environment.yml
conda activate prp
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# download weights
gdown --fuzzy https://drive.google.com/file/d/16VoHYgtEMFRiaOy34iyouYb5D5QDRm-b/view?usp=sharing
unzip weights.zip
rm weights.zip
```


## Training Single TCR Models
```
# TODO
```

## Human Proteome Inference
```
python scripts/inference_proteome.py --config <config path> \
      --cdr <cdr3b sequence> \
      --panel <choose between {SB/WB/SBWB/path to csv} \
      --device cuda:0
```

For an example with TCR 19.2
```
# Example with 19.2 
python scripts/inference_proteome.py --config configs/single_tcr/19.2_pretrain_finetune.yml \
      --cdr CASSPATYSTDTQYF \
      --panel SBWB \
      --device cuda:0 
```
