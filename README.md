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

================================================================================================
# download model weights
# if you want to try with just TCR 19.2 model
gdown --fuzzy https://drive.google.com/file/d/16VoHYgtEMFRiaOy34iyouYb5D5QDRm-b/view?usp=sharing
unzip weights.zip
rm weights.zip

# to download all models [WARNING: Compressed .tar.gz is > 180 GB]
gdown --fuzzy https://drive.google.com/file/d/1-ww-aI2QQ2NeZ3TOXvMkHbovnN4VQRCo/view?usp=sharing
tar -xzvf weights.tar.gz
rm weights.tar.gz

# to download pretrained VDJdb model
gdown --fuzzy https://drive.google.com/file/d/17KhZvZSm-XGDcOqZTlcjdpexeXNxuqCK/view?usp=sharing
tar -xzvf vdjdb-pretrained-ckpt.tar.gz
rm vdjdb-pretrained-ckpt.tar.gz

================================================================================================
# download all AS data
gdown --fuzzy https://drive.google.com/file/d/1F_aYA7fvd-P46uOBeGZWOQsRS7IMQYJ8/view?usp=sharing
unzip AS-data-all.zip -d data/
```

## Training Single TCR Models
```
python scripts/train.py <config>
```

For an example with TCR 19.2 (make sure to download example data first!)
```
python scripts/train.py configs/single_tcr/19.2_example_train.yml
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
python scripts/inference_proteome.py --config configs/single_tcr/19.2_example_inference.yml \
      --cdr CASSPATYSTDTQYF \
      --panel SBWB \
      --device cuda:0 
```
