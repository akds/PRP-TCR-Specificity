# PRP-TCR-Specificity

Repo currently under construction, please email Hugh (hughy@uchicago.edu), Ben (ben.lai@czbiohub.org), Jason (jason.perera@czbiohub.org), and Aly (aakhan@uchicago.edu) if you have any questions

## Training Single TCR Models
```
# TODO
```

## Human Proteome Inference
```
python scripts/inference_esm_proteome.py --config <config path> \
      --cdr <cdr3b sequence> \
      --panel <choose between {SB/WB/SBWB/path to csv} \
      --device cuda:0

# Example with 19.2 
python scripts/inference_esm_proteome.py --config configs/single_tcr/19.2_pretrain_finetune.yml \
      --cdr CASSPATYSTDTQYF \
      --panel SBWB \
      --device cuda:0 
```
