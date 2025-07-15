import pandas as pd
import numpy as np
import os
import sys
import gc
import shutil

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.append('source/')
from TCRPNet import *
from data import *
from trainer_esm import *
from helper import *

def run_inference(model, dl_test, args_model):
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dl_test), total=len(dl_test)):

            TCRb = batch['cdr'].to(args_model.device)
            epitope = batch['peptide'].to(args_model.device)
            TCR_ohe = batch['cdr_ohe'].to(args_model.device)
            epitope_ohe = batch['peptide_ohe'].to(args_model.device)
            TCRb_mask = batch['cdr_mask'].to(args_model.device)
            epitope_mask = batch['peptide_mask'].to(args_model.device)
            
            y_true_i = batch['y_true'].reshape(-1,1).float()
            y_pred_i = model(TCRb=TCRb, epitope=epitope,
                          TCRb_ohe=TCR_ohe, epitope_ohe=epitope_ohe,
                          TCRb_mask=TCRb_mask, epitope_mask=epitope_mask).detach().cpu()
            y_pred.append(y_pred_i)
            y_true.append(y_true_i)
    return torch.concat(y_pred,0), torch.concat(y_true,0)

if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        help='get config file with everything you need')
    parser.add_argument('--device', type=str, default='cuda:3',
                        help='get config file with everything you need')
    parser.add_argument('--cdr', type=str, default='')
    parser.add_argument('--panel', type=str, default='SB')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--save_filename', type=str, default='')
    args_raw = parser.parse_args()
    args, cfg_name = load_config(args_raw.config)
    
    print('Using config file: ', cfg_name)
    args_data = args.data
    args_model = args.model
    args_model.device = args_raw.device
    cdr3b = args_raw.cdr

    # prep gpu
    torch.cuda.empty_cache()
    gc.collect()
    torch.set_float32_matmul_precision('medium')
    set_seed(args_model)

    # initiate model and data
    model = TCRPNet()
    if args_raw.panel in ['SB', 'WB']:
        data = pd.read_csv('data/netmhc_WBSB_9mers_clean.csv')
        data = data.loc[data['Binding'] == args_raw.panel].reset_index(drop=True)
    elif args_raw.panel == 'SBWB':
        data = pd.read_csv('data/netmhc_WBSB_9mers_clean.csv')
    else:
        data = pd.read_csv(args_raw.panel)
        
    if 'Split' not in data.columns.tolist():
        data['Split'] = 'test'
    data['CDR3_b'] = cdr3b
    data['Score'] = 1
    ds_test = DeepSequencingRawSequenceDataset(args_data, 'test', data=data)
    
    # wrap model into PL
    if args_model.use_pos_weight:
        num_positives = torch.sum(torch.LongTensor(ds_test.data['Score']), dim=0)
        num_negatives = len(torch.LongTensor(ds_test.data['Score'])) - num_positives
        pos_weight  = num_negatives / num_positives
        print('Using pos_weight...')
    else: 
        pos_weight = None
    model = DiscriminativePL(args=args_model, model=model, pos_weight=pos_weight)#.to(args_model.device)

    collate_fn = BatchConverterCollater(alphabet=model.alphabet)
    load_esm = False
    dl_test = DataLoader(ds_test,
                     batch_size=128,
                     shuffle=False,
                     collate_fn=collate_fn)
    
    # load checkpoint
    checkpoint_path = args_model.tb_logger_path + args_model.tb_logger_folder + 'lightning_logs/' + args_model.version_name + '/checkpoints/' 
    checkpoint_path += [i for i in os.listdir(checkpoint_path) if 'last' not in i][0]
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', )
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(args_model.device).eval()

    # run inference
    y_pred, y_true = run_inference(model, dl_test, args_model)
    print(y_pred.shape)

    # create folder to save 
    if len(args_raw.save_path) == 0:
        prediction_path = args_model.tb_logger_path + args_model.tb_logger_folder + 'lightning_logs/' + args_model.version_name + '/prediction/'
    else:
        prediction_path = args_raw.save_path
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)
    else:
        pass
    print('Saving to: ', prediction_path)

    # save
    if len(args_raw.save_filename) == 0:
        fname = 'y_pred_proteome.npy'
    else:
        fname = args_raw.save_filename + '.npy'
    np.save(prediction_path + fname, y_pred)

    