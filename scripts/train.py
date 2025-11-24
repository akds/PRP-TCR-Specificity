import pandas as pd
import numpy as np
import os
import esm

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import sys
import gc
import shutil

sys.path.append('source/')
from TCRPNet import *
from data import *
from trainer_esm import *
from helper import *
    
if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, 
                        help='get config file with everything you need')
    args_raw = parser.parse_args()
    args, cfg_name = load_config(args_raw.config)
    print('Using config file: ', cfg_name)
    args_data = args.data
    args_model = args.model

    torch.cuda.empty_cache()
    gc.collect()
    torch.set_float32_matmul_precision('medium')
    set_seed(args_model)

    # grab base model and prep dataset
    model = TCRPNet(use_attn=args_model.use_attn)
    ds_train = DeepSequencingRawSequenceDataset(args_data, 'train')
    ds_valid = DeepSequencingRawSequenceDataset(args_data, 'valid')
    print('Train size: ', len(ds_train))
    print('Valid size: ', len(ds_valid))


    # wrap model into PL
    if args_model.use_pos_weight:
        num_positives = torch.sum(torch.LongTensor(ds_train.data['Score']), dim=0)
        num_negatives = len(torch.LongTensor(ds_train.data['Score'])) - num_positives
        pos_weight  = num_negatives / num_positives
        print('Using pos_weight...')
    else: 
        pos_weight = None
    model = DiscriminativePL(args=args_model, model=model, pos_weight=pos_weight)#.to(args_model.device)

    # build dataloaders
    collate_fn = BatchConverterCollater(alphabet=model.alphabet)
    dl_train = DataLoader(ds_train,
                             batch_size=args_data.batch_size,
                             shuffle=True,
                             collate_fn=collate_fn)
    dl_valid = DataLoader(ds_valid,
                          batch_size=args_data.batch_size,
                          shuffle=False,
                          collate_fn=collate_fn)
    if type(args_model.gpu_devices) == list:
        args_model.traindata_len = len(dl_train) // len(args_model.gpu_devices) #// args.acc_grad_batches
    elif type(args_model.gpu_devices) == int: 
        args_model.traindata_len = len(dl_train) // args_model.gpu_devices #// args.acc_grad_batches

    # load checkpoint
    if pd.isnull(args_model.pretrained_weights) == False:
        print('Using checkpoint to finetune...')
        checkpoint_path = args_model.pretrained_weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print('Not using pretrained checkpoint...')

    # train model
    trained_model = train_model(
            args=args_model,
            model=model,
            train_dataloader=dl_train,
            valid_dataloader=dl_valid,
            test_dataloader=None, 
            strategy=args_model.strategy,
    )

    # save copy of cfg file
    shutil.copyfile(args_raw.config, 
                    args_model.tb_logger_path + args_model.tb_logger_folder + 'lightning_logs/' + args_model.version_name + '/cfg.yml')






