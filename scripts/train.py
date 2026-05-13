import argparse
import gc
import os
import shutil
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append('source/')
from TCRPNet import TCRPNet
from data import BatchConverterCollater, DeepSequencingRawSequenceDataset
from trainer_esm import DiscriminativePL, train_model
from helper import load_config, set_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
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
    model = TCRPNet()
    ds_train = DeepSequencingRawSequenceDataset(args_data, 'train')
    ds_valid = DeepSequencingRawSequenceDataset(args_data, 'valid')
    print('Train size: ', len(ds_train))
    print('Valid size: ', len(ds_valid))

    # wrap model into PL
    if args_model.use_pos_weight:
        scores = torch.LongTensor(ds_train.data['Score'])
        num_positives = torch.sum(scores, dim=0)
        num_negatives = len(scores) - num_positives
        pos_weight = num_negatives / num_positives
        print('Using pos_weight...')
    else:
        pos_weight = None
    model = DiscriminativePL(args=args_model, model=model, pos_weight=pos_weight)

    # build dataloaders
    collate_fn = BatchConverterCollater(alphabet=model.alphabet)
    dl_train = DataLoader(
        ds_train, batch_size=args_data.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    dl_valid = DataLoader(
        ds_valid, batch_size=args_data.batch_size, shuffle=False, collate_fn=collate_fn,
    )
    args_model.traindata_len = len(dl_train) // len(args_model.device_num)

    # load checkpoint
    if not pd.isnull(args_model.pretrained_weights):
        print('Using checkpoint to finetune...')
        checkpoint = torch.load(args_model.pretrained_weights, map_location='cpu')
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
    cfg_dest = os.path.join(
        args_model.tb_logger_path,
        args_model.tb_logger_folder,
        'lightning_logs',
        args_model.version_name,
        'cfg.yml',
    )
    shutil.copyfile(args_raw.config, cfg_dest)
