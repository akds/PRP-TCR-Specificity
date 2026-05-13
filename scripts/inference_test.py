import argparse
import gc
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('source/')
from TCRPNet import TCRPNet
from data import BatchConverterCollater, DeepSequencingRawSequenceDataset
from trainer_esm import DiscriminativePL
from helper import load_config, set_seed


def run_inference(model, dl_test, args_model):
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in tqdm(dl_test, total=len(dl_test)):
            TCRb = batch['cdr'].to(args_model.device)
            epitope = batch['peptide'].to(args_model.device)
            TCR_ohe = batch['cdr_ohe'].to(args_model.device)
            epitope_ohe = batch['peptide_ohe'].to(args_model.device)
            TCRb_mask = batch['cdr_mask'].to(args_model.device)
            epitope_mask = batch['peptide_mask'].to(args_model.device)

            y_true_i = batch['y_true'].reshape(-1, 1).float()
            y_pred_i = model(
                TCRb=TCRb, epitope=epitope,
                TCRb_ohe=TCR_ohe, epitope_ohe=epitope_ohe,
                TCRb_mask=TCRb_mask, epitope_mask=epitope_mask,
            ).detach().cpu()
            y_pred.append(y_pred_i)
            y_true.append(y_true_i)
    return torch.concat(y_pred, 0), torch.concat(y_true, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--device', type=str, default='cuda:0', help='torch device')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_path', type=str, default='',
                        help='override save directory (default: <model_dir>/outputs/)')
    parser.add_argument('--save_filename', type=str, default='y_pred_test',
                        help='filename (without extension) for the saved predictions')
    parser.add_argument('--save_y_true', action='store_true',
                        help='also save the ground truth labels alongside predictions')
    args_raw = parser.parse_args()
    args, cfg_name = load_config(args_raw.config)

    print('Using config file: ', cfg_name)
    args_data = args.data
    args_model = args.model
    args_model.device = args_raw.device

    # prep gpu
    torch.cuda.empty_cache()
    gc.collect()
    torch.set_float32_matmul_precision('medium')
    set_seed(args_model)

    # load test split from the config's data_path
    ds_test = DeepSequencingRawSequenceDataset(args_data, 'test')
    print('Test size: ', len(ds_test))

    # wrap model into PL
    model = TCRPNet()
    if args_model.use_pos_weight:
        scores = torch.LongTensor(ds_test.data['Score'])
        num_positives = torch.sum(scores, dim=0)
        num_negatives = len(scores) - num_positives
        pos_weight = num_negatives / num_positives
        print('Using pos_weight...')
    else:
        pos_weight = None
    model = DiscriminativePL(args=args_model, model=model, pos_weight=pos_weight)

    collate_fn = BatchConverterCollater(alphabet=model.alphabet)
    dl_test = DataLoader(ds_test, batch_size=args_raw.batch_size, shuffle=False, collate_fn=collate_fn)

    # load checkpoint
    checkpoint_dir = os.path.join(
        args_model.tb_logger_path,
        args_model.tb_logger_folder,
        'lightning_logs',
        args_model.version_name,
        'checkpoints',
    )
    checkpoint_file = next(f for f in os.listdir(checkpoint_dir) if 'last' not in f)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(args_model.device).eval()

    # run inference
    y_pred, y_true = run_inference(model, dl_test, args_model)
    print(y_pred.shape)

    # save predictions to <model_dir>/outputs/ by default
    prediction_path = args_raw.save_path or os.path.join(
        args_model.tb_logger_path,
        args_model.tb_logger_folder,
        'lightning_logs',
        args_model.version_name,
        'outputs',
    )
    os.makedirs(prediction_path, exist_ok=True)
    print('Saving to: ', prediction_path)

    fname = args_raw.save_filename + '.npy'
    np.save(os.path.join(prediction_path, fname), y_pred)
    if args_raw.save_y_true:
        np.save(os.path.join(prediction_path, 'y_true_test.npy'), y_true)
