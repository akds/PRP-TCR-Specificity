import torch
from torch.utils.data import Dataset, DataLoader

import esm

import pandas as pd
from tqdm import tqdm
import numpy as np
import ast

from helper import *

class DeepSequencingDataset(Dataset):
    def __init__(self, args, split='train', data=''):
        if len(data) != 0:
            self.data = data
        else:
            self.data = pd.read_csv(args.data_path)
        if split == 'all':
            pass
        else:
            self.data = self.data.loc[self.data['Split'] == split]
        self.data = self.data.loc[~self.data['Epitope'].str.contains('X')]
        self.data = self.data.reset_index(drop=True)
        
        self.cdrs_seq_rep = torch.load(args.cdr_seq_rep_file)
        self.cdrs_pw_rep = torch.load(args.cdr_pw_rep_file)
        self.peptides_seq_rep = torch.load(args.peptide_seq_rep_file)
        # self.peptides_seq_rep = {k:v for k,v in tqdm(self.peptides_seq_rep.items(), total=len(self.peptides_seq_rep)) if k in self.data['Epitope'].tolist()}
        self.peptides_pw_rep = torch.load(args.peptide_pw_rep_file)
        # self.peptides_pw_rep = {k:v for k,v in tqdm(self.peptides_pw_rep.items(), total=len(self.peptides_pw_rep)) if k in self.data['Epitope'].tolist()}
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        cdr = row['CDR3_b']
        score = row['Score']
        peptide = row['Epitope']
        cdr_seq_emb = self.cdrs_seq_rep[cdr]
        cdr_pw_emb = self.cdrs_pw_rep[cdr]
        peptide_seq_emb = self.peptides_seq_rep[peptide]
        peptide_pw_emb = self.peptides_pw_rep[peptide]
        return cdr_seq_emb, cdr_pw_emb, peptide_seq_emb, peptide_pw_emb, score


class DeepSequencingRawSequenceDataset(Dataset):
    def __init__(self, args, split='train', data=''):
        if len(data) != 0:
            self.data = data
        else:
            self.data = pd.read_csv(args.data_path)
        if split == 'all':
            pass
        else:
            self.data = self.data.loc[self.data['Split'] == split]
        self.data = self.data.loc[~self.data['Epitope'].str.contains('X')]
        self.data = self.data.reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        cdr = row['CDR3_b']
        score = row['Score']
        peptide = row['Epitope']
        cdr_ohe = one_hot_encode(cdr)
        peptide_ohe = one_hot_encode(peptide)
        return {'cdr': cdr, 
                'peptide': peptide, 
                'cdr_ohe': cdr_ohe,
                'peptide_ohe': peptide_ohe,
                'y_true': score,}

class VDJDBRawSequenceDataset(Dataset):
    def __init__(self, args, split='train', data=''):
        if len(data) != 0:
            self.data = data
        else:
            self.data = pd.read_csv(args.data_path)
        if split == 'all':
            pass
        else:
            self.data = self.data.loc[self.data['Split'] == split]
        self.data = self.data.reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        cdr = row['CDR3_b']
        score = np.random.choice([0,1], 1)[0]
        if score == 0: # negative
            peptide = np.random.choice(ast.literal_eval(row['NegEpitope']),1)[0]
        else: # positive
            peptide = np.random.choice(ast.literal_eval(row['Epitope']), 1)[0]
        cdr_ohe = one_hot_encode(cdr)
        peptide_ohe = one_hot_encode(peptide)
        return {'cdr': cdr, 
                'peptide': peptide, 
                'cdr_ohe': cdr_ohe,
                'peptide_ohe': peptide_ohe,
                'y_true': score,}


class BatchConverterCollater(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

    def __call__(self, batch):
        cdr, peptide, y_true = [i['cdr'] for i in batch], [i['peptide'] for i in batch], [i['y_true'] for i in batch]
        data_cdr = [(s, s) for s in cdr]
        data_peptide = [(s, s) for s in peptide]
        batch_labels_cdr, batch_strs_cdr, batch_tokens_cdr = self.batch_converter(data_cdr)
        batch_labels_peptide, batch_strs_peptide, batch_tokens_peptide = self.batch_converter(data_peptide)
        
        cdr_mask = ~((batch_tokens_cdr == 0) + (batch_tokens_cdr == 1) + (batch_tokens_cdr == 2))
        peptide_mask = ~((batch_tokens_peptide == 0) + (batch_tokens_peptide == 1) + (batch_tokens_peptide == 2))

        cdr_maxlen = max([len(i) for i in cdr])
        peptide_maxlen = max([len(i) for i in peptide])
        cdr_ohe, peptide_ohe = [i['cdr_ohe'] for i in batch], [i['peptide_ohe'] for i in batch]
        
        cdr_ohe = torch.concat([pad_dimension(i.unsqueeze(0), cdr_maxlen) for i in cdr_ohe],0) # might need to pad before stacking
        peptide_ohe = torch.concat([pad_dimension(i.unsqueeze(0), peptide_maxlen) for i in peptide_ohe],0)
        return {'cdr': batch_tokens_cdr,
                'peptide': batch_tokens_peptide, 
                'cdr_ohe': cdr_ohe,
                'peptide_ohe': peptide_ohe,
                'cdr_mask': cdr_mask,
                'peptide_mask': peptide_mask,
                'y_true': torch.FloatTensor(y_true).reshape(-1,1)}
    