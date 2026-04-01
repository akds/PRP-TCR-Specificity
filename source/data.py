import ast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from helper import one_hot_encode, pad_dimension


class DeepSequencingDataset(Dataset):
    """Dataset with precomputed TCR CDR3 and peptide embeddings.

    Each item returns:
        - cdr_seq_emb: sequence-level CDR embedding
        - cdr_pw_emb: position-wise CDR embedding
        - peptide_seq_emb: sequence-level peptide embedding
        - peptide_pw_emb: position-wise peptide embedding
        - score: target interaction score

    Args:
        args: Namespace-like object containing:
            - data_path
            - cdr_seq_rep_file
            - cdr_pw_rep_file
            - peptide_seq_rep_file
            - peptide_pw_rep_file
        split: One of {"train", "val", "test", "all"}.
        data: Optional preloaded dataframe. If provided, args.data_path is ignored.
    """

    def __init__(self, args, split: str = "train", data="") -> None:
        if len(data) != 0:
            self.data = data
        else:
            self.data = pd.read_csv(args.data_path)

        if split != "all":
            self.data = self.data.loc[self.data["Split"] == split]

        # Remove peptides containing unknown amino acid tokens.
        self.data = self.data.loc[~self.data["Epitope"].str.contains("X")]
        self.data = self.data.reset_index(drop=True)

        # Load precomputed representations.
        self.cdrs_seq_rep = torch.load(args.cdr_seq_rep_file)
        self.cdrs_pw_rep = torch.load(args.cdr_pw_rep_file)
        self.peptides_seq_rep = torch.load(args.peptide_seq_rep_file)
        self.peptides_pw_rep = torch.load(args.peptide_pw_rep_file)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Return one sample from the dataset."""
        row = self.data.iloc[idx]

        cdr = row["CDR3_b"]
        peptide = row["Epitope"]
        score = row["Score"]

        cdr_seq_emb = self.cdrs_seq_rep[cdr]
        cdr_pw_emb = self.cdrs_pw_rep[cdr]
        peptide_seq_emb = self.peptides_seq_rep[peptide]
        peptide_pw_emb = self.peptides_pw_rep[peptide]

        return cdr_seq_emb, cdr_pw_emb, peptide_seq_emb, peptide_pw_emb, score


class DeepSequencingRawSequenceDataset(Dataset):
    """Dataset returning raw CDR3/peptide sequences and one-hot encodings.

    Each item returns a dictionary containing:
        - cdr
        - peptide
        - cdr_ohe
        - peptide_ohe
        - y_true

    Args:
        args: Namespace-like object containing data_path.
        split: One of {"train", "val", "test", "all"}.
        data: Optional preloaded dataframe. If provided, args.data_path is ignored.
    """

    def __init__(self, args, split: str = "train", data="") -> None:
        if len(data) != 0:
            self.data = data
        else:
            self.data = pd.read_csv(args.data_path)

        if split != "all":
            self.data = self.data.loc[self.data["Split"] == split]

        # Remove peptides containing unknown amino acid tokens.
        self.data = self.data.loc[~self.data["Epitope"].str.contains("X")]
        self.data = self.data.reset_index(drop=True)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return one sample with raw sequence and one-hot features."""
        row = self.data.iloc[idx]

        cdr = row["CDR3_b"]
        peptide = row["Epitope"]
        score = row["Score"]

        cdr_ohe = one_hot_encode(cdr)
        peptide_ohe = one_hot_encode(peptide)

        return {
            "cdr": cdr,
            "peptide": peptide,
            "cdr_ohe": cdr_ohe,
            "peptide_ohe": peptide_ohe,
            "y_true": score,
        }


class VDJDBRawSequenceDataset(Dataset):
    """VDJdb dataset with on-the-fly positive/negative peptide sampling.

    For each CDR, a label is sampled uniformly:
        - 0: sample one peptide from the negative epitope list
        - 1: sample one peptide from the positive epitope list

    Each item returns a dictionary containing:
        - cdr
        - peptide
        - cdr_ohe
        - peptide_ohe
        - y_true

    Args:
        args: Namespace-like object containing data_path.
        split: One of {"train", "val", "test", "all"}.
        data: Optional preloaded dataframe. If provided, args.data_path is ignored.
    """

    def __init__(self, args, split: str = "train", data="") -> None:
        if len(data) != 0:
            self.data = data
        else:
            self.data = pd.read_csv(args.data_path)

        if split != "all":
            self.data = self.data.loc[self.data["Split"] == split]

        self.data = self.data.reset_index(drop=True)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return one sampled positive or negative CDR-peptide pair."""
        row = self.data.iloc[idx]

        cdr = row["CDR3_b"]

        # Randomly sample a binary label.
        score = np.random.choice([0, 1], 1)[0]

        if score == 0:
            # Negative peptide sampled from NegEpitope list.
            peptide = np.random.choice(ast.literal_eval(row["NegEpitope"]), 1)[0]
        else:
            # Positive peptide sampled from Epitope list.
            peptide = np.random.choice(ast.literal_eval(row["Epitope"]), 1)[0]

        cdr_ohe = one_hot_encode(cdr)
        peptide_ohe = one_hot_encode(peptide)

        return {
            "cdr": cdr,
            "peptide": peptide,
            "cdr_ohe": cdr_ohe,
            "peptide_ohe": peptide_ohe,
            "y_true": score,
        }


class BatchConverterCollater:
    """Collate function for batching raw TCR/peptide sequences for ESM models.

    This collater:
        1. Tokenizes CDR and peptide sequences with an ESM alphabet batch converter.
        2. Builds boolean masks that exclude special tokens.
        3. Pads one-hot encodings to the longest sequence in the batch.
        4. Returns a batch dictionary for downstream models.

    Args:
        alphabet: ESM alphabet object with `get_batch_converter()`.
    """

    def __init__(self, alphabet) -> None:
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

    def __call__(self, batch: list[dict]) -> dict:
        """Convert a list of examples into a padded, tokenized batch."""
        cdr = [item["cdr"] for item in batch]
        peptide = [item["peptide"] for item in batch]
        y_true = [item["y_true"] for item in batch]

        # ESM batch converter expects (label, sequence) tuples.
        data_cdr = [(seq, seq) for seq in cdr]
        data_peptide = [(seq, seq) for seq in peptide]

        _, _, batch_tokens_cdr = self.batch_converter(data_cdr)
        _, _, batch_tokens_peptide = self.batch_converter(data_peptide)

        # Mask out special tokens:
        # 0 = padding, 1/2 = ESM special tokens depending on alphabet.
        cdr_mask = ~(
            (batch_tokens_cdr == 0)
            + (batch_tokens_cdr == 1)
            + (batch_tokens_cdr == 2)
        )
        peptide_mask = ~(
            (batch_tokens_peptide == 0)
            + (batch_tokens_peptide == 1)
            + (batch_tokens_peptide == 2)
        )

        cdr_maxlen = max(len(seq) for seq in cdr)
        peptide_maxlen = max(len(seq) for seq in peptide)

        cdr_ohe = [item["cdr_ohe"] for item in batch]
        peptide_ohe = [item["peptide_ohe"] for item in batch]

        # Pad one-hot encodings to the maximum sequence length in the batch.
        cdr_ohe = torch.concat(
            [pad_dimension(tensor.unsqueeze(0), cdr_maxlen) for tensor in cdr_ohe],
            dim=0,
        )
        peptide_ohe = torch.concat(
            [pad_dimension(tensor.unsqueeze(0), peptide_maxlen) for tensor in peptide_ohe],
            dim=0,
        )

        return {
            "cdr": batch_tokens_cdr,
            "peptide": batch_tokens_peptide,
            "cdr_ohe": cdr_ohe,
            "peptide_ohe": peptide_ohe,
            "cdr_mask": cdr_mask,
            "peptide_mask": peptide_mask,
            "y_true": torch.FloatTensor(y_true).reshape(-1, 1),
        }
