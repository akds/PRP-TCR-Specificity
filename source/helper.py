import os

import torch
import yaml
from easydict import EasyDict


def load_config(config_path: str):
    """Load a YAML config file and return both the config and its stem name.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        A tuple of:
            - config: configuration dictionary wrapped in EasyDict
            - config_name: filename stem without extension
    """
    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))

    basename = os.path.basename(config_path)
    config_name = basename[: basename.rfind(".")]

    return config, config_name


def set_seed(args) -> None:
    """Set the PyTorch random seed.

    Args:
        args: Namespace-like object with a `seed` attribute.
    """
    torch.manual_seed(args.seed)


def one_hot_encode(sequence: str) -> torch.Tensor:
    """One-hot encode an amino acid sequence.

    Amino acids are encoded using the canonical 20-residue alphabet:
        A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y

    Args:
        sequence: Amino acid sequence.

    Returns:
        Tensor of shape (L, 20), where L is sequence length.

    Raises:
        KeyError: If the sequence contains a residue outside the supported alphabet.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}

    indices = [aa_to_idx[aa] for aa in sequence]
    one_hot_tensor = torch.nn.functional.one_hot(
        torch.tensor(indices), num_classes=len(amino_acids)
    )

    return one_hot_tensor


def pad_dimension(tensor: torch.Tensor, max_length: int) -> torch.Tensor:
    """Pad or truncate the sequence dimension of a 3D tensor.

    Expects an input tensor of shape (B, L, D), where:
        - B is batch size
        - L is sequence length
        - D is feature dimension

    If L < max_length, the tensor is padded with zeros along dimension 1.
    If L >= max_length, the tensor is truncated to length max_length.

    Args:
        tensor: Input tensor of shape (B, L, D).
        max_length: Target sequence length.

    Returns:
        Tensor of shape (B, max_length, D).
    """
    batch_size, seq_len, feature_dim = tensor.shape

    if seq_len >= max_length:
        return tensor[:, :max_length, :]

    padding = torch.zeros(
        (batch_size, max_length - seq_len, feature_dim),
        dtype=tensor.dtype,
        device=tensor.device,
    )

    padded_tensor = torch.cat((tensor, padding), dim=1)
    return padded_tensor
