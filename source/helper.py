from easydict import EasyDict
import argparse
import yaml
import os
import torch

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name

def set_seed(args: any):
    torch.manual_seed(args.seed)
    return


def one_hot_encode(sequence):
    # Define the amino acids and create a mapping
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
    # Convert the sequence to indices
    indices = [aa_to_idx[aa] for aa in sequence]
    # One-hot encode the indices
    one_hot_tensor = torch.nn.functional.one_hot(torch.tensor(indices), num_classes=len(amino_acids))
    return one_hot_tensor
    

def pad_dimension(tensor, max_length):
    """
    Pads the middle dimension of a 3D tensor with zeros to a specified max length.

    Args:
        tensor (torch.Tensor): A 3D tensor of shape (B, L, D), where:
            - B is the batch size.
            - L is the middle dimension to be padded.
            - D is the feature dimension.
        max_length (int): The target length for the middle dimension (L).

    Returns:
        torch.Tensor: A padded tensor of shape (B, max_length, D).
    """
    # Get the current shape of the tensor
    B, L, D = tensor.shape

    if L >= max_length:
        # If the middle dimension is already at or beyond max_length, truncate it
        return tensor[:, :max_length, :]
    
    # Create a padding tensor filled with zeros
    padding = torch.zeros((B, max_length - L, D), dtype=tensor.dtype, device=tensor.device)
    
    # Concatenate the original tensor and the padding tensor along the middle dimension
    padded_tensor = torch.cat((tensor, padding), dim=1)
    
    return padded_tensor