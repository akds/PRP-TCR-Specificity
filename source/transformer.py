import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Adds fixed sinusoidal positional encodings to input embeddings.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum supported sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to the input.

        Args:
            x: Input tensor of shape (B, L, D).

        Returns:
            Tensor of shape (B, L, D) with positional encoding added.
        """
        return x + self.pe[:, : x.size(1), :].to(x.device)


class MultiHeadAttention(nn.Module):
    """Wrapper around PyTorch multi-head self-attention."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply multi-head attention.

        Args:
            query: Query tensor of shape (B, L, D).
            key: Key tensor of shape (B, L, D).
            value: Value tensor of shape (B, L, D).
            mask: Optional key padding mask where True indicates positions to ignore.

        Returns:
            Attention output of shape (B, L, D).
        """
        attn_output, _ = self.attn(
            query,
            key,
            value,
            key_padding_mask=mask,
        )
        return attn_output


class FeedForward(nn.Module):
    """Position-wise feedforward block."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feedforward transformation."""
        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    """Single transformer encoder block.

    Each block contains:
        - multi-head self-attention
        - feedforward network
        - residual connections
        - layer normalization
        - dropout
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one encoder layer."""
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder followed by projection and sequence pooling.

    Behavior:
        - Adds positional encodings
        - Applies a stack of encoder layers
        - Projects features from d_model to d_out
        - Pools across sequence length

    If a mask is provided:
        - True values are treated as padded/ignored positions
        - pooling is performed only over unmasked positions

    If no mask is provided:
        - simple mean pooling is used across sequence positions

    Args:
        d_model: Input embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of encoder layers.
        d_ff: Hidden dimension in the feedforward block.
        dropout: Dropout probability.
        max_len: Maximum sequence length for positional encoding.
        d_out: Output feature dimension after projection.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        max_len: int = 1024,
        d_out: int = 1280,
    ) -> None:
        super().__init__()

        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(d_model, d_out)

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode and pool a sequence input.

        Args:
            src: Input tensor of shape (B, L, D).
            mask: Optional boolean mask of shape (B, L), where True indicates
                positions to ignore.

        Returns:
            Tensor of shape (B, d_out) after masked or mean pooling.
        """
        src = self.positional_encoding(src)
        src = self.dropout(src)

        for layer in self.encoder_layers:
            src = layer(src, mask)

        src = self.proj_out(src)

        if mask is not None:
            valid_mask = (~mask).unsqueeze(-1)
            src = src * valid_mask
            src = src.sum(dim=1) / (~mask).sum(dim=-1).unsqueeze(-1)
        else:
            src = torch.mean(src, dim=1)

        return src


if __name__ == "__main__":
    d_model = 1280
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    max_len = 100

    encoder = TransformerEncoder(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
    )

    # Example input: (batch_size, sequence_length, embedding_dim)
    src = torch.randn(32, 20, 1280)

    output = encoder(src)
    print(output.size())
