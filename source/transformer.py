import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        # pe[:x.size(1), :]: positional encoding based on sequence_length
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, query, key, value, mask=None):
        attn_output, _ = self.attn(query, key, value, key_padding_mask=mask)
        return attn_output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1, max_len=1024, d_out = 1280):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.proj_out = nn.Linear(d_model, d_out)

    def forward(self, src, mask=None):
        # src shape: (batch_size, sequence_length, d_model)
        src = self.positional_encoding(src)

        for layer in self.encoder_layers:
            src = layer(src, mask)
        # print(src.shape)
        # print(src.max(-1)[1])
        if mask != None:
            src = src * (~mask).unsqueeze(-1)
            # print(src.shape)
            # print(src.max(-1)[1])
            src = self.proj_out(src)
            src = src * (~mask).unsqueeze(-1)
            # print(src.shape)
            # print(src.max(-1)[1])
            src = src.sum(1)/(~mask).sum(-1).unsqueeze(-1)
            # print(src.sum(1).shape)
            # print(~mask)
            # print((~mask).sum(-1))
        else:
            src = self.proj_out(src)
            src = torch.mean(src, dim = 1)
        return src

# Example usage
if __name__ == "__main__":
    d_model = 1280  # This should match the embedding size of your input
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    max_len = 100

    encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropout, max_len)

    # Example input: (batch_size,sequence_length, embedding_dim)
    src = torch.randn(32, 20, 1280)

    output = encoder(src)
    print(output.size())
