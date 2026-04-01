import torch
import torch.nn as nn
from einops import einsum

import transformer


class AttnVector(nn.Module):
    """Learnable attention vector with values constrained to [0, 1].

    This module stores an unconstrained parameter and applies a sigmoid in
    the forward pass so the returned weights lie between 0 and 1.
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.unconstrained_param = nn.Parameter(torch.zeros(size))

    def forward(self) -> torch.Tensor:
        """Return the constrained attention weights."""
        return torch.sigmoid(self.unconstrained_param)


class TCRPNet(nn.Module):
    """TCR beta / peptide interaction prediction network.

    The model:
        1. Encodes position-wise TCR beta and epitope features with separate
           transformer encoders.
        2. Builds pairwise outer-product representations from:
            - pooled single-sequence embeddings
            - encoded position-wise representations
        3. Concatenates both pair representations as 2 input channels.
        4. Uses a small 2D CNN + MLP head to predict an interaction score.

    Args:
        use_attn: If True, replaces the provided single embeddings with
            weighted averages computed from the position-wise embeddings.
    """

    def __init__(self, use_attn: bool = False) -> None:
        super().__init__()

        self.use_attn = use_attn

        if self.use_attn:
            # Learnable per-position weights for CDR and peptide residues.
            self.attn_cdr = AttnVector(15)
            self.attn_peptide = AttnVector(9)

        # Convolution / pooling hyperparameters.
        self.conv_kernel = (5, 5)
        self.pool_kernel = (5, 5)

        # Model dimensions.
        d_model = 1280 + 20  # ESM embedding + one-hot amino acid features
        num_heads = 10
        num_layers = 6
        d_ff = 2048
        dropout = 0.1
        max_len = 100

        # Separate encoders for TCR beta and epitope.
        self.encoder_tcrb = transformer.TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )
        self.encoder_epitope = transformer.TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )

        # CNN channel sizes.
        self.h1 = 32
        self.h2 = 64
        self.h3 = 128
        self.input_channels = 2  # single_pair and encoded_pair

        # 2D CNN over pairwise interaction maps.
        self.conv_net = nn.Sequential(
            nn.Conv2d(self.input_channels, self.h1, kernel_size=self.conv_kernel),
            nn.MaxPool2d(kernel_size=self.pool_kernel),
            nn.ReLU(),
            nn.InstanceNorm2d(self.h1),

            nn.Conv2d(self.h1, self.h2, kernel_size=self.conv_kernel),
            nn.MaxPool2d(kernel_size=self.pool_kernel),
            nn.ReLU(),
            nn.InstanceNorm2d(self.h2),

            nn.Conv2d(self.h2, self.h3, kernel_size=self.conv_kernel),
            nn.MaxPool2d(kernel_size=self.pool_kernel),
            nn.ReLU(),
            nn.InstanceNorm2d(self.h3),
        )

        # NOTE:
        # This linear input size assumes the conv output spatial size is 9 x 9.
        # If the upstream representation sizes or convolution settings change,
        # this value may need to be updated.
        self.classifier = nn.Sequential(
            nn.Linear(9 * 9 * self.h3, self.h3),
            nn.ReLU(inplace=True),
            nn.Linear(self.h3, 1),
            # nn.Sigmoid()  # Keep commented if using BCEWithLogitsLoss externally.
        )

    def forward(
        self,
        TCRb_pw: torch.Tensor,
        epitope_pw: torch.Tensor,
        TCRb_single: torch.Tensor,
        epitope_single: torch.Tensor,
        TCRb_mask: torch.Tensor | None = None,
        epitope_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a forward pass.

        Args:
            TCRb_pw: Position-wise TCR beta features, shape (B, L_tcr, D + 20).
            epitope_pw: Position-wise epitope features, shape (B, L_epi, D + 20).
            TCRb_single: Pooled TCR beta features, shape (B, D).
            epitope_single: Pooled epitope features, shape (B, D).
            TCRb_mask: Optional attention mask for TCR beta encoder.
            epitope_mask: Optional attention mask for epitope encoder.

        Returns:
            Predicted interaction scores of shape (B, 1).
        """
        if self.use_attn:
            # Recompute pooled single-sequence embeddings from the position-wise
            # embeddings by applying learned per-position weights.
            # The final 20 channels are assumed to be one-hot AA features and are excluded.
            TCRb_single = torch.clone(TCRb_pw[:, :, :-20])
            _, tcr_len, _ = TCRb_single.shape
            TCRb_single = TCRb_single * self.attn_cdr().unsqueeze(0).unsqueeze(-1)
            TCRb_single = TCRb_single.sum(dim=1) / tcr_len

            epitope_single = torch.clone(epitope_pw[:, :, :-20])
            _, epitope_len, _ = epitope_single.shape
            epitope_single = epitope_single * self.attn_peptide().unsqueeze(0).unsqueeze(-1)
            epitope_single = epitope_single.sum(dim=1) / epitope_len

        # Encode position-wise representations.
        TCRb_encoded = self.encoder_tcrb(TCRb_pw, mask=TCRb_mask)
        epitope_encoded = self.encoder_epitope(epitope_pw, mask=epitope_mask)

        # Build pair representations from:
        #   1) pooled single embeddings
        #   2) encoded position-wise embeddings
        outers_single = []
        outers_encoded = []

        for i in range(TCRb_single.size(0)):
            outer = einsum(
                TCRb_single[i, :].unsqueeze(0),
                epitope_single[i, :].unsqueeze(0),
                "a b, c d -> a b d",
            ).unsqueeze(1)
            outers_single.append(outer)

        for i in range(TCRb_encoded.size(0)):
            outer = einsum(
                TCRb_encoded[i, :].unsqueeze(0),
                epitope_encoded[i, :].unsqueeze(0),
                "a b, c d -> a b d",
            ).unsqueeze(1)
            outers_encoded.append(outer)

        single_pair = torch.cat(outers_single, dim=0)
        encoded_pair = torch.cat(outers_encoded, dim=0)

        # Stack both interaction maps as channels and predict a score.
        out = torch.cat([single_pair, encoded_pair], dim=1)
        out = self.conv_net(out)
        out = out.view(out.size(0), out.size(-1) * out.size(-2) * self.h3)
        out = self.classifier(out)

        return out


if __name__ == "__main__":
    model = TCRPNet()

    TCRb_pw = torch.randn((8, 20, 1300))
    epitope_pw = torch.randn((8, 9, 1300))

    TCRb_single = torch.randn((8, 1280))
    epitope_single = torch.randn((8, 1280))

    out = model(
        TCRb_pw=TCRb_pw,
        epitope_pw=epitope_pw,
        TCRb_single=TCRb_single,
        epitope_single=epitope_single,
    )

    print(out.shape)

    # Example dataloader usage:
    #
    # import data_loader
    # from torch.utils import data as D
    #
    # dataset = data_loader.data_loader(partition="train")
    # loader = D.DataLoader(dataset, batch_size=8, num_workers=20, shuffle=True)
    #
    # for TCRb, epitope, label, _, _ in loader:
    #     out = model(TCRb, epitope)
    #     print(out.size())
    #     break
