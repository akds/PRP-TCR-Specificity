import numpy as np
import esm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class DiscriminativePL(pl.LightningModule):
    """Lightning module for discriminative TCR-peptide interaction prediction.

    This module:
        1. Loads a pretrained ESM encoder.
        2. Extracts position-wise and pooled sequence representations.
        3. Concatenates ESM features with one-hot amino acid encodings.
        4. Passes the result through a downstream prediction head.

    Args:
        args: Namespace-like object containing training/model hyperparameters.
        model: Prediction head module that consumes encoded TCR/epitope features.
        y_test_true: Optional array of test labels for downstream evaluation.
        pos_weight: Optional positive-class weight for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        args,
        model: nn.Module,
        y_test_true=None,
        pos_weight=None,
    ) -> None:
        super().__init__()
        self.args = args
        self.model = model

        # Binary classification loss. Keep logits unnormalized here and apply
        # sigmoid externally only when needed for evaluation.
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.not_test_mode = True
        self.y_test_true = np.array(y_test_true)

        # Load pretrained ESM encoder and alphabet.
        self.encoder, self.alphabet = esm.pretrained.load_model_and_alphabet(
            args.esm_weights
        )
        self.encoder.output_hidden_states = True

        # Freeze or unfreeze the encoder depending on configuration.
        for _, param in self.encoder.named_parameters():
            param.requires_grad = not args.freeze_encoder

        # Store parameter trainability flags for debugging / inspection.
        self.layers_require_grad = [
            param.requires_grad for _, param in self.encoder.named_parameters()
        ]

    def forward(
        self,
        TCRb: torch.Tensor,
        epitope: torch.Tensor,
        TCRb_ohe: torch.Tensor,
        epitope_ohe: torch.Tensor,
        TCRb_mask: torch.Tensor,
        epitope_mask: torch.Tensor,
        repr_layer: int = 33,
    ) -> torch.Tensor:
        """Run a forward pass through the encoder and downstream model.

        Args:
            TCRb: Tokenized TCR beta sequences.
            epitope: Tokenized epitope sequences.
            TCRb_ohe: One-hot encoded TCR beta residues.
            epitope_ohe: One-hot encoded epitope residues.
            TCRb_mask: Boolean mask indicating valid TCR beta tokens.
            epitope_mask: Boolean mask indicating valid epitope tokens.
            repr_layer: ESM representation layer to extract.

        Returns:
            Predicted interaction logits of shape (B, 1).
        """
        # Extract position-wise ESM representations.
        TCRb_pw = self.encoder(
            TCRb,
            repr_layers=[repr_layer],
            return_contacts=False,
        )["representations"][repr_layer]
        epitope_pw = self.encoder(
            epitope,
            repr_layers=[repr_layer],
            return_contacts=False,
        )["representations"][repr_layer]

        # Compute pooled sequence embeddings using the valid-token masks.
        TCRb_single = TCRb_pw * TCRb_mask.unsqueeze(-1)
        TCRb_single = TCRb_single.sum(dim=1) / TCRb_mask.sum(dim=-1).unsqueeze(-1)

        epitope_single = epitope_pw * epitope_mask.unsqueeze(-1)
        epitope_single = epitope_single.sum(dim=1) / epitope_mask.sum(dim=-1).unsqueeze(-1)

        # Remove ESM special tokens (<cls>, <eos>) and concatenate one-hot features.
        TCRb_pw = torch.concat([TCRb_pw[:, 1:-1, :], TCRb_ohe], dim=-1)
        epitope_pw = torch.concat([epitope_pw[:, 1:-1, :], epitope_ohe], dim=-1)

        # Reverse masks because PyTorch MultiheadAttention interprets True as "ignore".
        out = self.model(
            TCRb_pw=TCRb_pw,
            epitope_pw=epitope_pw,
            TCRb_single=TCRb_single,
            epitope_single=epitope_single,
            TCRb_mask=~TCRb_mask[:, 1:-1],
            epitope_mask=~epitope_mask[:, 1:-1],
        )
        return out

    def general_step(self, batch: dict):
        """Shared forward logic for training and validation."""
        TCRb = batch["cdr"]
        epitope = batch["peptide"]
        TCRb_ohe = batch["cdr_ohe"]
        epitope_ohe = batch["peptide_ohe"]
        TCRb_mask = batch["cdr_mask"]
        epitope_mask = batch["peptide_mask"]
        y_true = batch["y_true"]

        y_pred = self(
            TCRb=TCRb,
            epitope=epitope,
            TCRb_ohe=TCRb_ohe,
            epitope_ohe=epitope_ohe,
            TCRb_mask=TCRb_mask,
            epitope_mask=epitope_mask,
        )
        return y_pred, y_true

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """Run one training step."""
        y_pred, y_true = self.general_step(batch)
        loss = self.criterion(y_pred, y_true)
        batch_size = y_pred.shape[0]

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Log current learning rate from the optimizer.
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log(
            "learning_rate",
            lr,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"loss": loss}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """Run one validation step."""
        y_pred, y_true = self.general_step(batch)
        loss = self.criterion(y_pred, y_true)
        batch_size = y_pred.shape[0]

        self.log(
            "valid_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return {"valid_loss": loss}

    def configure_optimizers(self) -> dict:
        """Configure the optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        return {"optimizer": optimizer}


def train_model(
    args,
    model: pl.LightningModule,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    test_dataloader: dict,
    strategy=None,
) -> pl.LightningModule:
    """Train a Lightning model and optionally run testing.

    Args:
        args: Namespace-like object containing trainer hyperparameters.
        model: Lightning module to train.
        train_dataloader: Training dataloader.
        valid_dataloader: Validation dataloader. Can be None.
        test_dataloader: Optional dictionary of named test dataloaders.
        strategy: Optional distributed strategy. Currently supports "ddp".

    Returns:
        The trained Lightning module.
    """
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=args.tb_logger_path + args.tb_logger_folder,
        version=args.version_name,
    )

    if valid_dataloader is None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=None,
            save_top_k=1,
            verbose=True,
            monitor="train_loss",
            mode="min",
            save_last=True,
        )
        callbacks = [checkpoint_callback]
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=None,
            save_top_k=1,
            verbose=True,
            monitor="valid_loss",
            mode="min",
            save_last=False,
        )
        early_stop_callback = EarlyStopping(
            monitor="valid_loss",
            min_delta=0.00,
            patience=args.patience,
            verbose=True,
            mode="min",
        )
        callbacks = [checkpoint_callback, early_stop_callback]

    # Configure trainer.
    if strategy == "ddp":
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            enable_progress_bar=True,
            enable_checkpointing=True,
            devices=args.device_num,
            accelerator="gpu",
            benchmark=True,
            callbacks=callbacks,
            logger=[tb_logger],
            log_every_n_steps=1,
            strategy="ddp_find_unused_parameters_true",
            precision=args.precision,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            enable_progress_bar=True,
            enable_checkpointing=True,
            devices=args.device_num,
            accelerator="gpu",
            benchmark=True,
            callbacks=callbacks,
            logger=[tb_logger],
            log_every_n_steps=1,
            precision=args.precision,
        )

    # Train model.
    trainer.fit(model, train_dataloader, valid_dataloader)

    # Evaluate on test sets, if provided.
    if test_dataloader is not None:
        for name, loader in test_dataloader.items():
            print("Test set:", name)
            trainer.model.y_test_true = loader.dataset.ys
            print(len(loader.dataset.ys))
            trainer.test(model, dataloaders=loader)

    return model
