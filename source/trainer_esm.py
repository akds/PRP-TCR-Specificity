import numpy as np
import esm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class AttnVector(nn.Module):
    def __init__(self, size):
        super(AttnVector, self).__init__()
        # Define the unconstrained parameter
        self.unconstrained_param = nn.Parameter(torch.zeros(size))

    def forward(self):
        # Apply the sigmoid function to constrain the parameter between 0 and 1
        return torch.sigmoid(self.unconstrained_param)


class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, min_lr, max_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linearly scale the learning rate
            return [
                self.min_lr + (self.max_lr - self.min_lr) * (self.last_epoch / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # After warmup, return the max learning rate
            return [self.max_lr for base_lr in self.base_lrs]


class DiscriminativePL(pl.LightningModule):
    def __init__(
            self,
            args: any,
            model: nn.Module, 
            y_test_true=None,
            valid_counts_datalader=None,
            pos_weight=None
            ):
        super().__init__()
        self.args = args
        self.model = model
        # self.criterion = nn.BCELoss()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.not_test_mode = True # relevant for only training and validation forward steps
        self.y_test_true = np.array(y_test_true)
        
        self.encoder, self.alphabet = \
            esm.pretrained.load_model_and_alphabet(args.esm_weights)
        self.encoder.output_hidden_states = True

        encoder_params_dict = dict(self.encoder.named_parameters())
        for name, param in self.encoder.named_parameters():
            if args.freeze_encoder:
                param.requires_grad = False
            else:
                param.requires_grad = True
        self.layers_require_grad = [] # log for checking
        for name, param in self.encoder.named_parameters():
            self.layers_require_grad.append(param.requires_grad)
            
            
    def forward(self, TCRb, epitope, TCRb_ohe, epitope_ohe, TCRb_mask, epitope_mask, repr_layer=33):
        # get encoder layers
        TCRb_pw = self.encoder(TCRb, repr_layers=[repr_layer], return_contacts=False)['representations'][repr_layer]
        epitope_pw = self.encoder(epitope, repr_layers=[repr_layer], return_contacts=False)['representations'][repr_layer]

        # single seq features
        TCRb_single = TCRb_pw * TCRb_mask.unsqueeze(-1)
        TCRb_single = TCRb_single.sum(1)/TCRb_mask.sum(-1).unsqueeze(-1)
        epitope_single = epitope_pw * epitope_mask.unsqueeze(-1)
        epitope_single = epitope_single.sum(1)/epitope_mask.sum(-1).unsqueeze(-1)

        # position seq features
        TCRb_pw = torch.concat([TCRb_pw[:,1:-1,:], TCRb_ohe],-1)
        epitope_pw = torch.concat([epitope_pw[:,1:-1,:], epitope_ohe],-1)

        # run through head
        out = self.model(TCRb_pw=TCRb_pw, epitope_pw=epitope_pw, 
                         TCRb_single=TCRb_single, epitope_single=epitope_single, 
                         TCRb_mask=~TCRb_mask[:,1:-1], epitope_mask=~epitope_mask[:,1:-1]) # reverse mask because multiheadattn True == ignore, chop off start and end tokens
        return out

    def general_step(self, batch):
        TCRb = batch['cdr']
        epitope = batch['peptide']
        TCR_ohe = batch['cdr_ohe']
        epitope_ohe = batch['peptide_ohe']
        TCRb_mask = batch['cdr_mask']
        epitope_mask = batch['peptide_mask']
        
        y_true = batch['y_true']
        y_pred = self(TCRb=TCRb, epitope=epitope,
                      TCRb_ohe=TCR_ohe, epitope_ohe=epitope_ohe,
                      TCRb_mask=TCRb_mask, epitope_mask=epitope_mask)
        return y_pred, y_true
        
    def training_step(self, batch: dict, batch_idx: list) -> dict:
        y_pred, y_true = self.general_step(batch)
        loss = self.criterion(y_pred, y_true)
        batch_size = y_pred.shape[0]
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)

        # Get optimizer (assuming you have only one optimizer)
        opt = self.optimizers()
        # Retrieve the learning rate from the optimizer's param_groups
        lr = opt.param_groups[0]['lr']
        # Log the learning rate if you want to monitor it in the logs
        self.log('learning_rate', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}


    def validation_step(self, batch: dict, batch_idx: list) -> dict:
        y_pred, y_true = self.general_step(batch)
        loss = self.criterion(y_pred, y_true)
        batch_size = y_pred.shape[0]
        self.log('valid_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return {'valid_loss': loss}

   
    def configure_optimizers(self,) -> dict:
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.args.lr, 
                                      weight_decay=self.args.weight_decay)
        return {
                'optimizer': optimizer
        }


def train_model(
        args: any,
        model: pl.LightningModule,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        test_dataloader: dict,
        strategy=None,
    ) -> pl.LightningModule:

    tb_logger = pl.loggers.TensorBoardLogger(
            args.tb_logger_path + args.tb_logger_folder, version=args.version_name
    )

    if valid_dataloader is None:
        checkpoint_callback = ModelCheckpoint(
                dirpath=None,
                save_top_k=1,
                verbose=True,
                monitor='train_loss',
                mode='min',
                save_last=True, # save the last model
        )
        callbacks = [checkpoint_callback]
    else:
        checkpoint_callback = ModelCheckpoint(
                dirpath=None,
                save_top_k=1,
                verbose=True,
                monitor='valid_loss',
                mode='min',
                save_last=False, # save the last model
        )
        early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, 
                                        patience=args.patience, verbose=True, mode="min")
        callbacks = [checkpoint_callback, early_stop_callback]
        
    # setup trainer
    if strategy == 'ddp':
        trainer = pl.Trainer(
                max_epochs=args.epochs,
                enable_progress_bar=True,
                enable_checkpointing=True,
                devices=args.device_num,
                accelerator='gpu',
                benchmark=True,
                callbacks=callbacks,
                logger=[tb_logger],
                log_every_n_steps=1,
                strategy='ddp_find_unused_parameters_true',
                precision=args.precision,
        )
    else:
        trainer = pl.Trainer(
                max_epochs=args.epochs,
                enable_progress_bar=True,
                enable_checkpointing=True,
                devices=args.device_num,
                accelerator='gpu',
                benchmark=True,
                callbacks=callbacks,
                logger=[tb_logger],
                log_every_n_steps=1,
                # strategy='ddp')
                precision=args.precision,
        )

    # training model
    trainer.fit(model, train_dataloader, valid_dataloader)

    # test dataset
    if test_dataloader is not None:
        for k,v in test_dataloader.items():
            print('Test set: ', k)
            trainer.model.y_test_true = v.dataset.ys
            print(len(v.dataset.ys))
            trainer.test(model, dataloaders=v)
    return model
