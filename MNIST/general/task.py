# VAE implementation from pytorch_lightning bolts is used in this source code:
# https://github.com/Lightning-Universe/lightning-bolts/tree/master/pl_bolts/models/autoencoders

import argparse
from os.path import join
from urllib import parse
from datetime import datetime
from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.utils as vutils

from utils.os_tools import create_dir, load_transformation, save_latent_space
from layers.components import EncoderBlock, DecoderBlock

class PartProtoVAE(LightningModule):
    def __init__(
        self,
        input_height=28,
        input_channels = 1,
        enc_out_channels = 32,
        kl_coeff = 0.1,
        latent_channels = 16,
        lr = 1e-4,
        ksize = 3
    ):

        super().__init__()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.ksize = ksize
        self.enc_out_channels = enc_out_channels
        self.latent_channels = latent_channels
        self.input_height = input_height
        self.input_channels = input_channels

        self.encoder = EncoderBlock(self.input_channels, self.latent_channels, self.ksize)
        self.decoder = DecoderBlock(self.latent_channels, self.input_channels, self.ksize)

        # lists to store loses from each step
        self.training_step_losses = []
        self.validation_step_losses = []
        self.test_step_losses = []

        # lists to store reconstruction images
        self.val_outs = []
        self.test_outs = []

        # dict to store epoch loss
        self.train_history = {'train_rec_loss': [],  'train_kl_loss':[], 'train_total_loss':[], 'train_acc': []}
        self.val_history = {'val_rec_loss': [], 'val_kl_loss': [], 'val_total_loss':[], 'val_acc': []}
        self.test_history = {'test_rec_loss': [], 'test_kl_loss': [], 'test_total_loss':[], 'test_acc': []}

    def forward(self, x):
        mu, logvar = self.encoder(x)
        p, q, z = self.sample(mu, logvar)
        return self.decoder(z)

    def _run_step(self, x):
        mu, logvar = self.encoder(x)
        p, q, z = self.sample(mu, logvar)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs, x_hat
    

    def training_step(self, batch, batch_idx):

        loss, logs , _ = self.step(batch, batch_idx)

        # modify train logs
        tr_logs = dict()
        for key, val in logs.items():
            new_key = "train_"+str(key)
            tr_logs[new_key] = val

        self.training_step_losses.append(tuple(tr_logs.values()))
        return loss

    def validation_step(self, batch, batch_idx):

        loss, logs, x_hat = self.step(batch, batch_idx)

        val_logs = dict()
        for key, val in logs.items():
            new_key = "val_"+str(key)
            val_logs[new_key] = val

        # store loss as well as reconstructed images
        self.validation_step_losses.append(tuple(val_logs.values()))
        return loss

    def test_step(self, batch, batch_idx):

        loss, logs = self.step(batch, batch_idx)

        test_logs = dict()
        for key, val in logs.items():
            new_key = "test_" + str(key)
            test_logs[new_key] = val
        self.test_step_losses.append(tuple(test_logs.values()))
        return loss

    def get_cumulative_losses(self, losses_list):
        num_items = len(losses_list)
        cum_rec_loss = 0
        cum_kl_loss = 0
        cum_total_loss = 0

        for rec_loss, kl_loss, total_loss in losses_list:
            cum_rec_loss += rec_loss.item()
            cum_kl_loss += kl_loss.item()
            cum_total_loss += total_loss.item()

        # get average loss
        cum_rec_loss += cum_rec_loss / num_items
        cum_kl_loss += cum_kl_loss / num_items
        cum_total_loss += cum_total_loss / num_items

        return cum_rec_loss, cum_kl_loss, cum_total_loss


    def on_train_epoch_end(self):

        cum_rec_loss, cum_kl_loss, cum_total_loss = self.get_cumulative_losses(self.training_step_losses)
        print(f'\nTraining Epoch({self.current_epoch}): rec_loss: {cum_rec_loss}, kl_loss:{cum_kl_loss}, total_loss:{cum_total_loss}')

        # store epoch loss
        self.train_history['train_rec_loss'].append(cum_rec_loss)
        self.train_history['train_kl_loss'].append(cum_kl_loss)
        self.train_history['train_total_loss'].append(cum_total_loss)
        self.training_step_losses.clear()
        

    def on_validation_epoch_end(self):
        cum_rec_loss, cum_kl_loss, cum_total_loss = self.get_cumulative_losses(self.validation_step_losses)
        print(f'\nValidation Epoch({self.current_epoch}): rec_loss: {cum_rec_loss}, kl_loss:{cum_kl_loss}, total_loss:{cum_total_loss}')

        # store epoch loss
        self.val_history['val_rec_loss'].append(cum_rec_loss)
        self.val_history['val_kl_loss'].append(cum_kl_loss)
        self.val_history['val_total_loss'].append(cum_total_loss)
        self.validation_step_losses.clear()
        self.log("val_loss", cum_total_loss)


    def on_test_epoch_end(self):
        cum_rec_loss, cum_kl_loss, cum_total_loss = self.get_cumulative_losses(self.test_step_losses)
        print(f'\nTest Epoch({self.current_epoch}): rec_loss: {cum_rec_loss}, kl_loss:{cum_kl_loss}, total_loss:{cum_total_loss}')

        # store epoch loss
        self.val_history['val_rec_loss'].append(cum_rec_loss)
        self.val_history['val_kl_loss'].append(cum_kl_loss)
        self.val_history['val_total_loss'].append(cum_total_loss)
        self.test_step_losses.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)