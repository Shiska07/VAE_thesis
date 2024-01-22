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
from layers.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)


_HTTPS_AWS_HUB = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com"


class VAE(LightningModule):
    """Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        vae = VAE()
        # pretrained on cifar10
        vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')
        # pretrained on stl10
        vae = VAE(input_height=32).from_pretrained('stl10-resnet18')
    """

    pretrained_urls = {
        "cifar10-resnet18": parse.urljoin(_HTTPS_AWS_HUB, "vae/vae-cifar10/checkpoints/epoch%3D89.ckpt"),
        "stl10-resnet18": parse.urljoin(_HTTPS_AWS_HUB, "vae/vae-stl10/checkpoints/epoch%3D89.ckpt"),
    }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--input_height",
            type = int,
            help = "Height of the images."
        )

        parser.add_argument(
            "--enc_type",
            type = str,
            default = "resnet18",
            help = "Either resnet18 or resnet50. [default: resnet18]"
        )

        parser.add_argument(
            "--first_conv",
            action = "store_true",
            help = "Use standard kernel_size 7, stride 2 at start or replace it with kernel_size 3, stride 1 conv. [default: If the flag is not passed --> False]"
        )

        parser.add_argument(
            "--maxpool1",
            action = "store_true",
            help = "Use standard maxpool to reduce spatial dim of feat by a factor of 2. [default: If the flag is not passed --> False]"
        )

        parser.add_argument(
            "--enc_out_dim",
            type = int,
            default = 512,
            help = "Set according to the out_channel count of encoder used (512 for resnet18, 2048 for resnet50, adjust for wider resnets). [default: 512]",
        )

        parser.add_argument(
            "--kl_coeff",
            type = float,
            default = 0.1,
            help = "Coefficient for kl term of the loss. [default: 0.1]"
        )

        parser.add_argument(
            "--latent_dim",
            type = int,
            default = 256,
            help = "Dim of latent space. [default: 256]"
        )

        parser.add_argument(
            "--lr",
            type = float,
            default = 1e-4,
            help = "Learning rate for Adam. [default: 1e-4]"
        )

        return parser


    def __init__(
        self,
        input_height,
        enc_type = "resnet18",
        first_conv = False,
        maxpool1 = False,
        enc_out_dim = 512,
        kl_coeff = 0.1,
        latent_dim = 256,
        lr = 1e-4,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super().__init__()

        # saving hparams to model state dict
        self.save_hyperparameters()

        # debugging
        self.example_input_array = torch.Tensor(1, 3, input_height, input_height)

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            "resnet18": {
                "enc": resnet18_encoder,
                "dec": resnet18_decoder,
            },
            "resnet50": {
                "enc": resnet50_encoder,
                "dec": resnet50_decoder,
            },
        }

        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1)
            self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)
        else:
            self.encoder = valid_encoders[enc_type]["enc"](first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]["dec"](self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

        self.time = datetime.now()
        self.val_outs = []
        self.test_outs = []


    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())


    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + " not present in pretrained weights.")
        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)


    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)


    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
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
        return loss, logs
    

    def training_step(self, batch, batch_idx):
        """
        #############################################################################
        ################################ PLACEHOLDER ################################
        #############################################################################

        This function should return the training loss of the batch.
        Here is what you have to implement (~ 3 lines of code):
            - read loss and logs by running self.step() on batch and batch_idx.
            - logging the training logs using self.log_dict() function. 
                logs is a dictionary. In order not to be mixed with logs from other steps, you have to add the string "train_"
                before all the keys in this dictionary. Make sure to call self.log_dict() with on_step=True, on_epoch=False, and
                sync_dist=True
            - return loss
        """
        loss, logs = self.step(batch, batch_idx)

        # modify train logs
        tr_logs = dict()
        for key, val in logs.items():
            new_key = "train_"+str(key)
            tr_logs[new_key] = val

        self.log_dict(tr_logs, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        #############################################################################
        ################################ PLACEHOLDER ################################
        #############################################################################

        This function should return the validation loss of the batch.
        Here is what you have to implement (~ 5 lines of code):
            - read loss and validation logs by running self.step() on batch and batch_idx.
            - logging the logs using self.log_dict() function. 
                logs is a dictionary. In order not to be mixed with logs from other steps, you have to add the string "val_"
                before all the keys in this dictionary. Make sure to call self.log_dict() with sync_dist=True
            - make sure for the 0th batch (batch_idx == 0), self.val_outs is set to batch
            - return loss
        """
        loss, logs = self.step(batch, batch_idx)

        val_logs = dict()
        for key, val in logs.items():
            new_key = "val_"+str(key)
            val_logs[new_key] = val
        self.log_dict(val_logs, on_step=True, on_epoch=False, sync_dist=True)

        if batch_idx == 0:
            self.val_outs = batch
        return loss

    def test_step(self, batch, batch_idx):
        """
        #############################################################################
        ################################ PLACEHOLDER ################################
        #############################################################################

        This function should return the test loss of the batch.
        Here is what you have to implement (~ 5 lines of code):
            - read loss and logs by running self.step() on batch and batch_idx.
            - logging the test logs using self.log_dict() function. 
                logs is a dictionary. In order not to be mixed with logs from other steps, you have to add the string "test_"
                before all the keys in this dictionary. Make sure to call self.log_dict() with on_step=True, on_epoch=False, and
                sync_dist=True
            - make sure for the 0th batch (batch_idx == 0), self.test_outs is set to batch
            - return loss
        """
        loss, logs = self.step(batch, batch_idx)

        test_logs = dict()
        for key, val in logs.items():
            new_key = "test_" + str(key)
            test_logs[new_key] = val
        self.log_dict(test_logs, on_step=True, on_epoch=False, sync_dist=True)

        if batch_idx == 0:
            self.test_outs = batch
        return loss


    def on_train_epoch_end(self):
            now = datetime.now()
            delta = now - self.time
            self.time = now
            tensorboard_logs = {'time_secs_epoch': delta.seconds}
            self.log_dict(tensorboard_logs, sync_dist=True)
    

    def on_validation_epoch_end(self):
        if self.global_rank == 0:
            val_dir = join(self.logger.save_dir, self.logger.name, f"version_{self.logger.version}", "validation_results")
            create_dir(val_dir)

            # Saving validation results.
            x, y = self.val_outs
            z, x_hat, p, q = self._run_step(x)

            # Loading inv_transformations
            self.inv_transformations_read_dir = self.logger.save_dir
            self.inv_transformations = load_transformation(join(self.inv_transformations_read_dir, "inv_trans.obj"))
            if self.inv_transformations is not None:
                x = self.inv_transformations(x)
                x_hat = self.inv_transformations(x_hat)

            if self.current_epoch == 0:
                grid = vutils.make_grid(x, nrow=8, normalize=False)
                vutils.save_image(x, join(val_dir, f"orig_{self.logger.name}_{self.current_epoch}.png"), normalize=False, nrow=8)
                self.logger.experiment.add_image(f"orig_{self.logger.name}_{self.current_epoch}", grid, self.global_step)

            grid = vutils.make_grid(x_hat, nrow=8, normalize=False)
            vutils.save_image(x_hat, join(val_dir, f"recons_{self.logger.name}_{self.current_epoch}.png"), normalize=False, nrow=8)
            self.logger.experiment.add_image(f"recons_{self.logger.name}_{self.current_epoch}", grid, self.global_step)


    def on_test_epoch_end(self):
        if self.global_rank == 0:
            test_dir = join(self.logger.save_dir, self.logger.name, f"version_{self.logger.version}", "test_results")
            create_dir(test_dir)

            # Saving test results.
            x, y = self.test_outs
            z, x_hat, p, q = self._run_step(x)


            # Loading inv_transformations
            self.inv_transformations_read_dir = self.logger.save_dir
            self.inv_transformations = load_transformation(join(self.inv_transformations_read_dir, "inv_trans.obj"))
            if self.inv_transformations is not None:
                x = self.inv_transformations(x)
                x_hat = self.inv_transformations(x_hat)

            grid = vutils.make_grid(x, nrow=8, normalize=False)
            vutils.save_image(x, join(test_dir, f"test_orig_{self.logger.name}_{self.current_epoch}.png"), normalize=False, nrow=8)
            self.logger.experiment.add_image(f"test_orig_{self.logger.name}_{self.current_epoch}", grid, self.global_step)

            grid = vutils.make_grid(x_hat, nrow=8, normalize=False)
            vutils.save_image(x_hat, join(test_dir, f"test_recons_{self.logger.name}_{self.current_epoch}.png"), normalize=False, nrow=8)
            self.logger.experiment.add_image(f"test_recons_{self.logger.name}_{self.current_epoch}", grid, self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)