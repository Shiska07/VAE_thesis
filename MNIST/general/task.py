# VAE implementation from pytorch_lightning bolts is used in this source code:
# https://github.com/Lightning-Universe/lightning-bolts/tree/master/pl_bolts/models/autoencoders
import json
import os
from os.path import join

import torch
import torchsummary
import torchvision.utils as vutils
from pytorch_lightning import LightningModule
from torch.nn import functional as F

from general.settings import prototype_shape, num_classes
from utils import create_dir
from components import EncoderBlock, DecoderBlock


class PartProtoVAE(LightningModule):
    def __init__(
        self,
        input_height,
        input_channels,
        ce_coeff,
        kl_coeff,
        recon_coeff,
        clst_coeff,
        sep_coeff,
        lr = 1e-4,

    ):

        super().__init__()

        # saving hparams to model state dict
        self.save_hyperparameters()


        self.input_height = input_height
        self.input_channels = input_channels
        self.ce_coeff=ce_coeff
        self.kl_coeff = kl_coeff
        self.recon_coeff = recon_coeff
        self.clst_coeff = clst_coeff
        self.sep_cpeff = sep_coeff
        self.lr = lr
        self.num_classes = num_classes
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.prototype_activation_function = 'log'
        self.ksize = 3


        self.encoder = EncoderBlock(self.input_channels, self.latent_channels, self.ksize)
        self.decoder = DecoderBlock(self.latent_channels, self.input_channels, self.ksize)

        # lists to store loses from each step
        # losses sores as tuple (rec_loss, kl_loss, total_loss)
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

        '''
        Initialization of prototype class identity. Thi part is from PrototPNet Implementation.
        '''
        assert (self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

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
        # recon_loss = F.binary_cross_entropy(x_hat, x, reduction="mean")
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

        self.log_dict(tr_logs, on_epoch=True, on_step=False)
        self.training_step_losses.append(tuple(tr_logs.values()))
        return loss

    def validation_step(self, batch, batch_idx):

        loss, logs, x_hat = self.step(batch, batch_idx)

        val_logs = dict()
        for key, val in logs.items():
            new_key = "val_"+str(key)
            val_logs[new_key] = val

        self.log_dict(val_logs, on_epoch=True, on_step=False)
        # store loss as well as reconstructed images
        self.validation_step_losses.append(tuple(val_logs.values()))

        if batch_idx == 0:
            self.val_outs = batch

        return loss

    def test_step(self, batch, batch_idx):

        loss, logs, x_hat = self.step(batch, batch_idx)

        test_logs = dict()
        for key, val in logs.items():
            new_key = "test_" + str(key)
            test_logs[new_key] = val

        self.log_dict(test_logs, on_epoch=True, on_step=False)
        self.test_step_losses.append(tuple(test_logs.values()))

        if batch_idx == 0:
            self.test_outs = batch

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

        if self.global_rank == 0:
            val_dir = join(self.logger.save_dir, self.logger.name, f"version_{self.logger.version}", "validation_results")
            create_dir(val_dir)

            # Saving validation results.
            x, y = self.val_outs
            z, x_hat, p, q = self._run_step(x)

            if self.current_epoch == 0:
                grid = vutils.make_grid(x, nrow=8, normalize=False)
                vutils.save_image(x, join(val_dir, f"orig_{self.logger.name}_{self.current_epoch}.png"), normalize=False, nrow=8)
                self.logger.experiment.add_image(f"orig_{self.logger.name}_{self.current_epoch}", grid, self.global_step)

            grid = vutils.make_grid(x_hat, nrow=8, normalize=False)
            vutils.save_image(x_hat, join(val_dir, f"recons_{self.logger.name}_{self.current_epoch}.png"), normalize=False, nrow=8)
            self.logger.experiment.add_image(f"recons_{self.logger.name}_{self.current_epoch}", grid, self.global_step)



    def on_test_epoch_end(self):
        cum_rec_loss, cum_kl_loss, cum_total_loss = self.get_cumulative_losses(self.test_step_losses)
        print(f'\nTest Epoch({self.current_epoch}): rec_loss: {cum_rec_loss}, kl_loss:{cum_kl_loss}, total_loss:{cum_total_loss}')

        # store epoch loss
        self.test_history['test_rec_loss'].append(cum_rec_loss)
        self.test_history['test_kl_loss'].append(cum_kl_loss)
        self.test_history['test_total_loss'].append(cum_total_loss)
        self.test_step_losses.clear()

        if self.global_rank == 0:
            test_dir = join(self.logger.save_dir, self.logger.name, f"version_{self.logger.version}", "test_results")
            create_dir(test_dir)

            # Saving test results
            x, y = self.test_outs
            z, x_hat, p, q = self._run_step(x)

            grid = vutils.make_grid(x, nrow=8, normalize=False)
            vutils.save_image(x, join(test_dir, f"test_orig_{self.logger.name}_{self.current_epoch}.png"), normalize=False, nrow=8)
            self.logger.experiment.add_image(f"test_orig_{self.logger.name}_{self.current_epoch}", grid, self.global_step)

            grid = vutils.make_grid(x_hat, nrow=8, normalize=False)
            vutils.save_image(x_hat, join(test_dir, f"test_recons_{self.logger.name}_{self.current_epoch}.png"), normalize=False, nrow=8)
            self.logger.experiment.add_image(f"test_recons_{self.logger.name}_{self.current_epoch}", grid, self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_history(self):
        return self.train_history, self.val_history

    def save_test_loss_data(self, path):
        save_path = os.path.join(path, 'test_loss.json')
        with open(save_path, 'w') as json_file:
            json.dump(self.test_history, json_file)


