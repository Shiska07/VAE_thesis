# VAE implementation from pytorch_lightning bolts is used in this source code:
# https://github.com/Lightning-Universe/lightning-bolts/tree/master/pl_bolts/models/autoencoders
import json
import os
from os.path import join

import torch
import torchsummary
import torchvision.utils as vutils
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from utils import create_dir
from general.settings import prototype_shape, n_classes, encoder_out_channels
from vae_components import EncoderBlock, EncoderBottleneck,DecoderBlock


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
        init_weights=True

    ):

        super().__init__()

        # saving hparams to model state dict
        self.save_hyperparameters()

        self.lr = lr
        self.ksize = 3
        self.epsilon = 1e-4
        self.n_classes = n_classes

        # coeffs for loss
        self.ce_coeff = ce_coeff
        self.kl_coeff = kl_coeff
        self.recon_coeff = recon_coeff
        self.clst_coeff = clst_coeff
        self.sep_cpeff = sep_coeff

        self.input_height = input_height
        self.input_channels = input_channels
        self.latent_channels = prototype_shape[1]

        # some params from ProtoPNet
        self.class_specific = True
        self.use_l1_mask = True

        # lists to store loses from each step
        # losses sores as tuple (rec_loss, kl_loss, total_loss)
        self.training_step_losses = []
        self.validation_step_losses = []
        self.test_step_losses = []

        # lists to store reconstruction images
        self.val_outs = []
        self.test_outs = []

        # VAE COMPONENTS
        self.encoder = EncoderBlock(self.input_channels, encoder_out_channels)
        self.bottleneck = EncoderBottleneck(encoder_out_channels, self.latent_channels)
        self.decoder = DecoderBlock(encoder_out_channels, self.latent_channels, self.input_channels)

        '''
        Initialization of prototype class identity. Thi part is from PrototPNet Implementation.
        '''

        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.prototype_activation_function = 'log'


        assert (self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        # PROTOTYPE AND CLASSIFIER COMPONENTS
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.n_classes,
                                    bias=False)  # do not use bias

        # itntialize weights for the last layer
        if init_weights:
            self._initialize_weights()

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    def _l2_convolution(self, x):

        # apply self.prototype_vectors as l2-convolution filters on input x
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def prototype_distances(self, x):

        # x is the sample from bottleneck
        distances = self._l2_convolution(x)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        return self.prototype_activation_function(distances)

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    '''
    The forward method is for running data through your model during testing and usage.
    In this case we want the output logit, reconstruction(we may not need this later) and the min distances. 
    '''
    def forward(self, x):
        encoder_out = self.encoder(x)
        mu, logvar = self.bottleneck(encoder_out)
        p, q, z = self.sample(mu, logvar)

        # get reconstruction
        x_hat = self.decoder(z)

        '''
        ProtoPNet
        '''
        # global min pooling because min distance corresponds to max similarity
        distances = self.prototype_distances(z)
        min_distances = -F.max_pool2d(-distances, kernel_size=(distances.size()[2], distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)

        # get prediction
        logits = self.last_layer(prototype_activations)
        return logits, min_distances, x_hat


    def _run_step(self, x):
        encoder_out = self.encoder(x)
        mu, logvar = self.bottleneck(encoder_out)
        p, q, z = self.sample(mu, logvar)

        # get reconstruction
        x_hat = self.decoder(z)

        '''
        ProtoPNet
        '''
        # global min pooling because min distance corresponds to max similarity
        distances = self.prototype_distances(z)
        min_distances = -F.max_pool2d(-distances, kernel_size=(distances.size()[2], distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)

        # get prediction
        logits = self.last_layer(prototype_activations)
        return p, q, z, x_hat, logits, min_distances


    def step(self, batch, batch_idx):
        x, y = batch
        p, q, z, x_hat, logits, min_distances = self._run_step(x)

        # recon_loss = F.binary_cross_entropy(x_hat, x, reduction="mean")
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        # kl loss
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        # cross entropy loss
        cross_entropy = torch.nn.functional.cross_entropy(logits, y)

        # CLST and SEP Cost
        if self.class_specific:
            max_dist = (self.prototype_shape[1]
                        * self.prototype_shape[2]
                        * self.prototype_shape[3])

            # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
            # calculate cluster cost
            prototypes_of_correct_class = torch.t(self.prototype_class_identity[:, y]).cuda()
            inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
            cluster_cost = torch.mean(max_dist - inverted_distances)

            # calculate separation cost
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = \
                torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

            # calculate avg cluster cost
            avg_separation_cost = \
                torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class,
                                                                                        dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)

            if self.use_l1_mask:
                l1_mask = 1 - torch.t(self.prototype_class_identity).cuda()
                l1 = (self.last_layer.weight * l1_mask).norm(p=1)
            else:
                l1 = self.last_layer.weight.norm(p=1)

        else:
            min_distance, _ = torch.min(min_distances, dim=1)
            cluster_cost = torch.mean(min_distance)
            l1 = self.last_layer.weight.norm(p=1)

        # TOTOAL LOSS
        if self.class_specific:
            loss = (self.kl_coeff * kl
                    + self.recon_coeff * recon_loss
                    + self.ce_coeff * cross_entropy
                    + self.clst_coeff + cluster_cost
                    + self.sep_coeff * separation_cost
                    + 1e-4 * l1)

            logs = {
                "recon_loss": recon_loss,
                "kl_loss": kl,
                "ce_loss": cross_entropy,
                "clst_loss": cluster_cost,
                "sep_loss": separation_cost,
                "l1_loss": l1,
                "total_loss": loss,
            }

        else: # excluse separation cost if not class specific
            loss = (self.kl_coeff * kl
                + self.recon_coeff * recon_loss
                + self.ce_coeff * cross_entropy
                + self.clst_coeff + cluster_cost
                + 1e-4 * l1)

            logs = {
                "recon_loss": recon_loss,
                "kl_loss": kl,
                "ce_loss": cross_entropy,
                "clst_loss": cluster_cost,
                "l1_loss": l1,
                "total_loss": loss,
            }

        return loss, logs, x_hat

    '''
    Here we return the loss and logs for a single batch.
    '''
    def training_step(self, batch, batch_idx):

        loss, logs , _ = self.step(batch, batch_idx)

        # modify train logs
        tr_logs = dict()
        for key, val in logs.items():
            new_key = "train_"+str(key)
            tr_logs[new_key] = val

        # automatically accumulates the losss for each epoch and logs
        self.log_dict(tr_logs, on_epoch=True, on_step=False)
        self.training_step_losses.append(tr_logs)
        return loss


    def validation_step(self, batch, batch_idx):

        loss, logs, x_hat = self.step(batch, batch_idx)

        val_logs = dict()
        for key, val in logs.items():
            new_key = "val_"+str(key)
            val_logs[new_key] = val

        self.log_dict(val_logs, on_epoch=True, on_step=False)
        self.validation_step_losses.append(val_logs)

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
        self.test_step_losses.append(test_logs)

        if batch_idx == 0:
            self.test_outs = batch

        return loss

    def get_cumulative_losses(self, losses_list):

        num_items = len(losses_list)

        cum_rec_loss = 0
        cum_kl_loss = 0
        cum_ce_loss = 0
        cum_clst_loss = 0
        cum_sep_loss = 0
        cum_l1_loss = 0
        cum_total_loss = 0

        for loss_dict in losses_list:
            cum_rec_loss += loss_dict["recon_loss"]
            cum_kl_loss += loss_dict["kl_loss"]
            cum_ce_loss += loss_dict["ce_loss"]
            cum_clst_loss += loss_dict["clst_loss"]
            cum_sep_loss += loss_dict["sep_loss"]
            cum_l1_loss += loss_dict["sep_loss"]
            cum_total_loss += loss_dict["total_loss"]


        # get average loss
        avg_rec_loss = cum_rec_loss / num_items
        avg_kl_loss = cum_kl_loss / num_items
        avg_ce_loss = cum_ce_loss / num_items
        avg_clst_loss = cum_clst_loss / num_items
        avg_sep_loss = cum_sep_loss / num_items
        avg_l1_loss = cum_l1_loss / num_items
        avg_total_loss = cum_total_loss / num_items

        return avg_rec_loss, avg_kl_loss, avg_ce_loss, \
        avg_clst_loss, avg_sep_loss, avg_l1_loss, avg_total_loss

    '''
    Here we would do the prototype projection depending on the epoch.
    '''
    def on_train_epoch_end(self):

        # calculate colulative losses per epoch
        avg_rec_loss, avg_kl_loss, avg_ce_loss, avg_clst_loss, avg_sep_loss, \
            avg_l1_loss, avg_total_loss = self.get_cumulative_losses(self.training_step_losses)

        print(f"\nTraining Epoch[{self.current_epoch}]:\n \
                rec_loss: {avg_rec_loss}\n\
                kl_loss: {avg_kl_loss}\n\
                ce_loss: {avg_ce_loss}\n\
                clst_loss: {avg_clst_loss}\n\
                sap_loss: {avg_sep_loss}\n\
                l1_loss: {avg_l1_loss}\n\
                total_loss: {avg_total_loss}\n")

        self.training_step_losses.clear()


    def on_validation_epoch_end(self):
        # calculate colulative losses per epoch
        avg_rec_loss, avg_kl_loss, avg_ce_loss, avg_clst_loss, avg_sep_loss, \
            avg_l1_loss, avg_total_loss = self.get_cumulative_losses(
            self.validation_step_losses)

        print(f"\nValidation Epoch[{self.current_epoch}]:\n \
                        rec_loss: {avg_rec_loss}\n\
                        kl_loss: {avg_kl_loss}\n\
                        ce_loss: {avg_ce_loss}\n\
                        clst_loss: {avg_clst_loss}\n\
                        sap_loss: {avg_sep_loss}\n\
                        l1_loss: {avg_l1_loss}\n\
                        total_loss: {avg_total_loss}\n")

        self.validation_step_losses.clear()

        if self.global_rank == 0:
            val_dir = join(self.logger.save_dir, self.logger.name, f"version_{self.logger.version}", "validation_results")
            create_dir(val_dir)

            # Saving validation results.
            x, y = self.val_outs
            p, q, z, x_hat, logits, min_distances = self._run_step(x)

            if self.current_epoch == 0:
                grid = vutils.make_grid(x, nrow=8, normalize=False)
                vutils.save_image(x, join(val_dir, f"orig_{self.logger.name}_{self.current_epoch}.png"), normalize=False, nrow=8)
                self.logger.experiment.add_image(f"orig_{self.logger.name}_{self.current_epoch}", grid, self.global_step)

            grid = vutils.make_grid(x_hat, nrow=8, normalize=False)
            vutils.save_image(x_hat, join(val_dir, f"recons_{self.logger.name}_{self.current_epoch}.png"), normalize=False, nrow=8)
            self.logger.experiment.add_image(f"recons_{self.logger.name}_{self.current_epoch}", grid, self.global_step)


    def on_test_epoch_end(self):
        avg_rec_loss, avg_kl_loss, avg_ce_loss, avg_clst_loss, avg_sep_loss, \
            avg_l1_loss, avg_total_loss = self.get_cumulative_losses(
            self.test_step_losses)

        print(f"\nTest Epoch[{self.current_epoch}]:\n \
                               rec_loss: {avg_rec_loss}\n\
                               kl_loss: {avg_kl_loss}\n\
                               ce_loss: {avg_ce_loss}\n\
                               clst_loss: {avg_clst_loss}\n\
                               sap_loss: {avg_sep_loss}\n\
                               l1_loss: {avg_l1_loss}\n\
                               total_loss: {avg_total_loss}\n")

        self.test_step_losses.clear()

        if self.global_rank == 0:
            test_dir = join(self.logger.save_dir, self.logger.name, f"version_{self.logger.version}", "test_results")
            create_dir(test_dir)

            # Saving test results
            x, y = self.test_outs
            p, q, z, x_hat, logits, min_distances = self._run_step(x)

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

