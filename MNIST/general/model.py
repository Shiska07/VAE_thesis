# VAE implementation from pytorch_lightning bolts is used in this source code:
# https://github.com/Lightning-Universe/lightning-bolts/tree/master/pl_bolts/models/autoencoders
import os
import json
from os.path import join

import torch
from torch import nn
import torchvision.utils as vutils
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from helpers import create_dir, get_average_losses
from settings import class_specific, use_l1_mask, n_classes, encoder_out_channels,\
   prototype_shape, num_prototypes, prototype_activation_function
from vae_components import EncoderBlock, EncoderBottleneck, DecoderBlock


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
        l1_coeff,
        lr = 1e-4,
        init_weights=True

    ):

        super().__init__()

        # saving hparams to model state dict
        self.save_hyperparameters()

        self.lr = lr
        self.epsilon = 1e-4

        # coeffs for loss
        self.ce_coeff = ce_coeff
        self.kl_coeff = kl_coeff
        self.recon_coeff = recon_coeff
        self.clst_coeff = clst_coeff
        self.sep_coeff = sep_coeff
        self.l1_coeff = l1_coeff

        self.input_height = input_height
        self.input_channels = input_channels
        self.latent_channels = prototype_shape[1]

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

        assert (num_prototypes % n_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(num_prototypes,
                                                    n_classes)

        num_prototypes_per_class = num_prototypes // n_classes
        for j in range(num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        # PROTOTYPE AND CLASSIFIER COMPONENTS
        self.prototype_vectors = nn.Parameter(torch.rand(prototype_shape),
                                              requires_grad=True)

        self.ones = nn.Parameter(torch.ones(prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(num_prototypes, n_classes,
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
        if prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif prototype_activation_function == 'linear':
            return -distances
        return prototype_activation_function(distances)

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
        min_distances = min_distances.view(-1, num_prototypes)
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
        min_distances = min_distances.view(-1, num_prototypes)
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
        if class_specific:
            max_dist = (prototype_shape[1]
                        * prototype_shape[2]
                        * prototype_shape[3])

            # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
            # calculate cluster cost
            prototypes_of_correct_class = torch.t(self.prototype_class_identity[:, y])
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

            if use_l1_mask:
                l1_mask = 1 - torch.t(self.prototype_class_identity)
                l1 = (self.last_layer.weight * l1_mask).norm(p=1)
            else:
                l1 = self.last_layer.weight.norm(p=1)

        else:
            min_distance, _ = torch.min(min_distances, dim=1)
            cluster_cost = torch.mean(min_distance)
            l1 = self.last_layer.weight.norm(p=1)

        # TOTAL LOSS
        if class_specific:
            loss = (self.kl_coeff * kl
                    + self.recon_coeff * recon_loss
                    + self.ce_coeff * cross_entropy
                    + self.clst_coeff + cluster_cost
                    + self.sep_coeff * separation_cost
                    + self.l1_coeff * l1)

            logs = {
                "recon_loss": recon_loss,
                "kl_loss": kl,
                "ce_loss": cross_entropy,
                "clst_loss": cluster_cost,
                "sep_loss": separation_cost,
                "l1_loss": l1,
                "total_loss": loss,
            }

        else: # exclude separation cost if not class specific
            loss = (self.kl_coeff * kl
                + self.recon_coeff * recon_loss
                + self.ce_coeff * cross_entropy
                + self.clst_coeff + cluster_cost
                + self.l1_coeff * l1)

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
        tag = "train"
        for key, val in logs.items():
            self.logger.experiment.add_scalars(key, {tag:val}, self.global_step)

        tr_logs = dict()
        for key, val in logs.items():
            new_key = str(key)+"/train"
            tr_logs[new_key] = val

        # automatically accumulates the losss for each epoch and logs
        #self.log_dict(tr_logs, on_epoch=True, on_step=False)
        self.training_step_losses.append(tr_logs)
        return loss


    def validation_step(self, batch, batch_idx):

        loss, logs, x_hat = self.step(batch, batch_idx)
        tag = "val"
        for key, val in logs.items():
            self.logger.experiment.add_scalars(key, {tag: val}, self.global_step)

        val_logs = dict()
        for key, val in logs.items():
            new_key = str(key)+"/val"
            val_logs[new_key] = val

        # self.log_dict(val_logs, on_epoch=True, on_step=False)
        self.validation_step_losses.append(val_logs)

        if batch_idx == 0:
            self.val_outs = batch

        return loss

    def test_step(self, batch, batch_idx):

        loss, logs, x_hat = self.step(batch, batch_idx)
        tag = "test"

        for key, val in logs.items():
            self.logger.experiment.add_scalars(key, {tag: val}, self.global_step)

        test_logs = dict()
        for key, val in logs.items():
            new_key = str(key)+"/test"
            test_logs[new_key] = val

        # self.log_dict(test_logs, on_epoch=True, on_step=False)
        self.test_step_losses.append(test_logs)

        if batch_idx == 0:
            self.test_outs = batch

        return loss

    '''
    Here we would do the prototype projection depending on the epoch.
    '''
    def on_train_epoch_end(self):

        # calculate colulative losses per epoch
        avg_rec_loss, avg_kl_loss, avg_ce_loss, avg_clst_loss, avg_sep_loss, \
            avg_l1_loss, avg_total_loss = get_average_losses(
            self.training_step_losses, "/train")

        print(f"\nTraining Epoch[{self.current_epoch}]:\n\
                rec_loss: {avg_rec_loss}\n\
                kl_loss: {avg_kl_loss}\n\
                ce_loss: {avg_ce_loss}\n\
                clst_loss: {avg_clst_loss}\n\
                sep_loss: {avg_sep_loss}\n\
                l1_loss: {avg_l1_loss}\n\
                total_loss: {avg_total_loss}\n")

        self.training_step_losses.clear()


    def on_validation_epoch_end(self):
        # calculate colulative losses per epoch
        avg_rec_loss, avg_kl_loss, avg_ce_loss, avg_clst_loss, avg_sep_loss, \
            avg_l1_loss, avg_total_loss = get_average_losses(
            self.validation_step_losses, "/val")

        print(f"\nValidation Epoch[{self.current_epoch}]:\n\
                rec_loss: {avg_rec_loss}\n\
                kl_loss: {avg_kl_loss}\n\
                ce_loss: {avg_ce_loss}\n\
                clst_loss: {avg_clst_loss}\n\
                sep_loss: {avg_sep_loss}\n\
                l1_loss: {avg_l1_loss}\n\
                total_loss: {avg_total_loss}\n")

        self.validation_step_losses.clear()

        # log this for early stopping
        self.log("avg_total_val_loss", avg_total_loss)

        # to save reconstructed images
        val_dir = join(self.logger.save_dir, self.logger.name, f"version_{self.logger.version}", "validation_results")
        create_dir(val_dir)

        # Saving validation results. val_outs contain very first batch
        x, y = self.val_outs
        p, q, z, x_hat, logits, min_distances = self._run_step(x)

        # If this is the first epoch save true images
        if self.current_epoch == 0:
            grid = vutils.make_grid(x, nrow=8, normalize=False)
            vutils.save_image(x, join(val_dir, f"orig_{self.logger.name}_{self.current_epoch}.png"), normalize=False, nrow=8)
            self.logger.experiment.add_image(f"orig_{self.logger.name}_{self.current_epoch}", grid, self.global_step)

        # save reconstructions of the first batch of validation images for each
        # epoch
        grid = vutils.make_grid(x_hat, nrow=8, normalize=False)
        vutils.save_image(x_hat, join(val_dir, f"recons_{self.logger.name}_{self.current_epoch}.png"), normalize=False, nrow=8)
        self.logger.experiment.add_image(f"recons_{self.logger.name}_{self.current_epoch}", grid, self.global_step)


    def on_test_epoch_end(self):
        avg_rec_loss, avg_kl_loss, avg_ce_loss, avg_clst_loss, avg_sep_loss, \
            avg_l1_loss, avg_total_loss = get_average_losses(
            self.test_step_losses, "/test")

        print(f"\nTest Epoch[{self.current_epoch}]:\n\
               rec_loss: {avg_rec_loss}\n\
               kl_loss: {avg_kl_loss}\n\
               ce_loss: {avg_ce_loss}\n\
               clst_loss: {avg_clst_loss}\n\
               sep_loss: {avg_sep_loss}\n\
               l1_loss: {avg_l1_loss}\n\
               total_loss: {avg_total_loss}\n")

        self.test_step_losses.clear()

        test_dir = join(self.logger.save_dir, self.logger.name, f"version_{self.logger.version}", "test_results")
        create_dir(test_dir)

        # Saving test results. test_outs contains images from the very first batch
        x, y = self.test_outs
        p, q, z, x_hat, logits, min_distances = self._run_step(x)

        # save true images of the very fitst batch (test_outs)
        grid = vutils.make_grid(x, nrow=8, normalize=False)
        vutils.save_image(x, join(test_dir, f"test_orig_{self.logger.name}_{self.current_epoch}.png"), normalize=False, nrow=8)
        self.logger.experiment.add_image(f"test_orig_{self.logger.name}_{self.current_epoch}", grid, self.global_step)

        # save reconstruction of the very first batch
        grid = vutils.make_grid(x_hat, nrow=8, normalize=False)
        vutils.save_image(x_hat, join(test_dir, f"test_recons_{self.logger.name}_{self.current_epoch}.png"), normalize=False, nrow=8)
        self.logger.experiment.add_image(f"test_recons_{self.logger.name}_{self.current_epoch}", grid, self.global_step)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_history(self):
        pass

    def save_test_loss_data(self, path):
        save_path = os.path.join(path, 'test_loss.json')
        with open(save_path, 'w') as json_file:
            json.dump(self.test_history, json_file)


