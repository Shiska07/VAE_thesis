# VAE implementation from pytorch_lightning bolts is used in this source code:
# https://github.com/Lightning-Universe/lightning-bolts/tree/master/pl_bolts/models/autoencoders
import os
import sys
import pdb
import json
from os.path import join

import torch
import torchsummary
from torch import nn
import torchvision.utils as vutils
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from helpers import create_dir, get_average_losses, get_accuracy
from settings import class_specific, use_l1_mask, n_classes, encoder_out_channels, \
    prototype_shape, num_prototypes, prototype_activation_function, push_start, \
    push_epochs_interval, weight_matrix_filename, prototype_img_filename_prefix, \
    proto_bound_boxes_filename_prefix, prototype_self_act_filename_prefix, latent_dim
from vae_components import EncoderBlock, DecoderBlock

'''
Some components of the following implementation were obtained from: https://github.com/cfchen-duke/ProtoPNet
 '''

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
        prototype_saving_dir,
        logging_dir,
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
        self.prototype_saving_dir = prototype_saving_dir
        self.logging_dir = logging_dir

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
        self.decoder = DecoderBlock(encoder_out_channels, self.latent_channels, self.input_channels)

        self.mode = None
        self.test_tag = "test"
        self.last_push_epoch = 0

        '''
        Initialization of prototype class identity. Thi part is from PrototPNet Implementation.
        '''

        assert (num_prototypes % n_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(num_prototypes,
                                                    n_classes).cuda()

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

    def on_fit_start(self):
        encoder_input_dim = (self.input_channels, self.input_height, self.input_height)
        decoder_input_dim = latent_dim
        prototype_input_dim = latent_dim
        prototype_out_dim = (num_prototypes, latent_dim[1], latent_dim[2])
        global_max_pooling_out = (num_prototypes,)

        orig_stdout = sys.stdout

        # save architecture
        model_arc_path = os.path.join(self.logging_dir, "model_arc")
        output_file = os.path.join(model_arc_path, "model_arc.txt")
        create_dir(model_arc_path)

        with open(output_file, "w", encoding="utf-8") as f:
            # Redirect stdout to the file
            sys.stdout = f

            # ENCODER ARCHITECTURE
            print(self.encoder)
            torchsummary.summary(self.encoder, encoder_input_dim)

            # DECODER ARCHITECTURE
            print(self.decoder)
            torchsummary.summary(self.decoder, decoder_input_dim)

            # CLASSIFIER ARCHITECTURE
            print("CLASSIFIER BLOCK ARC:")
            print("\n")
            print("PROTOTYPE LAYER:")
            print(f"prototypeL_input_dim: {prototype_input_dim}")
            print(f"Prototype Layer Dim: {(self.prototype_vectors.size())}")
            print(f"prototypeL_output_dim: {prototype_out_dim}")
            print("GLOBAL MAX POOLING")
            print(f"maxpool_output_dim: {global_max_pooling_out}")
            print("FULLY CONNECTED LAYER:")
            print(self.last_layer)
            torchsummary.summary(self.last_layer, global_max_pooling_out)
            print("\n")

        # Restore the original stdout
        sys.stdout = orig_stdout
        print(f"Model architecture saved to {output_file}")


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

    def set_mode(self, mode="joint"):
        self.mode = mode

        # warm up classifier block
        if self.mode == "vae_only":
            for param in self.encoder.parameters():
                param.requires_grad = True

            for param in self.decoder.parameters():
                param.requires_grad = True

            self.prototype_vectors.requires_grad = False

            for param in self.last_layer.parameters():
                param.requires_grad = False
            print("\t\t***************** Mode = vae_only ******************")

        elif self.mode == "warm":
            for param in self.encoder.parameters():
                param.requires_grad = False

            for param in self.decoder.parameters():
                param.requires_grad = False

            self.prototype_vectors.requires_grad = True

            for param in self.last_layer.parameters():
                param.requires_grad = True
            print("\t\t***************** Mode = warm ******************")

        elif self.mode == "joint":
            for param in self.encoder.parameters():
                param.requires_grad = True

            for param in self.decoder.parameters():
                param.requires_grad = True

            self.prototype_vectors.requires_grad = True

            for param in self.last_layer.parameters():
                param.requires_grad = False
            print("\t\t***************** Mode = joint ******************")


        elif self.mode == "last_only":
            for param in self.encoder.parameters():
                param.requires_grad = False

            for param in self.decoder.parameters():
                param.requires_grad = False

            self.prototype_vectors.requires_grad = False

            for param in self.last_layer.parameters():
                param.requires_grad = True
            print("\t\t***************** Mode = last layer ******************")

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

        # distances.shape = (64, 40, 2, 2)
        return distances

    def prototype_distances(self, x):

        # make sure values are between 0-1
        x = torch.sigmoid(x)

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

    def push_forward(self, x):
        encoder_out = self.encoder(x)
        mu = encoder_out[:,:self.latent_channels,:,:]
        logvar = encoder_out[:,self.latent_channels:,:,:]
        p, q, z = self.sample(mu, logvar)
        distances = self.prototype_distances(z)

        return p, q, z, distances

    '''
    The forward method is for running data through your model during testing and usage.
    In this case we want the output logit, reconstruction(we may not need this later) and the min distances. 
    '''
    def forward(self, x):
        p, q, z, distances = self.push_forward(x)

        # get reconstruction
        x_hat = self.decoder(z)

        '''
        We have L2 distance and wish to extract the patch woth the lowest 
        distance. [2, 3, 10, 6]
        1. -distance: makes sure that patch with lowest distance becomes the 
        highest value [-2, -3, -10, -6]
        2. maxpooling from these vaule extracts the number with the lowest 
        distance but negated [-2]
        3. negating this gives the original value: 2
        4. Then this is converted to similarity by negating again since lower 
        distance values should correspond to higher similarity
        '''
        distances = self.prototype_distances(z)
        min_distances = -F.max_pool2d(-distances, kernel_size=(distances.size()[2], distances.size()[3]))
        min_distances = min_distances.view(-1, num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)

        # get prediction
        logits = self.last_layer(prototype_activations)
        return logits, min_distances, x_hat


    def _run_step(self, x):
        encoder_out = self.encoder(x)
        mu = encoder_out[:, :self.latent_channels, :, :]
        logvar = encoder_out[:, self.latent_channels:, :, :]
        p, q, z = self.sample(mu, logvar)

        # get reconstruction
        x_hat = self.decoder(z)

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


        # recon_loss = F.mse(x_hat, x, reduction="mean")
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="mean")

        # kl loss
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()

        # cross entropy loss
        cross_entropy = F.cross_entropy(logits, y)

        # calculate accuracy
        acc = get_accuracy(logits, y)

        # CLST and SEP Cost
        if class_specific:

            '''
            CLUSTER COST
            '''
            max_dist = (prototype_shape[1]
                        * prototype_shape[2]
                        * prototype_shape[3])

            '''
            prototypes_of_correct_class is a tensor of shape batch_size * \
                                                              num_prototypes
            for each training example, it represents the 1 x m vector with 0's and 1's
            where 1's indicate the correct prototypes for the training example 
            according to the class it belongs to
            '''

            # calculate cluster cost
            prototypes_of_correct_class = torch.t(
                self.prototype_class_identity[:, y])

            '''
            1. The step below calculates the inverted distance of each training 
            example with the prototype of its class.
            '''
            # subtract with max distance such that higher values represent closer
            # datapoints
            similarity_vals = max_dist - min_distances
            similarity_vals_with_corr_prototypes = \
                similarity_vals * prototypes_of_correct_class

            # maximum with dim=1 i.e. columns to find one prototype that the
            # training example is closest to
            inverted_distances, _ = torch.max(similarity_vals_with_corr_prototypes,
                                              dim=1)

            # now that we have the similarity or inverted distance with the
            # closest prototype, we convert it again to distance
            distances_to_closest_correct_prototype = max_dist - inverted_distances
            cluster_cost = torch.mean(distances_to_closest_correct_prototype)

            '''
            SEPARATION COST
            '''
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            similarity_vals_with_incor_prototypes = similarity_vals * prototypes_of_wrong_class

            # find the 1 prototype from a wrong class with the highest similarity/
            # inverted distance
            inverted_distances, _ = torch.max(
                similarity_vals_with_incor_prototypes, dim=1)
            distances_to_closest_incor_prototype = max_dist - inverted_distances
            separation_cost = torch.mean(distances_to_closest_incor_prototype)

            # calculate avg cluster cost
            avg_separation_cost = \
                torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / \
                torch.sum(prototypes_of_wrong_class, dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)

            if use_l1_mask:
                l1_mask = 1 - torch.t(self.prototype_class_identity)
                l1 = (self.last_layer.weight * l1_mask).norm(p=1)
            else:
                l1 = self.last_layer.weight.norm(p=1)


            if self.mode == "vae_only":
                loss = self.kl_coeff * kl \
                       + self.recon_coeff * recon_loss

            else:
                # DEFINE LOSS TERM
                loss = (self.kl_coeff * kl
                        + self.recon_coeff * recon_loss
                        + self.ce_coeff * cross_entropy
                        + self.clst_coeff + cluster_cost
                        + self.sep_coeff * separation_cost
                        + self.l1_coeff * l1)

            total_loss = (self.kl_coeff * kl
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
                "total_loss": total_loss,
                "acc": acc
            }


        else:
            '''
            If we don't care about class specificity, find the closest prototype 
            to each training example (min_distance, _ = torch.min(min_distances, 
            dim=1)) and take the mean to calculate cluster cost. Do not calculate 
            separation cost.
            '''
            min_distance, _ = torch.min(min_distances, dim=1)
            cluster_cost = torch.mean(min_distance)
            l1 = self.last_layer.weight.norm(p=1)

            if self.mode == "vae_only":
                loss = self.kl_coeff * kl \
                       + self.recon_coeff * recon_loss

            else:
                # DEFINE LOSS TERM
                loss = (self.kl_coeff * kl
                        + self.recon_coeff * recon_loss
                        + self.ce_coeff * cross_entropy
                        + self.clst_coeff + cluster_cost
                        + self.l1_coeff * l1)

            total_loss = (self.kl_coeff * kl
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
                "total_loss": total_loss,
                "acc": acc
            }

        return loss, logs, x_hat

    '''
    Here we return the loss and logs for a single batch.
    '''
    def training_step(self, batch, batch_idx):

        loss, logs, x_hat = self.step(batch, batch_idx)

        self.training_step_losses.append(logs)
        return loss


    def validation_step(self, batch, batch_idx):

        loss, logs, x_hat = self.step(batch, batch_idx)
        self.validation_step_losses.append(logs)

        if batch_idx == 0:
            self.val_outs = batch

        return loss


    def test_step(self, batch, batch_idx):

        loss, logs, x_hat = self.step(batch, batch_idx)
        self.test_step_losses.append(logs)

        if batch_idx == 0:
            self.test_outs = batch

        return loss


    def on_validation_epoch_end(self):

        # calculate colulative losses per epoch
        avg_metric_dict = get_average_losses(
            self.validation_step_losses)

        tag = "val"
        print(f"\nVALIDATION Epoch[{self.current_epoch}]:")

        for key, val in avg_metric_dict.items():
            print(f"{key} : {val:0.4f}")
            self.logger.experiment.add_scalars(key, {tag: val}, self.current_epoch)
        print("\n")
        self.validation_step_losses.clear()

        # log this for early stopping
        self.log("val_total_loss", avg_metric_dict["total_loss"])

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

        # calculate colulative losses per epoch
        avg_metric_dict = get_average_losses(
            self.test_step_losses)

        print(f"\nTEST Epoch[{self.current_epoch}]:")
        tag = self.test_tag
        for key, val in avg_metric_dict.items():
            print(f"{key} : {val:0.4f}")
            self.logger.experiment.add_scalars(key, {tag: val}, self.current_epoch)
        print("\n")
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




