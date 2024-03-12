# VAE implementation from pytorch_lightning bolts is used in this source code:
# https://github.com/Lightning-Universe/lightning-bolts/tree/master/pl_bolts/models/autoencoders
import os
import json
import torch
from torch import nn
from os.path import join
from itertools import chain
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from configs.train_settings import class_specific, use_l1_mask
from utils.helpers import create_dir, get_average_losses, get_accuracy, get_logs
from configs.lr_configs import warm_vae_optimizer_lrs, joint_optimizer_lrs, last_layer_optimizer_lr, joint_lr_step_size
from configs.loss_configs import kl_coeff, ce_coeff, recon_coeff, l1_coeff, \
    clst_coeff, sep_coeff

from configs.proto_configs import input_height, input_channels, prototype_shape, \
    num_prototypes, latent_channels, latent_dim, encoder_out_channels, n_classes, \
    prototype_activation_function

from model_components.vae_components import EncoderBlock, DecoderBlock
from model_components.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
'''
Some components of the following implementation were obtained from: https://github.com/cfchen-duke/ProtoPNet
 '''

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features}

class PartProtoVAE(LightningModule):
    def __init__(
        self,
        base_architecture,
        layers_to_exclude,
        prototype_saving_dir,
        logging_dir,
        default_lr = 1e-4,
        init_weights=True

    ):

        super().__init__()

        # saving hparams to model state dict
        self.save_hyperparameters()

        self.default_lr = default_lr
        self.epsilon = 1e-4

        self.multi_stage_step_counter = 0

        self.input_height = input_height
        self.input_channels = input_channels
        self.latent_channels = prototype_shape[1]
        self.base_architecture = base_architecture
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

        # for receptive field calculation
        self.kernel_sizes = []
        self.strides = []
        self.paddings = []

        # PRETRAINED FEATURE EXTRACTOR
        self.layers_to_exclude = layers_to_exclude
        self.base_architecture = base_architecture_to_features[
            self.base_architecture](pretrained=True)
        self.base_features = list(self.base_architecture.children())[
                   :-layers_to_exclude]
        self.features = nn.Sequential(*self.base_features)

        # get input dim for encoder
        self.features_input_dim = (1, 3, self.input_height, self.input_height)
        dummy_input = torch.randn(self.features_input_dim)
        self.features_out_dim = (self.features(dummy_input)).size()

        # VAE COMPONENTS
        self.encoder = EncoderBlock(input_channels=self.features_out_dim[1],
                                    encoder_out_channels=encoder_out_channels)
        self.decoder = DecoderBlock(encoder_out_channels=encoder_out_channels,
                                    latent_channels=latent_channels,
                                    output_channels=self.features_out_dim[1])

        self.mode = None

        '''
        Initialization of prototype class identity. Thi part is from PrototPNet Implementation.
        '''

        assert (num_prototypes % n_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        # self.prototype_class_identity = torch.zeros(num_prototypes,
        #                                             n_classes).cuda()

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

    def conv_info(self):
        feat_kernel_sizes, feat_strides, feat_paddings = \
            self.base_architecture.conv_info()

        feat_kernel_sizes = feat_kernel_sizes[:-self.layers_to_exclude]
        feat_strides = feat_strides[:-self.layers_to_exclude]
        feat_paddings = feat_paddings[:-self.layers_to_exclude]

        self.kernel_sizes = list(chain.from_iterable(feat_kernel_sizes))
        self.strides = list(chain.from_iterable(feat_strides))
        self.paddings = list(chain.from_iterable(feat_paddings))

        encoder_kernel_sizes, encoder_strides, encoder_paddings = self.encoder.conv_info()

        self.kernel_sizes.extend(encoder_kernel_sizes)
        self.strides.extend(encoder_strides)
        self.paddings.extend(encoder_paddings)

        return self.kernel_sizes, self.strides, self.paddings


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


    def configure_optimizers(self):

        if self.mode == "warm_vae":
            print(f"Configuring optimizers for mode {self.mode}\n")
            warm_vae_specs = \
                [{'params': self.encoder.parameters(),
                  'lr': warm_vae_optimizer_lrs['encoder'], 'weight_decay': 1e-3},
                 # bias are now also being regularized
                 {'params': self.decoder.parameters(),
                  'lr': warm_vae_optimizer_lrs['decoder'], 'weight_decay':
                      1e-3},]

            return torch.optim.Adam(warm_vae_specs)

        elif self.mode == "warm_proto":
            print(f"Configuring optimizers for mode {self.mode}\n")
            warm_proto_specs = [{'params': self.prototype_vectors,
                                'lr': warm_vae_optimizer_lrs['prototype_vectors']}]

            return torch.optim.Adam(warm_proto_specs)

        elif self.mode == "joint":
            print(f"Configuring optimizers for mode {self.mode}\n")
            joint_specs = \
                [{'params': self.encoder.features(),
                  'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
                    {'params': self.encoder.parameters(),
                  'lr': joint_optimizer_lrs['encoder'], 'weight_decay': 1e-3},
                 # bias are now also being regularized
                 {'params': self.decoder.parameters(),
                  'lr': joint_optimizer_lrs['decoder'], 'weight_decay':
                      1e-3},
                 {'params': self.prototype_vectors,
                  'lr': joint_optimizer_lrs['prototype_vectors']}
                 ]

            joint_optimizer = torch.optim.Adam(joint_specs)
            joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer,
                                                                 step_size=joint_lr_step_size,
                                                                 gamma=0.1)

            return {"optimizer": joint_optimizer,
                        "lr_scheduler": { "scheduler": joint_lr_scheduler,
                                            "monitor": "val_loss" }}

        elif self.mode == "last_layer":
            last_layer_specs =  [{'params': self.last_layer.parameters(),
                                  'lr': last_layer_optimizer_lr}]

            return torch.optim.Adam(last_layer_specs)

        return torch.optim.Adam(self.parameters(), lr=self.default_lr)


    def set_mode(self, mode=None):
        self.mode = mode

        # warm up classifier block
        if self.mode == "warm_vae":
            self.configure_optimizers()
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True
            self.prototype_vectors.requires_grad = False
            for param in self.last_layer.parameters():
                param.requires_grad = False
            print("\t\t***************** Mode = warm_vae ******************")


        elif self.mode == "warm_proto":
            self.configure_optimizers()
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.prototype_vectors.requires_grad = True
            for param in self.last_layer.parameters():
                param.requires_grad = False
            print("\t\t***************** Mode = warm proto ******************")


        elif self.mode == "joint":
            self.configure_optimizers()
            for param in self.features.parameters():
                param.requires_grad = True
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True
            self.prototype_vectors.requires_grad = True
            for param in self.last_layer.parameters():
                param.requires_grad = False
            print("\t\t***************** Mode = joint ******************")


        elif self.mode == "last_only":
            self.configure_optimizers()
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.prototype_vectors.requires_grad = False
            for param in self.last_layer.parameters():
                param.requires_grad = True
            print("\t\t***************** Mode = last layer ******************")


    def view_lr_info(self):
        pass

    def view_params_grad_info(self):
        pass

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
        feat_out = self.features(x)
        encoder_out = self.encoder(feat_out)
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
        decoder_out = self.decoder(z)

        '''
        We have L2 distance and wish to extract the patch with the lowest 
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
        return logits, min_distances, decoder_out


    def _run_step(self, x):
        feat_out = self.features(x)
        encoder_out = self.encoder(feat_out)
        mu = encoder_out[:, :self.latent_channels, :, :]
        logvar = encoder_out[:, self.latent_channels:, :, :]
        p, q, z = self.sample(mu, logvar)

        # get reconstruction
        decoder_out = self.decoder(z)

        # global min pooling because min distance corresponds to max similarity
        distances = self.prototype_distances(z)
        min_distances = -F.max_pool2d(-distances, kernel_size=(distances.size()[2], distances.size()[3]))
        min_distances = min_distances.view(-1, num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)

        # get prediction
        logits = self.last_layer(prototype_activations)
        return p, q, z, feat_out, decoder_out, logits, min_distances


    def step(self, batch, batch_idx):
        x, y = batch
        p, q, z, feat_out, decoder_out, logits, min_distances = self._run_step(x)

        recon_loss = F.mse_loss(decoder_out, feat_out, reduction="mean")
        kl = (torch.distributions.kl_divergence(q, p)).mean()
        cross_entropy = F.cross_entropy(logits, y)
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

            # adjust coefficients according to the stage
            if self.mode == "warm_vae":
                coeffs = [kl_coeff, recon_coeff, 0, 0, 0, 0]

            elif self.mode == "warm_proto":
                coeffs = [0, 0, ce_coeff, clst_coeff, sep_coeff, 0]

            elif self.mode == "joint":
                coeffs = [recon_coeff, clst_coeff, ce_coeff, clst_coeff, sep_coeff, 0]

            elif self.mode == "last_layer":
                coeffs = [0, 0, ce_coeff, 0, 0,
                          l1_coeff]

            loss = (coeffs[0] * recon_loss
                        + coeffs[1] * kl
                        + coeffs[2] * cross_entropy
                        + coeffs[3] + cluster_cost
                        + coeffs[4] * separation_cost
                        + coeffs[5] * l1)

            total_loss = (recon_coeff * recon_loss
                        + kl_coeff * kl
                        + ce_coeff * cross_entropy
                        + clst_coeff + cluster_cost
                        + sep_coeff * separation_cost
                        + l1_coeff * l1)

            step_losses = [recon_loss, kl, cross_entropy, cluster_cost,
                           separation_cost, l1, total_loss, acc]
            logs = get_logs(self.mode, step_losses, class_specific)


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

            # adjust coefficients according to the stage
            if self.mode == "warm_vae":
                coeffs = [kl_coeff, recon_coeff, 0, 0, 0]

            elif self.mode == "warm_proto":
                coeffs = [0, 0, ce_coeff, clst_coeff, 0]

            elif self.mode == "joint":
                coeffs = [recon_coeff, kl_coeff, ce_coeff, clst_coeff,
                          0]

            elif self.mode == "last_layer":
                coeffs = [0, 0, ce_coeff, 0,
                          l1_coeff]

            loss = (coeffs[0] * recon_loss
                    + coeffs[1] * kl
                    + coeffs[2] * cross_entropy
                    + coeffs[3] + cluster_cost
                    + coeffs[4] * l1)

            total_loss = (recon_coeff * recon_loss
                          + kl_coeff * kl
                          + ce_coeff * cross_entropy
                          + clst_coeff + cluster_cost
                          + l1_coeff * l1)

            step_losses = [recon_loss, kl, cross_entropy, cluster_cost, l1,
                           total_loss, acc]
            logs = get_logs(self.mode, step_losses, class_specific)

        return loss, logs, decoder_out

    '''
    Here we return the loss and logs for a single batch.
    '''
    def training_step(self, batch, batch_idx):
        loss, logs, _ = self.step(batch, batch_idx)
        self.training_step_losses.append(logs)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, logs, _ = self.step(batch, batch_idx)
        self.validation_step_losses.append(logs)
        if batch_idx == 0:
            self.val_outs = batch
        return loss


    def test_step(self, batch, batch_idx):
        loss, logs, _ = self.step(batch, batch_idx)
        self.test_step_losses.append(logs)
        if batch_idx == 0:
            self.test_outs = batch
        return loss


    def on_train_epoch_end(self):
        # calculate colulative losses per epoch
        avg_metric_dict = get_average_losses(
            self.training_step_losses)

        tag = "train"
        print(f"\nTRAINING Epoch[{self.current_epoch}]:")
        for key, val in avg_metric_dict.items():
            print(f"{key} : {val:0.4f}")
            self.logger.experiment.add_scalars(key, {tag: val},
                                                    self.current_epoch)
        print("\n")
        self.training_step_losses.clear()


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

        file_path = os.path.join(test_dir, "test_loss.json")
        with open(file_path, 'w') as json_file:
            json.dump(avg_metric_dict, json_file, indent=4)






