import os
import pdb
import time
import math
import shutil

import cv2
import torch
import helpers
import numpy as np
from PIL import Image
import receptive_field
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from settings import img_size, class_specific, use_l1_mask, n_classes, \
    encoder_out_channels, \
    prototype_shape, num_prototypes, prototype_activation_function, push_start, \
    push_epochs_interval, weight_matrix_filename, prototype_img_filename_prefix, \
    proto_bound_boxes_filename_prefix, prototype_self_act_filename_prefix, \
    save_prototype_class_identity, vae_only

'''
Some components of the following implementation were obtained from: https://github.com/cfchen-duke/ProtoPNet
 '''

early_stopping_callback = EarlyStopping(
                monitor="val_total_loss",
                min_delta=0.1,
                patience=5,
                verbose=True,
                mode="min"
            )

def getmodel_ckpt_callback(best_model_ckpt_path, params):

    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_loss",
        dirpath=best_model_ckpt_path,  # Directory to save checkpoints
        filename=f"best_model{params['logging_name']}", # Prefix for the
        # checkpoint
        # filenames
        save_top_k=1,  # Save the best model only
        mode="min",
        every_n_epochs=1
    )

    return checkpoint_callback

# clearing logs with the same name
def clear_prev_logs(dir_path):
    if os.path.exists(dir_path):
        # If it does, delete its contents
        shutil.rmtree(dir_path)
        print(f"Previous logs in {dir_path} have been cleared.")


'''
Here we would do the prototype projection depending on the epoch.
Some of the code has been obtained from ProtoPNet's impementation on github.
'''
class PrototypeProjectionCallback(pl.Callback):

    def __init__(self, params):
        self.proto_rf_boxes = None
        self.proto_bound_boxes = None
        self.proto_epoch_dir = None

        self.prototype_save_dir = None

        self.params = params

    def on_train_epoch_end(self, trainer, pl_module):

        self.prototype_save_dir = os.path.join(self.params["prototype_saving_dir"],
                                               self.params["logging_name"])

        # calculate colulative losses per epoch
        avg_metric_dict = helpers.get_average_losses(
            pl_module.training_step_losses)

        tag = "train"
        print(f"\nTRAINING Epoch[{pl_module.current_epoch}]:")
        for key, val in avg_metric_dict.items():
            print(f"{key} : {val:0.4f}")
            pl_module.logger.experiment.add_scalars(key, {tag: val},
                                                    pl_module.current_epoch)
        print("\n")
        pl_module.training_step_losses.clear()

        if pl_module.current_epoch == "vae_only":
            pl_module.set_mode("joint")

        # if last training mode was convex optimization, reset mode ot joint training
        if pl_module.mode == "last_only":
            pl_module.set_mode("joint")


        if pl_module.current_epoch > push_start and \
                (pl_module.current_epoch - pl_module.last_push_epoch) >= \
                push_epochs_interval:

            self.proto_epoch_dir = os.path.join(self.prototype_save_dir,
                                    f"epoch{pl_module.current_epoch}")
            helpers.create_dir(self.proto_epoch_dir)

            '''
            saves the closest distance seen so far to track if the closest
            prototype has been found, initialized with infinity for each prototype
            '''
            self.global_min_proto_dist = np.full(num_prototypes, np.inf)

            '''
            saves the latent space patch representation of training data that gives
            the current smallest distance
            '''
            self.global_min_fmap_patches = np.zeros(
                [num_prototypes,
                 prototype_shape[1],
                 prototype_shape[2],
                 prototype_shape[3]])

            '''
             proto_rf_boxes and proto_bound_boxes column:
             0: image index in the entire dataset
             1: height start index
             2: height end index
             3: width start index
             4: width end index
             5: (optional) class identity
             '''

            if save_prototype_class_identity:
                self.proto_rf_boxes = np.full(shape=[num_prototypes, 6],
                                         fill_value=-1)
                self.proto_bound_boxes = np.full(shape=[num_prototypes, 6],
                                            fill_value=-1)
            else:
                self.proto_rf_boxes = np.full(shape=[num_prototypes, 5],
                                         fill_value=-1)
                self.proto_bound_boxes = np.full(shape=[num_prototypes, 5],
                                            fill_value=-1)


            self.dataloader = trainer.datamodule.train_dataloader()
            self.test_dataloader = trainer.datamodule.test_dataloader()

            # # log test losses before push
            # pl_module.test_tag = "pre_push_test"
            # print(f"TEST PRE-PUSH FOR EPOCH {pl_module.current_epoch}:\n")
            # trainer.test(pl_module, self.test_dataloader)

            self.search_batch_size = self.dataloader.batch_size
            num_batches = len(self.dataloader)
            '''
            DO THIS FOR EACH BATCH
            '''
            for push_iter, (search_batch_input, search_y) in enumerate(
                    self.dataloader):
                '''
                start_index_of_search keeps track of the index of the image
                assigned to serve as prototype
                '''
                start_index_of_search_batch = push_iter * self.search_batch_size

                self.update_prototypes_on_batch(trainer,
                                                pl_module,
                                                search_batch_input,
                                                search_y,
                                                start_index_of_search_batch)

            if self.proto_epoch_dir is not None and proto_bound_boxes_filename_prefix \
                    is not None:
                # save data corresponding to the receptive field
                np.save(os.path.join(self.proto_epoch_dir,
                                     proto_bound_boxes_filename_prefix + "-receptive_field" + str(
                                         pl_module.current_epoch) + ".npy"),
                        self.proto_rf_boxes)

                # save data corresponding to the bounding boxes
                np.save(os.path.join(self.proto_epoch_dir,
                                     proto_bound_boxes_filename_prefix + str(
                                         pl_module.current_epoch) + ".npy"),
                        self.proto_bound_boxes)

            print(f"\tExecuting push ...EPOCH {pl_module.current_epoch}")
            prototype_update = np.reshape(self.global_min_fmap_patches,
                                          tuple(prototype_shape))
            pl_module.prototype_vectors.data.copy_(
                torch.tensor(prototype_update, dtype=torch.float32))

            # note push epoch
            pl_module.last_push_epoch = pl_module.current_epoch

            # # get post push test results
            # pl_module.test_tag = "post_push_test"
            # print(f"POST-PUSH TEST FOR EPOCH {pl_module.current_epoch}:\n")
            # trainer.test(pl_module, self.test_dataloader)
            # pl_module.test_tag = "test"


            # set mode to convex optimization
            # pl_module.set_mode("last_only")


    def update_prototypes_on_batch(self,
                                   trainer,
                                   pl_module,
                                   search_batch_input,
                                   search_y,
                                   start_index_of_search_batch,
                                   prototype_activation_function_in_numpy=None,
                                   preprocess_input_function=None,
                                   prototype_layer_stride=1):

        # preprocess batch if necessary
        if preprocess_input_function is not None:
            # print("preprocessing input for pushing ...")
            # search_batch = copy.deepcopy(search_batch_input)
            search_batch = preprocess_input_function(search_batch_input)

        else:
            search_batch = search_batch_input

        '''
        Compute forward upto the bottleneck to get latent space representation
        and results from _l2_convolution
        '''

        _, _, protoL_input_torch, proto_dist_torch = pl_module.push_forward(
            search_batch)

        # make sure values are between 0 and 1
        protoL_input_torch = torch.sigmoid(protoL_input_torch)
        protoL_input_ = np.copy(protoL_input_torch.detach().numpy())
        proto_dist_ = np.copy(proto_dist_torch.detach().numpy())

        del protoL_input_torch, proto_dist_torch

        if class_specific:
            class_to_img_index_dict = {key: [] for key in range(n_classes)}
            # img_y is the image's integer label
            for img_index, img_y in enumerate(search_y):
                img_label = img_y.item()
                '''
                dinctionary containing indices of images belonging to each class in a
                list.
                class label is the key and the corresponding list is the value.
                '''
                class_to_img_index_dict[img_label].append(img_index)

        proto_h = prototype_shape[2]
        proto_w = prototype_shape[3]
        max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

        for j in range(num_prototypes):
            # if n_prototypes_per_class != None:
            if class_specific:
                # target_class is the class of the class_specific prototype
                target_class = torch.argmax(
                    pl_module.prototype_class_identity[
                        j]).item()
                # if there is not images of the target_class from this batch
                # we go on to the next prototype
                if len(class_to_img_index_dict[target_class]) == 0:
                    continue
                proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,
                               j, :, :]
            else:
                # if it is not class specific, then we will search through
                # every example
                '''
                Get min_distances values for the batch with the jtj prototype but only with images
                belonging to prototype j's class identity
                proto_dist_.shape = (batch_size, num_prototypes, 7, 7)
                proto_dist_j = (n, 7, 7) where n = number of images beloging to
                prototype j's class identity.
                '''
                proto_dist_j = proto_dist_[:, j, :, :]

            '''
            Returns the minimum value in the entire distance array i.e.
            distance with
            the closest training patch
            '''
            batch_min_proto_dist_j = np.amin(proto_dist_j)
            if batch_min_proto_dist_j < self.global_min_proto_dist[j]:
                '''
                If the distance found is the smalles so far, find the 3D index at
                which the value exists
                '''
                batch_argmin_proto_dist_j = \
                    list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                          proto_dist_j.shape))
                if class_specific:
                    '''
                    change the argmin index from the index among
                    images of the target class to the index in the entire search
                    batch
                    '''
                    batch_argmin_proto_dist_j[0] = \
                    class_to_img_index_dict[target_class][
                        batch_argmin_proto_dist_j[0]]

                # retrieve the corresponding feature map patch
                img_index_in_batch = batch_argmin_proto_dist_j[0]

                '''
                Get index information for extracting closest training patch.
                '''
                fmap_height_start_index = batch_argmin_proto_dist_j[
                                              1] * prototype_layer_stride
                fmap_height_end_index = fmap_height_start_index + proto_h
                fmap_width_start_index = batch_argmin_proto_dist_j[
                                             2] * prototype_layer_stride
                fmap_width_end_index = fmap_width_start_index + proto_w

                '''
                Extract patch from the closeset training image.
                '''
                batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                         :,
                                         fmap_height_start_index:fmap_height_end_index,
                                         fmap_width_start_index:fmap_width_end_index]

                # all index info of the closest training patch
                self.global_min_proto_dist[j] = batch_min_proto_dist_j

                # value of the closest traning patch
                self.global_min_fmap_patches[j] = batch_min_fmap_patch_j


                '''
                This part uses the receptive field information to generate
                visualization in the pixel space.
                '''
                # get the receptive field boundary of the image patch
                # that generates the representation
                layer_filter_sizes, layer_strides, layer_paddings = \
                    pl_module.encoder.conv_info()

                '''
                Compute receptive field at prototype layer
                '''
                protoL_rf_info = receptive_field.compute_proto_layer_rf_info_v2(img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])

                '''
                Using the network's receptive field, find the corresponding
                spatial indices for cropping in image space. [y1, y1, x1, x2]
                '''
                rf_prototype_j = receptive_field.compute_rf_prototype(img_size,
                                                      batch_argmin_proto_dist_j,
                                                      protoL_rf_info)

                # get the whole image
                original_img_j = search_batch_input[rf_prototype_j[0]]
                original_img_j = original_img_j.numpy()

                '''
                original shape is (channels, height, width)
                transpose to (height, width, channels)
                '''
                original_img_j = np.transpose(original_img_j, (1, 2, 0))
                original_img_size = original_img_j.shape[0]

                # crop out the receptive field covered by th prototype in the
                # original image
                rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                           rf_prototype_j[3]:rf_prototype_j[4], :]

                '''
                save the prototype receptive field information
                proto_rf_boxes and proto_bound_boxes column:
                0: image index in the entire dataset
                1: height start index
                2: height end index
                3: width start index
                4: width end index
                5: (optional) class identity
                '''
                self.proto_rf_boxes[j, 0] = rf_prototype_j[
                                           0] + start_index_of_search_batch
                self.proto_rf_boxes[j, 1] = rf_prototype_j[1]
                self.proto_rf_boxes[j, 2] = rf_prototype_j[2]
                self.proto_rf_boxes[j, 3] = rf_prototype_j[3]
                self.proto_rf_boxes[j, 4] = rf_prototype_j[4]
                if self.proto_rf_boxes.shape[1] == 6 and search_y is not None:
                    self.proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

                # find the highly activated region of the original image
                proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]


                # apply activation function to distance for visualization
                if prototype_activation_function == "log":
                    proto_act_img_j = np.log((proto_dist_img_j + 1) / (
                                proto_dist_img_j + pl_module.epsilon))
                elif prototype_activation_function == "linear":
                    proto_act_img_j = max_dist - proto_dist_img_j
                else:
                    proto_act_img_j = prototype_activation_function_in_numpy(
                        proto_dist_img_j)

                # upsample the activation map
                upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(
                original_img_size, original_img_size),
                                                 interpolation=cv2.INTER_CUBIC)
                proto_bound_j = receptive_field.find_high_activation_crop(
                    upsampled_act_img_j)
                # crop out the image patch with high activation as prototype image
                proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                              proto_bound_j[2]:proto_bound_j[3], :]

                # save the prototype boundary (rectangular boundary of highly activated region)
                self.proto_bound_boxes[j, 0] = self.proto_rf_boxes[j, 0]
                self.proto_bound_boxes[j, 1] = proto_bound_j[0]
                self.proto_bound_boxes[j, 2] = proto_bound_j[1]
                self.proto_bound_boxes[j, 3] = proto_bound_j[2]
                self.proto_bound_boxes[j, 4] = proto_bound_j[3]
                if self.proto_bound_boxes.shape[1] == 6 and search_y is not None:
                    self.proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()


                if self.proto_epoch_dir is not None:
                    if prototype_self_act_filename_prefix is not None:
                        # save the numpy array of the prototype self activation

                        np.save(os.path.join(self.proto_epoch_dir,
                                             prototype_self_act_filename_prefix + str(
                                                 j) + ".npy"),
                                proto_act_img_j)

                    if prototype_img_filename_prefix is not None:

                        '''
                        1. SAVE original image
                        '''
                        #save the whole image containing the prototype as png
                        original_img_j_norm = helpers.normalize_array_0_1(
                            original_img_j)
                        original_img_j_pil = Image.fromarray(np.squeeze(
                            original_img_j_norm*255).astype("uint8"), "L")
                        original_img_j_pil.save(os.path.join(self.proto_epoch_dir,
                                                prototype_img_filename_prefix + "-original" + str(
                                                    j) + ".png"))
                        # plt.imsave(os.path.join(self.proto_epoch_dir,
                        #                         prototype_img_filename_prefix + "-original" + str(
                        #                             j) + ".png"),
                        #            original_img_j,
                        #            cmap = "gray")
                        # overlay (upsampled) self activation on original image and save the result
                        rescaled_act_img_j = upsampled_act_img_j - np.amin(
                            upsampled_act_img_j)
                        rescaled_act_img_j = rescaled_act_img_j / np.amax(
                            rescaled_act_img_j)
                        heatmap = cv2.applyColorMap(
                            np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
                        heatmap = np.float32(heatmap) / 255
                        heatmap = heatmap[..., ::-1]
                        overlayed_original_img_j = 0.5 * original_img_j_norm + \
                                                   0.3 * heatmap

                        '''
                        2. SAVE original image overlayed with activation map
                        '''

                        plt.imsave(os.path.join(self.proto_epoch_dir,
                                                prototype_img_filename_prefix + "-original_with_self_act" + str(
                                                    j) + ".png"),
                                   overlayed_original_img_j,
                                   vmin=0.0,
                                   vmax=1.0)

                        '''
                        3. SAVE image corresponding to the prototype's
                        receptive field
                        '''
                        # if different from the original (whole) image, save the prototype receptive field as png
                        if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[
                            1] != original_img_size:

                            # normalize
                            rf_img_j_norm = helpers.normalize_array_0_1(rf_img_j)
                            rf_img_j_pil = Image.fromarray(np.squeeze(
                                rf_img_j_norm*255).astype("uint8"), "L")
                            rf_img_j_pil.save(os.path.join(self.proto_epoch_dir,
                                                    prototype_img_filename_prefix + "-receptive_field" + str(
                                                        j) + ".png"))

                            # plt.imsave(os.path.join(self.proto_epoch_dir,
                            #                         prototype_img_filename_prefix + "-receptive_field" + str(
                            #                             j) + ".png"),
                            #            rf_img_j,
                            #            cmap = "gray")

                            '''
                            4. SAVE image corresponding to the prototype's
                            receptive field overlayed with activation map
                            '''
                            overlayed_rf_img_j = overlayed_original_img_j[
                                                 rf_prototype_j[1]:rf_prototype_j[2],
                                                 rf_prototype_j[3]:rf_prototype_j[4]]
                            plt.imsave(os.path.join(self.proto_epoch_dir,
                                                    prototype_img_filename_prefix + "-receptive_field_with_self_act" + str(
                                                        j) + ".png"),
                                       overlayed_rf_img_j,
                                       vmin=0.0,
                                       vmax=1.0)

                        # save the prototype image (highly activated region of the whole image)
                        '''
                        5. SAVE prototype image corresponding to the highly
                        activated region.
                        '''
                        proto_img_j_norm = helpers.normalize_array_0_1(
                            proto_img_j)
                        proto_img_j_pil = Image.fromarray(np.squeeze(
                            proto_img_j_norm*255).astype("uint8"), "L")
                        proto_img_j_pil.save(os.path.join(self.proto_epoch_dir,
                                                prototype_img_filename_prefix + str(
                                                    j) + ".png"))

                        # plt.imsave(os.path.join(self.proto_epoch_dir,
                        #                         prototype_img_filename_prefix + str(
                        #                             j) + ".png"),
                        #            proto_img_j,
                        #            vmin=0.0,
                        #            vmax=1.0)

        if class_specific:
            del class_to_img_index_dict

