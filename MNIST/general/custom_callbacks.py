import math
import os
import shutil
import time

import numpy as np
import torch
import helpers
import receptive_field
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from settings import img_size, class_specific, use_l1_mask, n_classes, \
    encoder_out_channels, \
    prototype_shape, num_prototypes, prototype_activation_function, push_start, \
    push_epochs_interval, weight_matrix_filename, prototype_img_filename_prefix, \
    proto_bound_boxes_filename_prefix, prototype_self_act_filename_prefix, \
    save_prototype_class_identity

'''
Some components of the following implementation were obtained from: https://github.com/cfchen-duke/ProtoPNet
 '''

early_stopping_callback = EarlyStopping(
                monitor='avg_total_val_loss',
                min_delta=0.1,
                patience=5,
                verbose=True,
                mode='min'
            )

def getmodel_ckpt_callback(best_model_ckpt_path, params):

    checkpoint_callback = ModelCheckpoint(
        monitor='avg_total_val_loss',
        dirpath=best_model_ckpt_path,  # Directory to save checkpoints
        filename=f"best_model{params['logging_name']}", # Prefix for the
        # checkpoint
        # filenames
        save_top_k=1,  # Save the best model only
        mode='min',
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

    def __init__(self):
        self.proto_rf_boxes = None
        self.proto_bound_boxes = None
        self.proto_epoch_dir = None

    def on_train_epoch_end(self, trainer, pl_module):
        # calculate colulative losses per epoch
        avg_rec_loss, avg_kl_loss, avg_ce_loss, avg_clst_loss, avg_sep_loss, \
            avg_l1_loss, avg_total_loss, avg_acc = helpers.get_average_losses(
            pl_module.training_step_losses, "/train")

        print(f"\nTraining Epoch[{pl_module.current_epoch}]:\n\
           rec_loss: {avg_rec_loss}\n\
           kl_loss: {avg_kl_loss}\n\
           ce_loss: {avg_ce_loss}\n\
           clst_loss: {avg_clst_loss}\n\
           sep_loss: {avg_sep_loss}\n\
           l1_loss: {avg_l1_loss}\n\
           total_loss: {avg_total_loss}\n\
           avg_accuracy: {avg_acc}\n")

        pl_module.training_step_losses.clear()

        if pl_module.current_epoch > push_start and \
                pl_module.current_epoch % push_epochs_interval == 0:

            start = time.time()
            self.proto_epoch_dir = os.path.join(pl_module.prototype_saving_dir,
                                    f'epoch{pl_module.current_epoch}')
            helpers.create_dir(self.proto_epoch_dir)

            prototype_shape = pl_module.prototype_shape
            n_prototypes = pl_module.num_prototypes

            '''
            saves the closest distance seen so far to track if the closest 
            prototype has been found, initialized with infinity for each prototype
            '''
            self.global_min_proto_dist = np.full(n_prototypes, np.inf)

            '''
            saves the latent space patch representation of training data that gives 
            the current smallest distance
            '''
            self.global_min_fmap_patches = np.zeros(
                [n_prototypes,
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
                self.proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                         fill_value=-1)
                self.proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1)
            else:
                self.proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                         fill_value=-1)
                self.proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)

            self.dataloader = trainer.datamodule.train_dataloader()

            self.search_batch_size = self.dataloader.batch_size

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
                                     proto_bound_boxes_filename_prefix + '-receptive_field' + str(
                                         pl_module.current_epoch) + '.npy'),
                        self.proto_rf_boxes)

                # save data corresponding to the bounding boxes
                np.save(os.path.join(self.proto_epoch_dir,
                                     proto_bound_boxes_filename_prefix + str(
                                         pl_module.current_epoch) + '.npy'),
                        self.proto_bound_boxes)

            print('\tExecuting push ...')
            prototype_update = np.reshape(self.global_min_fmap_patches,
                                          tuple(prototype_shape))
            pl_module.prototype_vectors.data.copy_(
                torch.tensor(prototype_update, dtype=torch.float32))

            end = time.time()
            print('\tpush time: \t{0}'.format(end - start))

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
            # print('preprocessing input for pushing ...')
            # search_batch = copy.deepcopy(search_batch_input)
            search_batch = preprocess_input_function(search_batch_input)

        else:
            search_batch = search_batch_input

        '''
        Compute forward upto the bottleneck to get latent space representation 
        and results from _l2_convolution
        '''
        protoL_input_torch, proto_dist_torch = pl_module.push_forward(
            search_batch)

        protoL_input_ = np.copy(protoL_input_torch.numpy())
        proto_dist_ = np.copy(proto_dist_torch.numpy())

        del protoL_input_torch, proto_dist_torch

        if class_specific:
            class_to_img_index_dict = {key: [] for key in range(n_classes)}
            # img_y is the image's integer label
            for img_index, img_y in enumerate(search_y):
                img_label = img_y.item()
                class_to_img_index_dict[img_label].append(img_index)

        prototype_shape = pl_module.prototype_shape
        n_prototypes = prototype_shape[0]
        proto_h = prototype_shape[2]
        proto_w = prototype_shape[3]
        max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

        for j in range(n_prototypes):
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
                proto_dist_j = proto_dist_[:, j, :, :]

            batch_min_proto_dist_j = np.amin(proto_dist_j)
            if batch_min_proto_dist_j < self.global_min_proto_dist[j]:
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
                fmap_height_start_index = batch_argmin_proto_dist_j[
                                              1] * prototype_layer_stride
                fmap_height_end_index = fmap_height_start_index + proto_h
                fmap_width_start_index = batch_argmin_proto_dist_j[
                                             2] * prototype_layer_stride
                fmap_width_end_index = fmap_width_start_index + proto_w

                batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                         :,
                                         fmap_height_start_index:fmap_height_end_index,
                                         fmap_width_start_index:fmap_width_end_index]

                self.global_min_proto_dist[j] = batch_min_proto_dist_j
                self.global_min_fmap_patches[j] = batch_min_fmap_patch_j

                # get the receptive field boundary of the image patch
                # that generates the representation
                layer_filter_sizes, layer_strides, layer_paddings = \
                    pl_module.encoder.conv_info()
                protoL_rf_info = receptive_field.compute_proto_layer_rf_info_v2(img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])

                rf_prototype_j = receptive_field.compute_rf_prototype(search_batch.size(2),
                                                      batch_argmin_proto_dist_j,
                                                      protoL_rf_info)

                # get the whole image
                original_img_j = search_batch_input[rf_prototype_j[0]]
                original_img_j = original_img_j.numpy()
                original_img_j = np.transpose(original_img_j, (1, 2, 0))
                original_img_size = original_img_j.shape[0]

                # crop out the receptive field
                rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                           rf_prototype_j[3]:rf_prototype_j[4], :]

                # save the prototype receptive field information
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
                if prototype_activation_function == 'log':
                    proto_act_img_j = np.log((proto_dist_img_j + 1) / (
                                proto_dist_img_j + pl_module.epsilon))
                elif prototype_activation_function == 'linear':
                    proto_act_img_j = max_dist - proto_dist_img_j
                else:
                    proto_act_img_j = prototype_activation_function_in_numpy(
                        proto_dist_img_j)
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
                                                 j) + '.npy'),
                                proto_act_img_j)
                    if prototype_img_filename_prefix is not None:
                        # save the whole image containing the prototype as png
                        plt.imsave(os.path.join(self.proto_epoch_dir,
                                                prototype_img_filename_prefix + '-original' + str(
                                                    j) + '.png'),
                                   original_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        # overlay (upsampled) self activation on original image and save the result
                        rescaled_act_img_j = upsampled_act_img_j - np.amin(
                            upsampled_act_img_j)
                        rescaled_act_img_j = rescaled_act_img_j / np.amax(
                            rescaled_act_img_j)
                        heatmap = cv2.applyColorMap(
                            np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
                        heatmap = np.float32(heatmap) / 255
                        heatmap = heatmap[..., ::-1]
                        overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                        plt.imsave(os.path.join(self.proto_epoch_dir,
                                                prototype_img_filename_prefix + '-original_with_self_act' + str(
                                                    j) + '.png'),
                                   overlayed_original_img_j,
                                   vmin=0.0,
                                   vmax=1.0)

                        # if different from the original (whole) image, save the prototype receptive field as png
                        if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[
                            1] != original_img_size:
                            plt.imsave(os.path.join(self.proto_epoch_dir,
                                                    prototype_img_filename_prefix + '-receptive_field' + str(
                                                        j) + '.png'),
                                       rf_img_j,
                                       vmin=0.0,
                                       vmax=1.0)
                            overlayed_rf_img_j = overlayed_original_img_j[
                                                 rf_prototype_j[1]:rf_prototype_j[2],
                                                 rf_prototype_j[3]:rf_prototype_j[4]]
                            plt.imsave(os.path.join(self.proto_epoch_dir,
                                                    prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(
                                                        j) + '.png'),
                                       overlayed_rf_img_j,
                                       vmin=0.0,
                                       vmax=1.0)

                        # save the prototype image (highly activated region of the whole image)
                        plt.imsave(os.path.join(self.proto_epoch_dir,
                                                prototype_img_filename_prefix + str(
                                                    j) + '.png'),
                                   proto_img_j,
                                   vmin=0.0,
                                   vmax=1.0)

        if class_specific:
            del class_to_img_index_dict

