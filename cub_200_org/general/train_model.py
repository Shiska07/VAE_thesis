import os
import cv2
import torch
import numpy as np
import pytorch_lightning as pl
from model import PartProtoVAE
from utils import receptive_field
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from configs.train_settings import use_l1_mask, class_specific, \
    save_prototype_class_identity
from utils.helpers import create_dir, clear_prev_logs, normalize_array_0_1, \
    save_model_specs_txt
from custom_callbacks import getmodel_ckpt_callback, \
    get_earlystopping_callback
from configs.epoch_configs import max_epochs_dict, push_start, push_epochs_interval
from configs.proto_configs import num_prototypes, prototype_shape, \
     proto_bound_boxes_filename_prefix, latent_dim, n_classes, \
    input_height, prototype_activation_function, prototype_self_act_filename_prefix, \
    prototype_img_filename_prefix


'''
Here we would do the prototype projection depending on the epoch.
Some of the code has been obtained from ProtoPNet's impementation on github.
'''
class TrainingPipiline:
    def __init__(self, params, datamodule, prev_model_ckpt):

        self.params = params
        self.datamodule = datamodule

        if prev_model_ckpt is not None:
            self.model = PartProtoVAE.load_from_checkpoint(prev_model_ckpt)
        else:
            self.model = PartProtoVAE(params['base_architecture'],
                                        params['layers_to_exclude'],
                                        params['prototype_saving_dir'],
                                        params['logging_dir'],
                                        params['default_lr']
                                        )

        # save model architecture as  a txt file
        save_model_specs_txt(self.model, params['logging_name'], latent_dim,
                             num_prototypes)

        # save dataloaders
        self.train_dataloader = self.datamodule.train_dataloader()
        self.train_push_dataloader = self.datamodule.train_push_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.test_dataloader = self.datamodule.test_dataloader()

        self.trainer = None
        self.tb_logger = None
        self.checkpoint_callback = None

        # clear previous logs
        self.root_logging_dir = os.path.join(params['logging_dir'], params[
            'logging_name'])
        clear_prev_logs(self.root_logging_dir)

        # root checkpoint directory
        self.root_ckpt_path = os.path.join(self.root_logging_dir, 'checkpoints')
        create_dir(self.root_ckpt_path)

        # path to store the most recent checkpoint for testing
        self.last_ckpt_path = None

        # prototype dir info
        self.prototype_save_dir = os.path.join(self.params["prototype_saving_dir"],
                                               self.params["logging_name"])

        # counter to track training epoch for entire training pipeline
        self.global_epoch = 0


    def load_model_from_checkpoint(self, mode):

        if mode == "test":
            ckpt_path = self.last_ckpt_path
        else:
            ckpt_path = os.path.join(self.root_ckpt_path, mode, 'best_model.ckpt')
        self.model = PartProtoVAE.load_from_checkpoint(ckpt_path)

        print(f'Model for mode {mode} loaded from {self.root_ckpt_path}')


    def initalize_trainer(self, mode, custom_callbacks):
        stage_logging_dir = os.path.join(self.root_logging_dir, mode)
        create_dir(stage_logging_dir)
        self.tb_logger = TensorBoardLogger(stage_logging_dir,
                                           name=self.params[
            'logging_name'], log_graph=False)

        self.trainer = pl.Trainer(
            accelerator=self.params['accelerator'],
            strategy=self.params['strategy'],
            enable_progress_bar=True,
            check_val_every_n_epoch=1,
            enable_checkpointing=True,
            callbacks=custom_callbacks,
            max_epochs=max_epochs_dict[mode],
            logger=self.tb_logger,
            limit_val_batches=1,
            limit_train_batches=1,
            limit_test_batches=1,
            log_every_n_steps=1
        )

        print(f"Trainer initialization for stage '{mode}' complete.")


    def fit_warm_vae(self):
        # initialize checkpoint callback
        curr_ckpt_path = os.path.join(self.root_ckpt_path, "warm_vae")
        create_dir(curr_ckpt_path)
        checkpoint_callback = getmodel_ckpt_callback(curr_ckpt_path)
        early_stoppping_callback = get_earlystopping_callback(min_delta=0.1,
                                                              patience=5)
        custom_callbacks = [early_stoppping_callback, checkpoint_callback]

        # configure model and initialize trainer
        self.model.set_mode("warm_vae")
        self.initalize_trainer("warm_vae", custom_callbacks)

        print('BEGIN training custom vae layers.\n')
        self.trainer.fit(self.model,
                         train_dataloaders=self.train_dataloader,
                         val_dataloaders=self.val_dataloader,
                         )
        self.global_epoch += max_epochs_dict["warm_vae"]
        print('END training custom vae layers.\n')


    def fit_warm_proto(self):
        # initialize checkpoint callback
        curr_ckpt_path = os.path.join(self.root_ckpt_path, "warm_proto")
        create_dir(curr_ckpt_path)
        checkpoint_callback = getmodel_ckpt_callback(curr_ckpt_path)

        early_stoppping_callback = get_earlystopping_callback(min_delta=0.1,
                                                              patience=5)
        custom_callbacks = [early_stoppping_callback, checkpoint_callback]

        # configure model and initialize traine
        self.load_model_from_checkpoint("warm_vae")
        self.model.set_mode("warm_proto")
        self.initalize_trainer("warm_proto", custom_callbacks)

        print('BEGIN training prototype layers.\n')
        self.trainer.fit(self.model,
                         train_dataloaders=self.train_dataloader,
                         val_dataloaders=self.val_dataloader,
                         )
        self.global_epoch += max_epochs_dict["warm_proto"]
        print('END training prototype layers.\n')


    def fit_joint(self):
        # initialize checkpoint callback
        curr_ckpt_path = os.path.join(self.root_ckpt_path, "joint")
        create_dir(curr_ckpt_path)

        checkpoint_callback = getmodel_ckpt_callback(curr_ckpt_path)
        early_stoppping_callback = get_earlystopping_callback(min_delta=0.1,
                                                              patience=5)
        custom_callbacks = [early_stoppping_callback, checkpoint_callback]

        # configure model and initialize traine
        self.load_model_from_checkpoint("warm_proto")
        self.model.set_mode("joint")
        self.initalize_trainer("joint", custom_callbacks)

        # set trainer parameters for current stage
        print('BEGIN training joint layers.\n')
        self.trainer.fit(self.model,
                         train_dataloaders=self.train_dataloader,
                         val_dataloaders=self.val_dataloader,
                         test_dataloaders=self.test_dataloader
                         )
        self.global_epoch += max_epochs_dict["joint"]
        print('END training joint layers.\n')


    def fit_last_layer(self):
        # initialize checkpoint callback
        curr_ckpt_path = os.path.join(self.root_ckpt_path, "last_layer")
        create_dir(curr_ckpt_path)
        checkpoint_callback = getmodel_ckpt_callback(curr_ckpt_path)
        early_stoppping_callback = get_earlystopping_callback(min_delta=0.1,
                                                              patience=5)
        custom_callbacks = [early_stoppping_callback, checkpoint_callback]

        '''
        Probably don't reinitialize trainer between these steps as the model needs
        to be tested before and after prototype projection as well as after convex 
        optimization fo last layer.
        '''
        # configure model and initialize traine
        self.load_model_from_checkpoint("joint")
        self.model.set_mode("last_layer")
        self.initalize_trainer("last_layer", custom_callbacks)

        # set trainer parameters for current stage
        print('BEGIN training last layer.\n')
        self.trainer.fit(self.model,
                         train_dataloaders=self.train_dataloader,
                         val_dataloaders=self.val_dataloader,
                         test_dataloaders=self.test_dataloader
                         )
        self.global_epoch += max_epochs_dict["last_layer"]
        print('END training last layer.\n')


    def test_model(self, ckpt_path = None):

        if ckpt_path is not None:
            self.last_ckpt_path = ckpt_path
            self.load_model_from_checkpoint("test")
        self.trainer.test(self.model, self.test_dataloader)

    def get_model(self):
        return self.model

    def push_prototypes(self):
        self.proto_epoch_dir = os.path.join(self.prototype_save_dir,
                                            f"epoch{self.global_epoch}")
        create_dir(self.proto_epoch_dir)

        # load model
        self.load_model_from_checkpoint("push")

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

        self.search_batch_size = self.train_push_dataloader.batch_size
        num_batches = len(self.train_push_dataloader)

        for push_iter, (search_batch_input, search_y) in enumerate(
                self.train_push_dataloader):
            '''
            start_index_of_search keeps track of the index of the image
            assigned to serve as prototype
            '''
            start_index_of_search_batch = push_iter * self.search_batch_size

            self.update_prototypes_on_batch(search_batch_input,
                                            search_y,
                                            start_index_of_search_batch)

        if self.proto_epoch_dir is not None and proto_bound_boxes_filename_prefix \
                is not None:
            # save data corresponding to the receptive field
            np.save(os.path.join(self.proto_epoch_dir,
                                 proto_bound_boxes_filename_prefix + "-receptive_field" + str(
                                     self.global_epoch) + ".npy"),
                    self.proto_rf_boxes)

            # save data corresponding to the bounding boxes
            np.save(os.path.join(self.proto_epoch_dir,
                                 proto_bound_boxes_filename_prefix + str(
                                     self.global_epoch) + ".npy"),
                    self.proto_bound_boxes)

        print(f"\tExecuting push ...EPOCH {self.global_epoch}")
        prototype_update = np.reshape(self.global_min_fmap_patches,
                                      tuple(prototype_shape))
        self.model.prototype_vectors.data.copy_(
            torch.tensor(prototype_update, dtype=torch.float32))


    def update_prototypes_on_batch(self,
                                   search_batch_input,
                                   search_y,
                                   start_index_of_search_batch,
                                   prototype_activation_function_in_numpy=None,
                                   preprocess_input_function=None,
                                   prototype_layer_stride = 1):


        self.model.eval()

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
        with torch.no_grad():
            search_batch = search_batch.cuda()
            # this computation currently is not parallelized
            _, _, protoL_input_torch, proto_dist_torch = self.model.push_forward(
                search_batch)

        # make sure values are between 0 and 1
        protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
        proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

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
                    self.model.prototype_class_identity[
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
                    self.model.encoder.conv_info()

                '''
                Compute receptive field at prototype layer
                '''
                protoL_rf_info = receptive_field.compute_proto_layer_rf_info_v2(input_height,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])

                '''
                Using the network's receptive field, find the corresponding
                spatial indices for cropping in image space. [y1, y1, x1, x2]
                '''
                rf_prototype_j = receptive_field.compute_rf_prototype(input_height,
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
                                proto_dist_img_j + self.model.epsilon))
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

                            plt.imsave(os.path.join(self.proto_epoch_dir,
                                                    prototype_img_filename_prefix + "-receptive_field" + str(
                                                        j) + ".png"),
                                       rf_img_j,
                                       cmap = "gray")

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
                        plt.imsave(os.path.join(self.proto_epoch_dir,
                                                prototype_img_filename_prefix + str(
                                                    j) + ".png"),
                                   proto_img_j,
                                   vmin=0.0,
                                   vmax=1.0)

        if class_specific:
            del class_to_img_index_dict






