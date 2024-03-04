img_size = 224
n_classes = 200
val_ratio = 0.2
prototype_shape = (2000, 128, 1, 1)
latent_dim = (128, 7, 7)
num_prototypes = prototype_shape[0]
encoder_out_channels = 256
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

data_path = './data/cub200_cropped/'
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'

class_specific = True
use_l1_mask = True

vae_only = 30
num_train_epochs = 1000
num_warm_epochs = 5
push_start = 10
push_epochs_interval = 10

weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'
save_prototype_class_identity=True


