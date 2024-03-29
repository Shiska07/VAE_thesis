img_size = 28
n_classes = 10
val_ratio = 0.2

prototype_shape = (40, 16, 1, 1)
num_prototypes = prototype_shape[0]
encoder_out_channels = 32
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

class_specific = True
use_l1_mask = True

vae_only = 2
push_start = 4
push_epochs_interval = 3

weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'
save_prototype_class_identity=True


