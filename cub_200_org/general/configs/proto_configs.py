# configuration specific to prototypes
input_height = 224
input_channels = 1
n_classes = 200
prototype_shape = (2000, 128, 1, 1)
latent_dim = (128, 7, 7)
latent_channels = latent_dim[0]
num_prototypes = prototype_shape[0]
encoder_out_channels = 256
prototype_activation_function = 'log'

weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'
