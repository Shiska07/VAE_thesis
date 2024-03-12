# optimizer config for different stages
warm_vae_optimizer_lrs = {'encoder': 3e-3,
                'decoder': 3e-3,}

warm_protoL_optimizer_lrs = {'prototype_vectors': 3e-3}

joint_optimizer_lrs = {'features': 1e-4,
                       'encoder': 3e-3,
                       'decoder': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

last_layer_optimizer_lr = 1e-4