# epoch configuration for different stages
max_epochs_dict = {}
push_start = 0
push_epochs_interval = 10

def set_epoch_configs(params):

    global max_epochs_dict
    global push_start
    global push_epochs_interval

    max_epochs_dict["wanm_vae"] = params["vae_train_epochs"]
    max_epochs_dict["warm_proto"] = params["protoL_train_epochs"]
    max_epochs_dict["joint"] = params["joint_train_epochs"]
    max_epochs_dict["last_layer"] = params["last_layer_train_epochs"]
    push_epochs_interval = params["push_epochs_interval"]
    push_start = params["vae_train_epochs"] + push_epochs_interval
