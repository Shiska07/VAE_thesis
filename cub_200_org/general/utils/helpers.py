import os
import sys
import json
import torch
import shutil
import numpy as np
import torchsummary
from os import makedirs
from os.path import exists


def get_average_losses(losses_list):

    keys_list = list(losses_list[0].keys())
    num_items = len(losses_list)
    avg_metric_dict = {}

    # initialize average metrict dict wih zero
    for key in keys_list:
        avg_metric_dict[key] = 0

    # sum loss values
    for logs_dict in losses_list:
        for key, val in logs_dict.items():
            avg_metric_dict[key] += val

    # average values for number of items
    for key, value in avg_metric_dict.items():
        avg_metric_dict[key] = value/num_items

    return avg_metric_dict


def get_logs(mode, step_losses, class_specific = True):

    logs = {}

    """
    loss order:
    0. recon loss
    1. kl_loss
    2. ce_loss
    3. clst_loss
    4. sep_loss
    5. l1_loss
    6. total_loss
    7. acc
    """

    if mode == "warm_vae":
        logs["recon_loss"] = step_losses[0]
        logs["kl_loss"] = step_losses[1]
        logs["total_loss"] = step_losses[6]
        return logs

    if class_specific:
        if mode == "warm_proto":
            logs["ce_loss"] = step_losses[2]
            logs["clst_loss"] = step_losses[3]
            logs["sep_loss"] = step_losses[4]
            logs["total_loss"] = step_losses[6]
            logs["acc"] = step_losses[7]
            return logs

        elif mode == "joint":
            logs["recon_loss"] = step_losses[0]
            logs["kl_loss"] = step_losses[1]
            logs["ce_loss"] = step_losses[2]
            logs["clst_loss"] = step_losses[3]
            logs["sep_loss"] = step_losses[4]
            logs["total_loss"] = step_losses[6]
            logs["acc"] = step_losses[7]
            return logs

        elif mode == "last_only":
            logs["ce_loss"] = step_losses[2]
            logs["l1_loss"] = step_losses[5]
            logs["total_loss"] = step_losses[6]
            logs["acc"] = step_losses[7]
            return logs

    else:

        """
        if not class specific sep cost is notcalculated
        loss order:
        0. recon loss
        1. kl_loss
        2. ce_loss
        3. clst_loss
        4. l1_loss
        5. total_loss
        6. acc
        """

        if mode == "warm_proto":
            logs["ce_loss"] = step_losses[2]
            logs["clst_loss"] = step_losses[3]
            logs["total_loss"] = step_losses[5]
            logs["acc"] = step_losses[6]
            return logs

        elif mode == "joint":
            logs["recon_loss"] = step_losses[0]
            logs["kl_loss"] = step_losses[1]
            logs["ce_loss"] = step_losses[2]
            logs["clst_loss"] = step_losses[3]
            logs["total_loss"] = step_losses[5]
            logs["acc"] = step_losses[6]
            return logs

        elif mode == "last_only":
            logs["ce_loss"] = step_losses[2]
            logs["l1_loss"] = step_losses[5]
            logs["total_loss"] = step_losses[5]
            logs["acc"] = step_losses[6]
            return logs


def get_accuracy(logits, targets):

    probabilities = torch.softmax(logits, dim=1)

    # Get the predicted class labels
    predicted_labels = torch.argmax(probabilities, dim=1)

    # Calculate the accuracy
    correct_predictions = (predicted_labels == targets).sum().item()
    total_samples = targets.size(0)
    accuracy = correct_predictions / total_samples

    return accuracy

def normalize_array_0_1(arr):

    # Compute the minimum and maximum values
    min_value = np.min(arr)
    max_value = np.max(arr)

    if min_value == max_value:
        normalized_arr = np.full_like(arr, 0.5)
    else:
        # Normalize the array between 0 and 1
        normalized_arr = (arr - min_value) / (max_value - min_value)

    return normalized_arr

def create_dir(directory, verbose=False):
    if not exists(directory):
        makedirs(directory)
    if verbose:
        print(f"directory created: {directory}")

def load_parameters(json_file):
    try:
        with open(json_file, 'r') as file:
            parameters = json.load(file)
        return parameters
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: JSON file '{json_file}' is not a valid JSON file.")
        return None


# clearing logs with the same name
def clear_prev_logs(dir_path):
    if os.path.exists(dir_path):
        # If it does, delete its contents
        shutil.rmtree(dir_path)
        print(f"Previous logs in {dir_path} have been cleared.")


def save_model_specs_txt(model, logging_name, latent_dim, num_prototypes):
    encoder_input_dim = (model.input_channels, model.input_height,
                         model.input_height)
    decoder_input_dim = latent_dim
    prototype_input_dim = latent_dim
    prototype_out_dim = (num_prototypes, latent_dim[1], latent_dim[2])
    global_max_pooling_out = (num_prototypes,)

    orig_stdout = sys.stdout

    # save architecture
    model_arc_path = os.path.join(model.logging_dir, logging_name, "model_arc")
    create_dir(model_arc_path)
    output_file = os.path.join(model_arc_path, "model_arc.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        # Redirect stdout to the file
        sys.stdout = f

        # FEATURE EXTRACTOR ARCHITECTURE
        print(model.features)
        torchsummary.summary(model.features, model.features_input_dim[1:])

        # ENCODER ARCHITECTURE
        print(model.encoder)
        torchsummary.summary(model.encoder, model.features_out_dim[1:])

        # DECODER ARCHITECTURE
        print(model.decoder)
        torchsummary.summary(model.decoder, latent_dim)

        # CLASSIFIER ARCHITECTURE
        print("CLASSIFIER BLOCK ARC:")
        print("\n")
        print("PROTOTYPE LAYER:")
        print(f"prototypeL_input_dim: {prototype_input_dim}")
        print(f"Prototype Layer Dim: {(model.prototype_vectors.size())}")
        print(f"prototypeL_output_dim: {prototype_out_dim}")
        print("GLOBAL MAX POOLING")
        print(f"maxpool_output_dim: {global_max_pooling_out}")
        print("FULLY CONNECTED LAYER:")
        print(model.last_layer)
        torchsummary.summary(model.last_layer, global_max_pooling_out)
        print("\n")

    # Restore the original stdout
    sys.stdout = orig_stdout
    print(f"Model architecture saved to {output_file}")
