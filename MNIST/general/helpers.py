import os
import json

import numpy as np
import torch
import pandas as pd
from os import makedirs
from os.path import exists
import matplotlib.pyplot as plt


# calulates the average losses for each epoch
def get_average_losses(losses_list, stage):
    num_items = len(losses_list)

    cum_rec_loss = 0
    cum_kl_loss = 0
    cum_ce_loss = 0
    cum_clst_loss = 0
    cum_sep_loss = 0
    cum_l1_loss = 0
    cum_total_loss = 0
    cum_total_acc = 0

    for loss_dict in losses_list:
        cum_rec_loss += loss_dict[f"recon_loss{stage}"]
        cum_kl_loss += loss_dict[f"kl_loss{stage}"]
        cum_ce_loss += loss_dict[f"ce_loss{stage}"]
        cum_clst_loss += loss_dict[f"clst_loss{stage}"]
        cum_sep_loss += loss_dict[f"sep_loss{stage}"]
        cum_l1_loss += loss_dict[f"sep_loss{stage}"]
        cum_total_loss += loss_dict[f"total_loss{stage}"]
        cum_total_acc += loss_dict[f"acc{stage}"]

    # get average loss
    avg_rec_loss = cum_rec_loss / num_items
    avg_kl_loss = cum_kl_loss / num_items
    avg_ce_loss = cum_ce_loss / num_items
    avg_clst_loss = cum_clst_loss / num_items
    avg_sep_loss = cum_sep_loss / num_items
    avg_l1_loss = cum_l1_loss / num_items
    avg_total_loss = cum_total_loss / num_items
    avg_acc = cum_total_acc / num_items

    return avg_rec_loss, avg_kl_loss, avg_ce_loss, \
        avg_clst_loss, avg_sep_loss, avg_l1_loss, avg_total_loss, avg_acc


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


def save_history(history, history_dir, model_name, params_fname, batch_size, training_type, h_params):

    history_file_path = os.path.join(history_dir,
        str(model_name), str(params_fname), str(batch_size), str(training_type))

    # create directory if non-existent
    try:
        os.makedirs(history_file_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {history_file_path}: {e}")


    # create and save df
    csv_file_path = os.path.join(history_file_path, f'history{training_type}.csv')
    if os.path.isfile(csv_file_path):
        os.remove(csv_file_path)
    df = pd.DataFrame(history)
    df.to_csv(csv_file_path, index = False)

    hp_file_path = os.path.join(history_file_path, f'hyperparameters{training_type}.json')
    with open(hp_file_path, 'w') as json_file:
        json.dump(h_params, json_file)

    return history_file_path


def save_plots(train_epoch_hist, val_epoch_hist, plots_file_path):

    train_rec_loss = train_epoch_hist['train_rec_loss']
    train_kl_loss = train_epoch_hist['train_kl_loss']
    train_total_loss = train_epoch_hist['train_total_loss']


    # pop(0) as the first value is from sanity check
    val_epoch_hist['val_rec_loss'].pop(0)
    val_epoch_hist['val_kl_loss'].pop(0)
    val_epoch_hist['val_total_loss'].pop(0)
    val_rec_loss = val_epoch_hist['val_rec_loss']
    val_kl_loss = val_epoch_hist['val_kl_loss']
    val_total_loss = val_epoch_hist['val_total_loss']


    plt.figure(figsize=(24, 6))
    plt.subplot(1, 3, 1)
    # create train_loss vs. val_loss
    plt.plot(train_rec_loss, label='Train Loss', color='blue')
    plt.plot(val_rec_loss, label='Validation Loss', color='red')
    plt.title(f'Training Vs Validation Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(train_kl_loss, label='Train Loss', color='blue')
    plt.plot(val_kl_loss, label='Validation Loss', color='red')
    plt.title(f'Training Vs Validation KL Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(train_total_loss, label='Train Loss', color='blue')
    plt.plot(val_total_loss, label='Validation Loss', color='red')
    plt.title(f'Training Vs Validation Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    name = os.path.join(plots_file_path, 'loss_plots.jpeg')
    if os.path.isfile(name):
        os.remove(name)
    plt.savefig(name)
    plt.close('all')
