import os
import json
import pandas as pd
from os import makedirs
from os.path import exists
import matplotlib.pyplot as plt


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
