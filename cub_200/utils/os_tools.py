from os import makedirs
from os.path import exists
import pickle
import json

import numpy as np

def create_dir(directory, verbose=False):
    if not exists(directory):
        makedirs(directory)
    if verbose:
        print(f"directory created: {directory}")


def save_np_array(np_array, file_path, verbose=False):
    np.savetxt(file_path, np_array)
    if verbose:
        print(f"numpy array successfully saved: {file_path}")


def load_np_array(file_path, verbose=False):
    array = np.loadtxt(file_path)
    if verbose:
        print(f"numpy array successfully loaded from: {file_path}")
    return array


def save_transformation(transformation, file_path, verbose=False):
    with open(file_path, 'wb') as file_handle:
        pickle.dump(transformation, file_handle)
    if verbose:
        print(f"Transformations successfully saved as: {file_path}")


def load_transformation(file_path, verbose=False):
    with open(file_path, 'rb') as file_handle:
        transformation = pickle.load(file_handle)
    if verbose:
        print(f"Transformations successfully loaded from: {file_path}")
    return transformation


def save_latent_space(latent_space, file_path, verbose=False):
    with open(file_path, 'wb') as file_handle:
        pickle.dump(latent_space, file_handle)
    if verbose:
        print(f"Latent space successfully saved as: {file_path}")


def load_latent_space(file_path, verbose=False):
    with open(file_path, 'rb') as file_handle:
        latent_space = pickle.load(file_handle)
    if verbose:
        print(f"Latent space successfully loaded from: {file_path}")
    return latent_space


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