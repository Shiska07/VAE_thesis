import os
import shutil

import torch
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

early_stopping_callback = EarlyStopping(
                monitor='avg_total_val_loss',
                min_delta=0.1,
                patience=5,
                verbose=True,
                mode='min'
            )

def getmodel_ckpt_callback(best_model_ckpt_path, params):

    checkpoint_callback = ModelCheckpoint(
        monitor='avg_total_val_loss',
        dirpath=best_model_ckpt_path,  # Directory to save checkpoints
        filename=f"best_model{params['logging_name']}", # Prefix for the
        # checkpoint
        # filenames
        save_top_k=1,  # Save the best model only
        mode='min',
        every_n_epochs=1
    )

    return checkpoint_callback

# clearing logs with the same name
def clear_prev_logs(dir_path):
    if os.path.exists(dir_path):
        # If it does, delete its contents
        shutil.rmtree(dir_path)
        print(f"Previous logs in {dir_path} have been cleared.")







