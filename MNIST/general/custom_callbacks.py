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

class ClearPreviousLogsCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        # Check if the directory exists
        if os.path.exists(trainer.log_dir):
            # If it does, delete its contents
            shutil.rmtree(trainer.log_dir)
            print(f"Previous logs in {trainer.log_dir} have been cleared.")

# this function logs train and validation losses in the same chart
class LogCustomScalarsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.writer = None

    def on_train_start(self, trainer, pl_module):
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=trainer.logger.log_dir)

    def on_train_end(self, trainer, pl_module):
        train_metric = trainer.logged_metrics

        print(type(train_metric))
        print(train_metric)
        # Close TensorBoard writer
        self.writer.close()







