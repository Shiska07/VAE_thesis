from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

'''
Some components of the following implementation were obtained from: https://github.com/cfchen-duke/ProtoPNet
 '''

def get_earlystopping_callback(min_delta, patience):
    early_stopping_callback = EarlyStopping(
                    monitor="val_total_loss",
                    min_delta=min_delta,
                    patience=patience,
                    verbose=True,
                    mode="min"
    )
    return early_stopping_callback


def getmodel_ckpt_callback(model_ckpt_path):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_loss",
        dirpath=model_ckpt_path,
        filename="{epoch}-{val_total_loss:.2f}",
        save_top_k=5,
        mode="min",
        every_n_epochs=1
    )
    return checkpoint_callback



