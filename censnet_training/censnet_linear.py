import logging
import numpy as np
import tensorflow as tf
from keras import initializers
import tensorflow
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Layer,
    Input,
    Normalization,
    Dense,
    LeakyReLU,
    BatchNormalization,
    ReLU,
)

from spektral.layers import CensNetConv

# import spektral
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
import pickle as pkl
import pandas as pd

# from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os
import optuna
import pandas as pd
import gc


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_loader import data_loader
from utils.censnet_model import CensNetModel

# from ...gas_gnn_censnet.utils.data_loader import data_loader
# from ..utils.data_loader import data_loader

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

savepath = "./results/"
folder_path = "./db/"
method_folder = ["/Dummy/npy/", "/Col/npy/"]

method = 0
ampa = data_loader(folder_path, method_folder, [0])


X_train = ampa.X_train_padding
An_train = ampa.An_train_padding
Ae_train = ampa.Ae_train_padding
T_train = ampa.T_train_padding
Z_train = ampa.Z_train_padding
fw_train = ampa.fw_train_padding
p_train = ampa.p_train_padding
ye_train = ampa.ye_train_padding
IR_train = ampa.IR_train_padding

X_test = ampa.X_test_padding
An_test = ampa.An_test_padding
Ae_test = ampa.Ae_test_padding
T_test = ampa.T_test_padding
Z_test = ampa.Z_test_padding
fw_test = ampa.fw_test_padding
p_test = ampa.p_test_padding
ye_test = ampa.ye_test_padding
IR_test = ampa.IR_test_padding

X_val = ampa.X_val_padding
An_val = ampa.An_val_padding
Ae_val = ampa.Ae_val_padding
T_val = ampa.T_val_padding
Z_val = ampa.Z_val_padding
fw_val = ampa.fw_val_padding
p_val = ampa.p_val_padding
ye_val = ampa.ye_val_padding
IR_val = ampa.IR_val_padding

training_outputs = [
    fw_train,
    p_train,
    ye_train,
    np.zeros([len(fw_train), fw_train[0].shape[0]]),
    np.zeros([len(ye_train), ye_train[0].shape[0]]),
]
test_outputs = [
    fw_test,
    p_test,
    ye_test,
    np.zeros([len(fw_test), fw_test[0].shape[0]]),
    np.zeros([len(ye_test), ye_test[0].shape[0]]),
]
val_outputs = [
    fw_val,
    p_val,
    ye_val,
    np.zeros([len(fw_val), fw_val[0].shape[0]]),
    np.zeros([len(ye_val), ye_val[0].shape[0]]),
]


training_inputs = [X_train, [An_train, Ae_train, IR_train], Z_train]
val_inputs = [X_val, [An_val, Ae_val, IR_val], Z_val]
test_inputs = [X_test, [An_test, Ae_test, IR_test], Z_test]


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2, decay_steps=1000, decay_rate=0.9
)


def objective(trial):
    tf.keras.backend.clear_session()
    callback_val = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5000, restore_best_weights=False, mode="min"
    )

    callback_model = tf.keras.callbacks.ModelCheckpoint(
        savepath + "best_model_lineal.weights.h5",
        mode="min",
        save_best_only=True,
        verbose=1,
        save_weights_only=True,
    )

    callback_lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)

    callback_csv = tf.keras.callbacks.CSVLogger(savepath + "csv_edge.csv")

    callback_ron = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5000,
        min_lr=0.000001,
        mode="min",
        verbose=1,
    )

    log_dir = "./log"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  # Enable histogram computation with each epoch.
        embeddings_freq=1,
        write_graph=True,
    )

    N_channels = trial.suggest_int("N_channels", 16, 64)
    N_layers = trial.suggest_int("N_layers", 1, 5)
    N_dense = trial.suggest_int("N_dense", 2, 32)
    fe_loss = trial.suggest_int("fe_loss", 0, 1)

    mdl = CensNetModel(
        N_channels=N_channels,
        N_layers=N_layers,
        N_dense=N_dense,
        Node_size=63,
        Edge_size=62,
    ).build_model()

    mdl.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=["mse", "mse", "mse", "mse", "mse"],  # fw, p, fe, bal, wey
        loss_weights=[1, 0, fe_loss, 0, 0],
    )

    full_history = mdl.fit(
        training_inputs,
        training_outputs,
        # sample_weight=w_train,
        verbose=1,
        epochs=1500,
        batch_size=128,
        # validation_data=(val_inputs,val_outputs,w_val),
        validation_data=(val_inputs, val_outputs),
        callbacks=[
            callback_model,
            callback_lr,
            #  callback_csv_drive,
            callback_csv,
            #  callback_ron,
            tensorboard_callback,
        ],
    )
    df = pd.read_csv(savepath + "csv_edge.csv")
    best_model = df[df["val_fw_loss"] == df["val_fw_loss"].min()]
    # df[df[df['val_fw_loss'] ==  df['val_fw_loss'].min()]]

    best_results = best_model.loc[
        :,
        [
            "val_loss",
            "val_fw_loss",
            "val_p_loss",
            "val_balance_loss",
            "val_weymouth_loss",
        ],
    ]
    best_model = df[df["val_fw_loss"] == df["val_fw_loss"].min()]
    print(best_model["val_loss"])

    # Clear the TensorFlow session
    tf.keras.backend.clear_session()

    # Manually trigger garbage collection
    gc.collect()

    return float(best_model["val_loss"])


# Configure TensorFlow to grow GPU memory usage as needed
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Add stream handler of stdout to show the messages
# optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "linear"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(
    direction="minimize",
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
)

# study = optuna.create_study(direction="maximize")
study.optimize(
    objective,
    n_trials=100,
)


print("eso")
