
import os
import sys
from .base import Base_Model
from src.utils import Pkl
import tensorflow as tf
from sklearn.metrics import average_precision_score
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from .keras_1d_cnn import build_model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import Sequence
import numpy as np
from tensorflow.keras.models import load_model


def getNewestModel(model, dirname):
    """get the newest model file within a directory"""
    from glob import glob
    target = os.path.join(dirname, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    if len(files) == 0:
        return model
    else:
        newestModel = sorted(files, key=lambda files: files[1])[-1]
        print(f"load model: {newestModel[0]}")
        model.load_weights(newestModel[0])
        return model


class DataGenerator(Sequence):
    def __init__(self, x, y, batch_size=64, cyclic_shift__alpha=0.0, skew__skew=0.0):
        self.input = x
        self._x = x["input_signal"]
        self._y = y
        self.indices = np.arange(y.shape[0])
        #self.indices_mixup = np.arange(y.shape[0])
        self._batch_size = batch_size
        self._cyclic_shift__alpha = cyclic_shift__alpha
        self.skew__skew = skew__skew

    def __getitem__(self, index):
        indexes = self.indices[index * self._batch_size:(index + 1) * self._batch_size]
        #indexes_mixup = self.indices_mixup[index * self._batch_size:(index + 1) * self._batch_size]
        batch_x, batch_y = [], []
        batch_meta_x = []

        #i_mixup = 0
        for index in indexes:
            _x = self._x[index]
            _y = self._y[index]
            _x = self.cyclic_shift(_x, self._cyclic_shift__alpha)
            #_x, _y = self.mixup(_x, _y, self._x[indexes_mixup[i_mixup]], self._y[indexes_mixup[i_mixup]])

            batch_x.append(_x)
            batch_y.append(_y)
            #i_mixup += 1

            batch_meta_x.append(self.input["input_meta"][index])

        return (
            {
                "input_signal": np.array(batch_x),
                "input_meta": np.array(batch_meta_x)
            },
            np.array(batch_y)
        )

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        #np.random.shuffle(self.indices_mixup)

    def __len__(self):
        return len(self._y) // self._batch_size

    @staticmethod
    def cyclic_shift(x, alpha=0.5):
        s = np.random.uniform(0, alpha)
        part = int(len(x) * s)
        x_ = x[:part, :]
        _x = x[-len(x) + part:, :]
        return np.concatenate([_x, x_], axis=0)

    @staticmethod
    def skew(x, skew=0.05):
        s = 1 + np.random.normal(loc=0, scale=skew)
        return np.clip(x * s, -1, 1)

    @staticmethod
    def mixup(x, y, x_mix, y_mix, alpha=0.2):
        l = np.random.beta(alpha, alpha)
        x = x * l + x_mix * (1 - l)
        y = y * l + y_mix * (1 - l)
        return x, y


class Model_1DCNN(Base_Model):
    def train(self, x_trn, y_trn, x_val, y_val):
        self.num_fe = len(x_trn.columns)

        wv_cols = [f"wavelength_{i}" for i in range(512)]
        meta_cols = [c for c in x_trn.columns if c not in wv_cols]

        signal_trn = x_trn[wv_cols].values.reshape(-1, 512, 1)
        meta_trn = x_trn[meta_cols].values
        signal_val = x_val[wv_cols].values.reshape(-1, 512, 1)
        meta_val = x_val[meta_cols].values

        pr_auc = AUC(curve='PR', num_thresholds=10000, name="pr_auc")
        cnn_model_params = self.params["model_params"]
        optimizer = optimizers.Adam(lr=cnn_model_params["lr"])
        model = build_model(
            signal_size=signal_trn.shape[1:],
            meta_size=meta_trn.shape[1],
            output_size=1
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=[pr_auc],
        )

        save_path = "./data/output/.tensorlog/"

        trn_gen = DataGenerator(
            {
                "input_signal": signal_trn,
                "input_meta": meta_trn,
            },
            y_trn,
            batch_size=cnn_model_params["batch_size"],
            cyclic_shift__alpha=cnn_model_params["cyclic_shift__alpha"]
        )

        history = model.fit(
            trn_gen,
            steps_per_epoch=len(y_trn) // cnn_model_params["batch_size"],
            epochs=cnn_model_params["epochs"],
            batch_size=cnn_model_params["batch_size"],
            callbacks=[
                ModelCheckpoint(
                    filepath=save_path + self.run_fold_name + '_weights-{epoch:02d}-{val_pr_auc:.2f}.hdf5',
                    monitor='val_pr_auc', verbose=1, save_best_only=True, mode='max'
                ),
                EarlyStopping(
                    monitor='val_pr_auc',
                    patience=cnn_model_params["patience"],
                    mode="max"
                )
            ],
            validation_data=({"input_signal": signal_val, "input_meta": meta_val}, y_val),
            verbose=1,
            shuffle=True,
        )

        self.model = getNewestModel(model, save_path)

    def predict(self, x):
        wv_cols = [f"wavelength_{i}" for i in range(512)]
        meta_cols = [c for c in x.columns if c not in wv_cols]

        signal = x[wv_cols].values.reshape(-1, 512, 1)
        meta = x[meta_cols].values

        return self.model.predict({
                "input_signal": signal,
                "input_meta": meta,
            }).reshape(-1)

    def get_feature_importance(self):
        return np.zeros(self.num_fe)

    def get_best_iteration(self):
        return 0

    def save_model(self):
        model_path = self.model_output_dir / f'{self.run_fold_name}.h5'
        self.model.save(model_path)

    def load_model(self):
        model_path = self.model_output_dir / f'{self.run_fold_name}.h5'
        self.model = load_model(model_path)
