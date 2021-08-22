import gw_dataset as gwds

import tensorflow as tf
from typing import Literal, Tuple, Union
from enum import Enum
import os
import matplotlib.pyplot as plt
import numpy as np
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


class TfDevice(Enum):
    CPU = 'CPU'
    GPU = 'GPU'
    TPU = 'TPU'


def show_batch(image_batch, label_batch):
    batch_size = len(image_batch)
    columns_count = int(math.sqrt(batch_size))
    rows_count = math.ceil(batch_size / columns_count)

    plt.figure(figsize=(10, 10))
    for n in range(batch_size):
        ax = plt.subplot(rows_count, columns_count, n + 1)
        plt.imshow(image_batch[n] / 255.0)
        plt.title(label_batch[n])
        plt.axis("off")
    plt.show()


class GwModelBase:
    def __init__(self,
                 name: str,
                 mode: Literal[TfDevice.TPU, TfDevice.GPU, TfDevice.TPU],
                 image_size: Tuple[int, int, int],
                 image_scale_factor: Union[None, float] = None):
        self._mode = mode
        self._name = name
        self._model_fn = f'{self._name}.h5'

        if self._mode == TfDevice.TPU:
            try:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                self._strategy = tf.distribute.TPUStrategy(tpu)
                print("Device:", tpu.master())
            except:
                tf.debugging.set_log_device_placement(True)
                dev = tf.config.list_logical_devices(self._mode.value)
                self._strategy = tf.distribute.MirroredStrategy(dev)
                print("Devices:", dev)
        else:
            tf.debugging.set_log_device_placement(True)
            dev = tf.config.list_logical_devices(self._mode.value)
            self._strategy = tf.distribute.MirroredStrategy(dev)
            print("Devices:", dev)

        print("Number of replicas:", self._strategy.num_replicas_in_sync)

        self._dataset_train = None
        self._dataset_valid = None
        self._dataset_test = None
        self._image_size = image_size
        self._image_scale_factor = image_scale_factor
        self._model = None
        self._history = None

    def load_train_dataset(
            self,
            data_path: str,
            train_dataset_volume: float = 0.9,
            valid_dataset_volume: float = 0.1,
            shuffle=2048,
            batch_size=64):
        train_fn, valid_fn = gwds.load_filenames(
            data_path, split=(train_dataset_volume, valid_dataset_volume))

        if len(train_fn) > 0.0:
            self._dataset_train = gwds.get_dataset(
                train_fn, batch_size=batch_size, shuffle=shuffle,
                image_size=self._image_size[0:2], image_scale=self._image_scale_factor)

        if len(valid_fn) > 0.0:
            self._dataset_valid = gwds.get_dataset(
                valid_fn, batch_size=batch_size, shuffle=shuffle,
                image_size=self._image_size[0:2], image_scale=self._image_scale_factor)

    def load_test_dataset(
            self,
            data_path,
            shuffle=2048,
            batch_size=64):
        test_fn = gwds.load_filenames(data_path)

        if len(test_fn) > 0.0:
            self._dataset_test = gwds.get_dataset(
                test_fn, labeled=False, batch_size=batch_size,
                shuffle=shuffle, image_size=self._image_size[0:2], image_scale=self._image_scale_factor)

    def compile(self, **kwargs):
        with self._strategy.scope():
            self._model = self.make_model(**kwargs)

    def make_model(self, **kwargs):
        raise NotImplementedError('make_model() procedure must be implemented in GwModelBase class child')

    def load_model(self, path):
        self._model.load_weights(os.path.join(path, self._model_fn))

    def print_model(self):
        print(self._model.summary())

    def fit(self, **kwargs):
        if 'callbacks' not in kwargs:
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                self._model_fn, save_best_only=True
            )
            early_stopping_cb = tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True
            )
            kwargs.setdefault('callbacks', [checkpoint_cb, early_stopping_cb])

        self._history = self._model.fit(
            self._dataset_train,
            validation_data=self._dataset_valid,
            **kwargs
        )

        return self._history

    def predict(self, **kwargs) -> np.ndarray:
        kwargs.setdefault('x', self._dataset_test)
        return self._model.predict(**kwargs)

    def show_random_train_batch(self, subs=None):
        image_batch, label_batch = next(iter(self._dataset_train))
        if subs is not None:
            label_batch = list(map(lambda x: subs[str(x.numpy())], label_batch))
        show_batch(image_batch, label_batch)

    def show_random_test_batch(self):
        image_batch = next(iter(self._dataset_test))
        label_batch = self.predict(x=image_batch)
        show_batch(image_batch, label_batch)


class GwXception(GwModelBase):
    def make_model(self, learn_rate_schedule=None, **kwargs):
        base_model = tf.keras.applications.Xception(
            input_shape=self._image_size, include_top=False, weights=None
        )

        base_model.trainable = True

        inputs = tf.keras.layers.Input(self._image_size)
        x = tf.keras.applications.xception.preprocess_input(inputs)
        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(8, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.7)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        if learn_rate_schedule is None:
            initial_learning_rate = 0.01
            learn_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
            )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=tf.keras.metrics.AUC(name="auc"),
        )

        return model


class GwEfficientNetB0(GwModelBase):
    def make_model(self, learn_rate_schedule=None, **kwargs):
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=self._image_size, include_top=False, weights=None
        )

        base_model.trainable = True

        inputs = tf.keras.layers.Input(self._image_size)
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(8, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.7)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        if learn_rate_schedule is None:
            initial_learning_rate = 0.01
            learn_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
            )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate_schedule),
            loss="binary_crossentropy",
            metrics=tf.keras.metrics.AUC(name="auc"),
        )

        return model


class GwLeNet(GwModelBase):
    def make_model(self, optimizer, loss, metrics, **kwargs):
        model = Sequential()
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same",
                         input_shape=self._image_size))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        # sigmoid classifier
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

        return model


if __name__ == '__main__':
    data_path = './data/tfrecords/'

    optimizer = 'adam'
    loss = 'binary_crossentropy'
    metrics = ['AUC']

    solver = GwEfficientNetB0('eff_net_b0', TfDevice.CPU, (71, 71, 1), 255.0)
    solver.load_train_dataset(data_path, batch_size=64)
    solver.show_random_train_batch(subs={'1': 'GW_TRUE', '0': 'GW_FALSE'})

    solver.compile()
    solver.print_model()
    history = solver.fit(epochs=20)