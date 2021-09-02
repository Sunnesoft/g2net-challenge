import tensorflow as tf

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Tuple, Union
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from .gw_dataset import load_filenames, get_dataset
from .gw_device_manager import TfDevice, init_strategy


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
                 mode: Literal[TfDevice.TPU, TfDevice.GPU, TfDevice.CPU],
                 image_size: Tuple[int, int, int],
                 image_scale_factor: Union[None, float] = None,
                 model_path: str= ''):
        self._mode = mode
        self._name = name
        self._model_path = model_path

        self._strategy = init_strategy(self._mode)

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
        train_fn, valid_fn = load_filenames(
            data_path, split=(train_dataset_volume, valid_dataset_volume))

        if len(train_fn) > 0.0:
            self._dataset_train = get_dataset(
                train_fn, batch_size=batch_size, shuffle=shuffle,
                image_size=self._image_size[0:2], image_scale=self._image_scale_factor)

        if len(valid_fn) > 0.0:
            self._dataset_valid = get_dataset(
                valid_fn, batch_size=batch_size, shuffle=shuffle,
                image_size=self._image_size[0:2], image_scale=self._image_scale_factor)

    def load_test_dataset(self, data_path, batch_size=64):
        test_fn = load_filenames(data_path)

        if len(test_fn) > 0.0:
            self._dataset_test = get_dataset(
                test_fn, labeled=False, linked=True, batch_size=batch_size,
                shuffle=None, image_size=self._image_size[0:2], image_scale=self._image_scale_factor)

    def compile(self, **kwargs):
        with self._strategy.scope():
            self._model = self.make_model(**kwargs)

    def make_model(self, **kwargs):
        raise NotImplementedError('make_model() procedure must be implemented in GwModelBase class child')

    def load_model(self):
        self._model.load_weights(self.model_fullpath)

    def print_model(self):
        print(self._model.summary())

    def fit(self, **kwargs):
        if 'callbacks' not in kwargs:
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                self.model_fullpath, save_best_only=True
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
        kwargs.setdefault('x', self._dataset_test.map(lambda x, r: x))
        return self._model.predict(**kwargs)

    def _test_dataset_refs(self):
        return list(self._dataset_test.map(lambda x, r: r).as_numpy_iterator())

    def infer_test_dataset(self):
        result = self.predict()
        links = self._test_dataset_refs()

        df = pd.DataFrame(columns=['id', 'target'])
        df['id'] = links
        df['target'] = result
        os.makedirs(os.path.dirname(self.infer_filename), exist_ok=True)
        df.to_csv(path_or_buf=self.infer_filename, index=False)

    def show_random_train_batch(self, subs=None):
        image_batch, label_batch = next(iter(self._dataset_train))
        if subs is not None:
            label_batch = list(map(lambda x: subs[str(x.numpy())], label_batch))
        show_batch(image_batch, label_batch)

    def show_random_test_batch(self):
        image_batch = next(iter(self._dataset_test))
        label_batch = self.predict(x=image_batch)
        show_batch(image_batch, label_batch)

    @property
    def model_filename(self):
        return f'{self._name}.h5'

    @property
    def infer_filename(self):
        return f'{self._name}.csv'

    @property
    def model_fullpath(self):
        return os.path.join(self._model_path, self.model_filename)

    @property
    def infer_fullpath(self):
        return os.path.join(self._model_path, self.infer_filename)


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

    def show_hist(self):
        plt.clf()
        plt.plot(self._history["auc"])
        plt.plot(self._history["val_auc"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()


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