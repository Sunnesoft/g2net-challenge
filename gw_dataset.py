import tensorflow as tf
import pandas as pd
import os
import random
from functools import partial
import matplotlib.pyplot as plt
import uuid
import shutil

AUTOTUNE = tf.data.AUTOTUNE

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def decode_image(example):
    image = tf.image.decode_png(example["image_raw"], channels=0)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [example["width"], example["height"], example["depth"]])
    return image


def open_image(filename):
    return open(filename, 'rb').read()


def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'target': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }
        if labeled
        else {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example)
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    return image


def write_tfrecord(writer, image, label):
    image_shape = tf.io.decode_png(image).shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'image_raw': _bytes_feature(image)
    }

    if label is not None:
        feature['target'] = _int64_feature(label)

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def load_labels(filename):
    df = pd.read_csv(filename, header=0)
    return df


def get_id_from_image_name(image_name):
    filename = os.path.basename(image_name)
    return filename.split('.')[0]


def get_labels(df, targ_id):
    return df.loc[df['id'].isin(targ_id)]


def get_task(files, path, labels):
    if isinstance(labels, pd.DataFrame):
        ids = [get_id_from_image_name(fname) for fname in files]
        rows = labels.loc[labels['id'].isin(ids)]
        return [(os.path.join(path, row['id'] + '.png'), row['target']) for i, row in rows.iterrows()]
    return [(os.path.join(path, fname), None) for fname in files]


def create_task(path, labels, shuffle, split, batch_size):
    task = []
    for root, dirs, files in os.walk(path):
        task.extend(get_task(files, root, labels))

    if shuffle:
        random.shuffle(task)

    if batch_size is not None:
        task = [task[i:i + batch_size] for i in range(0, len(task), batch_size)]

    if isinstance(split, dict):
        n = len(task)
        m = 0
        m_prev = 0
        splitted_task = {}
        for key, split_val in split.items():
            m += int(split_val * n)
            splitted_task[key] = task[m_prev:m]
            m_prev = m
        task = splitted_task

    if not isinstance(task, dict):
        task = dict(default=task)

    return task


def create_tfrecords(
        path,
        out_task,
        labels=None,
        shuffle=True,
        batch_size=64,
        remove_older=False):
    if isinstance(labels, str):
        labels = load_labels(labels)

    tasks = create_task(path, labels, shuffle, out_task, batch_size)

    for out_path, splitted_tasks in tasks.items():
        if batch_size is None:
            splitted_tasks = [splitted_tasks]

        if remove_older and os.path.isdir(out_path):
            shutil.rmtree(out_path)

        os.makedirs(out_path, exist_ok=True)

        for batched_tasks in splitted_tasks:
            with tf.io.TFRecordWriter(os.path.join(out_path, f'{uuid.uuid4()}.tfrecord')) as writer:
                for task in batched_tasks:
                    filename, label = task
                    try:
                        write_tfrecord(writer, open_image(filename), label)
                    except:
                        print(f'Skip invalid file {filename}')
                        continue



def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


def load_filenames(path, split=None):
    fns = tf.io.gfile.glob(f'{path}*.tfrecord')
    if split is None:
        return fns

    res = []
    prev_ind = 0
    for split_val in split:
        split_ind = int(split_val * len(fns))
        res.append(fns[prev_ind:prev_ind+split_ind])
        prev_ind = split_ind
    return res


def get_dataset(filenames, labeled=True, shuffle=2048, batch_size=64):
    dataset = load_dataset(filenames, labeled=labeled)
    dataset = dataset.shuffle(shuffle)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255.0)
        if label_batch[n]:
            plt.title("GW_TRUE")
        else:
            plt.title("GW_FALSE")
        plt.axis("off")


def make_model(image_size):
    base_model = tf.keras.applications.Xception(
        input_shape=image_size, include_top=False, weights=None
    )

    base_model.trainable = False

    inputs = tf.keras.layers.Input(image_size)
    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=tf.keras.metrics.AUC(name="auc"),
    )

    return model


if __name__ == '__main__':
    mode = 'CPU' #'GPU'
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Device:", tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    except:
        tf.debugging.set_log_device_placement(True)
        gpus = tf.config.list_logical_devices(mode)
        strategy = tf.distribute.MirroredStrategy(gpus)

    print("Number of replicas:", strategy.num_replicas_in_sync)


    fpath = './data/cqt/train/'
    labels_fn = './training_labels.csv'
    out_path = './data/tfrecords/'
    out_task = {}
    out_task[out_path] = 100

    # create_tfrecords(fpath, out_task, labels_fn, shuffle=True, batch_size=64, remove_older=True)

    batch_size = 32
    train_fn, valid_fn = load_filenames(out_path, split=(0.9, 0.1))
    train_dataset = get_dataset(train_fn, batch_size=batch_size)
    valid_dataset = get_dataset(valid_fn, batch_size=batch_size)

    image_batch, label_batch = next(iter(train_dataset))
    show_batch(image_batch.numpy(), label_batch.numpy())
    plt.show()

    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "gw_model.h5", save_best_only=True
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )

    with strategy.scope():
        model = make_model(image_size=(760, 760, 1))

    history = model.fit(
        train_dataset,
        epochs=2,
        validation_data=valid_dataset,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )