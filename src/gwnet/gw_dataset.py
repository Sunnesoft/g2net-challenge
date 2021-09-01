import tensorflow as tf
import pandas as pd
import os
import random
from functools import partial
import uuid
import shutil

AUTOTUNE = tf.data.AUTOTUNE


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def decode_image(example, size, scale):
    image = tf.image.decode_png(example["image_raw"])
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [example["width"], example["height"], example["depth"]])

    if scale is not None:
        image = image / tf.constant(scale, dtype=tf.float32)

    if size is not None and size[0] != example["width"] \
            and size[1] != example["height"]:
        image = tf.image.resize(image, size)
    return image


def open_image(filename):
    return open(filename, 'rb').read()


def get_image_name(filename):
    bn = os.path.basename(filename)
    name, file_extension = os.path.splitext(bn)
    return name


def read_tfrecord(example, labeled, linked, image_size, image_scale):
    tfrecord_format = (
        {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'target': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'filename': tf.io.FixedLenFeature([], tf.string),
        }
        if labeled
        else {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'filename': tf.io.FixedLenFeature([], tf.string),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example, size=image_size, scale=image_scale)
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        if linked:
            reference = tf.cast(example["filename"], tf.string)
            return image, label, reference
        return image, label

    if linked:
        reference = tf.cast(example["filename"], tf.string)
        return image, reference

    return image


def write_tfrecord(writer, filename, label, image_size):
    image = open_image(filename)
    dimage = tf.image.decode_png(image)

    if image_size is not None and image_size[0:2] != dimage.shape[0:2]:
        dimage = tf.image.resize(dimage, image_size)
        dimage = tf.cast(dimage, tf.uint8)
        image = tf.image.encode_png(dimage)

    image_shape = dimage.shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'image_raw': _bytes_feature(image),
        'filename': _bytes_feature(get_image_name(filename)),
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
        remove_older=False,
        image_size = None):
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
                        write_tfrecord(writer, filename, label, image_size)
                    except:
                        print(f'Skip invalid file {filename}')
                        continue


def load_dataset(filenames, labeled=True, linked=False, image_size=None, image_scale=None):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord,
                labeled=labeled,
                linked=linked,
                image_size=image_size,
                image_scale=image_scale), num_parallel_calls=AUTOTUNE
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
        res.append(fns[prev_ind:prev_ind + split_ind])
        prev_ind = split_ind
    return res


def get_dataset(filenames, labeled=True, linked=False,
                shuffle=2048, batch_size=64,
                image_size=None, image_scale=None):
    dataset = load_dataset(filenames, labeled=labeled, linked=linked,
                           image_size=image_size, image_scale=image_scale)

    if shuffle is not None:
        dataset = dataset.shuffle(shuffle)

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset
