import tensorflow as tf
import pandas as pd
import os


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


image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
  return tf.io.parse_single_example(example_proto, image_feature_description)


def open_image(filename):
    return open(filename, 'rb').read()


def load_labels(filename):
    df = pd.read_csv(filename, header=0)
    return df


def id_from_image_name(image_name):
    filename = os.path.basename(image_name)
    return filename.split('.')[0]


def get_labels(df, targ_id):
    return df.loc[df['id'].isin(targ_id)]


def get_task(df, files, path):
    result = []
    for fname in files:
        id = id_from_image_name(fname)
        row = df.loc[df['id'] == id]
        result.append((os.path.join(path, fname), row['target']))
    return result


def image_example(image_string, label):
    image_shape = tf.io.decode_png(image_string).shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def images_from_train_tfrecord(fn):
    raw_image_dataset = tf.data.TFRecordDataset(fn)
    return raw_image_dataset.map(_parse_image_function)


def train_dir_to_task(path, labels_fn):
    labels = load_labels(labels_fn)
    task = []
    for root, dirs, files in os.walk(path):
        task.extend(get_task(labels, files, root))
    return task


def images_to_train_tfrecord(tasks, out_fn):
    with tf.io.TFRecordWriter(out_fn) as writer:
        for task in tasks:
            filename, label = task
            image_string = open_image(filename)
            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    fpath = './data/cqt/train/'
    labels_fn = './training_labels.csv'
    out_fn = './data/train.tfrecord'
    task = train_dir_to_task(fpath, labels_fn)
    images_to_train_tfrecord(task, out_fn)