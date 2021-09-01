import tensorflow as tf
from enum import Enum
from typing import Literal


class TfDevice(Enum):
    CPU = 'CPU'
    GPU = 'GPU'
    TPU = 'TPU'


def init_processing_device(mode: Literal[TfDevice.TPU, TfDevice.GPU, TfDevice.TPU], log_on=False):
    if mode == TfDevice.TPU:
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            print("Device:", tpu.master())
        except:
            tf.debugging.set_log_device_placement(log_on)
            dev = tf.config.list_logical_devices(mode.value)
            strategy = tf.distribute.MirroredStrategy(dev)
            print("Devices:", dev)
    else:
        tf.debugging.set_log_device_placement(log_on)
        dev = tf.config.list_logical_devices(mode.value)
        strategy = tf.distribute.MirroredStrategy(dev)
        print("Devices:", dev)

    print("Number of replicas:", strategy.num_replicas_in_sync)
    return strategy
