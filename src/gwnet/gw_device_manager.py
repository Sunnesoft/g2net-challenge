import tensorflow as tf
from enum import Enum

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class TfDevice(Enum):
    CPU = 'CPU'
    GPU = 'GPU'
    TPU = 'TPU'


def init_strategy(mode: Literal[TfDevice.TPU, TfDevice.GPU, TfDevice.CPU], log_on=False):
    strategy = None
    if mode == TfDevice.TPU:
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            print("Device:", tpu.master())
            print("Number of replicas:", strategy.num_replicas_in_sync)
            return strategy
        except:
            mode = TfDevice.GPU

    if mode in [TfDevice.GPU, TfDevice.CPU]:
        tf.debugging.set_log_device_placement(log_on)
        dev = get_logical_devices(mode)
        dev_count = len(dev)

        if dev_count < 1:
            raise RuntimeError('list of logical devices is empty')
        elif dev_count == 1:
            strategy = tf.distribute.OneDeviceStrategy(dev[0])
        else:
            strategy = tf.distribute.MirroredStrategy(dev)
        print("Device:", dev)
        print("Number of replicas:", strategy.num_replicas_in_sync)

    return strategy


def get_logical_devices(mode: Literal[TfDevice.TPU, TfDevice.GPU, TfDevice.CPU]):
    return tf.config.list_logical_devices(mode.value)


def get_first_logical_device(mode: Literal[TfDevice.TPU, TfDevice.GPU, TfDevice.CPU]):
    dev = get_logical_devices(mode)
    dev_count = len(dev)

    if dev_count < 1:
        raise RuntimeError('list of logical devices is empty')
    return dev[0]
