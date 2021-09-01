import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import irfft, rfft, rfftfreq
from scipy.interpolate import interp1d

import tensorflow as tf
import gw_utils as gwu


class GwTimeseries:
    def __init__(self, ts, freq, epoch=0, name=''):
        self._ts = ts
        self._freq = freq
        self._epoch = epoch
        self._duration = len(self._ts) / self._freq
        self._time = np.linspace(self._epoch, self._epoch + self._duration, self._ts.shape[0])
        self._name = name

    @staticmethod
    def load(fp, freq, epoch=0):
        filename, file_extension = os.path.splitext(fp)
        if file_extension in ['.npy']:
            d = np.load(fp)

            if d.shape[0] != d.size:
                return [GwTimeseries(array, freq, epoch, f'{filename}_{i}') for i, array in enumerate(d)]

            return GwTimeseries(d, freq, epoch, filename)
        elif file_extension in ['.hdf5']:
            with h5py.File(fp, 'r') as f:
                d = f['strain']['Strain']
                return GwTimeseries(d, freq, epoch, filename)
        raise ValueError('unexpected timeseries file extension, use .npy or .hdf5 files')

    @staticmethod
    def save(fp, ts):
        filename, file_extension = os.path.splitext(fp)
        if file_extension in ['.npy']:
            result = None
            if isinstance(ts, GwTimeseries):
                result = ts.value
            elif isinstance(ts, list):
                result = []
                for d in ts:
                    result.append(d.value)
                result = np.array(result)
            else:
                raise ValueError('unexpected type of timeseries data')
            np.save(fp, result)
            return True
        raise ValueError('unexpected timeseries file extension, use .npy')

    def plot(self, gtype='scatter'):
        if gtype == 'scatter':
            plt.scatter(self._time, self._ts, label=self._name)
        elif gtype == 'line':
            plt.plot(self._time, self._ts, label=self._name)
        raise ValueError('unexpected plot mode, use `scatter` or `line`')

    def crop(self, range):
        if range is not None:
            self._ts = self._ts[range]
            self._duration = len(self._ts) / self._freq

    def copy(self):
        return GwTimeseries(np.copy(self._ts), self._freq, self._epoch, self._name)

    def apply_window(self, window):
        self._ts = gwu.apply_window(self._ts, window)

    def whiten(self, psd_val):
        f, Pxx = psd_val
        psd_interp = interp1d(f, Pxx)

        Nt = len(self._ts)
        dt = 1.0 / self.sample_rate
        freqs = rfftfreq(Nt, dt)
        hf = rfft(self._ts)
        norm = 1. / np.sqrt(1. / (dt * 2))
        white_hf = hf / np.sqrt(psd_interp(freqs)) * norm
        self._ts = irfft(white_hf, n=Nt)

    def psd(self, fftlength, nperseg=256, overlap=0.5, window=('tukey', 0.25)):
        nwindow = signal.get_window(window, nperseg)
        f, Pxx = gwu.psd(data=self._ts, sample_rate=self._freq,
                         fftlength=fftlength, overlap=overlap, window=nwindow)
        return f, Pxx

    def filter(self, frange, psd_val=None, outlier_threshold=None):
        if outlier_threshold is None:
            outliers = (60, 120, 180)
        else:
            f, Pxx = psd_val
            outliers = f[np.abs(Pxx - Pxx.mean()) > outlier_threshold * Pxx.std()]

        bp = gwu.bandpass(frange[0], frange[1], self._freq)
        notches = [gwu.notch(f, self._freq) for f in outliers if f > 1.0 and f + 1 < self._freq / 2.0]
        zpk = gwu.concatenate_zpks(bp, *notches)
        self._ts = gwu.apply_filter(self._ts, *zpk)

    @property
    def value(self):
        return self._ts

    @property
    def time(self):
        return self._time

    @property
    def sample_rate(self):
        return self._freq

    @property
    def duration(self):
        return self._duration

    @property
    def time_step(self):
        return self._time[1] - self._time[0]

    @property
    def name(self):
        return self._name


if __name__ == '__main__':
    import time
    import timeit
    from gw_cqt_gpu import CQT

    fname = './data/tests/000a5b6e5c.npy'
    OUT_PATH = '../../data/tmp/train/'

    # tf.config.run_functions_eagerly(True)

    tf.debugging.set_log_device_placement(False)
    dev = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(dev)

    cqt = CQT(2048, (4, 64), 0.05, time_range=(0, 2.0, 1e-2), freq_range=(50, 250, 1))

    ts_batch = []
    fn_batch = []
    for i in range(80):
        tss = GwTimeseries.load(fname, 2048)
        sps = []
        for ts in tss:
            f, Pxx = ts.psd(fftlength=ts.duration, nperseg=2048, overlap=0.75, window=('tukey', 0.5))
            ts.apply_window(window=('tukey', 0.1))
            ts.whiten(psd_val=(f, Pxx))
            ts.filter(frange=(50, 250),
                      psd_val=(f, Pxx),
                      outlier_threshold=3.0)
            sps.append(ts.value)
        ts_batch.append(sps)
        fn_batch.append(os.path.splitext(os.path.basename(fname))[0] + '.png')

    ts_batch = np.array(ts_batch)

    with strategy.scope():
        start = time.time()
        data = cqt.transform(ts_batch)
        cqt_batch = cqt.interpolate(data)
        cqt.save(fn_batch, cqt_batch)
        end = time.time()
        print("tf.Tensor time elapsed: ", (end - start))

    # print(tf.autograph.to_code(gwu.cqt.python_function))

    # def power(x, y):
    #     result = tf.eye(10, dtype=tf.dtypes.int32)
    #     for _ in range(y):
    #         result = tf.matmul(x, result)
    #     return result
    #
    # @tf.function(jit_compile=True)
    # def power_tf(a):
    #     x, y = a
    #     i = tf.constant(0)
    #     result = tf.eye(10, dtype=tf.dtypes.int32)
    #     while tf.less(i, y):
    #         result = tf.matmul(x, result)
    #         i += 1
    #     return result
    #
    # @tf.function
    # def exp_tf(x, y, num):
    #     c = tf.constant([num, 1], tf.int32)
    #     xb = tf.tile(x, c)
    #     xb = tf.reshape(xb, [num, 10, 10])
    #     yb = tf.fill([num, 1], y)
    #     return tf.vectorized_map(power_tf, elems=(xb, yb))
    #     # return tf.map_fn(power_tf, elems=(xb, yb),
    #     #                  fn_output_signature=tf.TensorSpec(shape=[None, None], dtype=tf.dtypes.int32),
    #     #                  parallel_iterations=640)
    #
    # def power_np(x, y):
    #     result = np.eye(10, dtype=np.int32)
    #     for _ in range(y):
    #         result = np.matmul(x, result)
    #     return result
    #
    # with strategy.scope():
    #     x = tf.random.uniform(shape=[10, 10], minval=-1, maxval=2, dtype=tf.dtypes.int32)
    #
    #     print("Eager execution:", timeit.timeit(lambda: power(x, 100), number=10000))
    #
    #     power_as_graph = tf.function(power)
    #     print("Graph execution:", timeit.timeit(lambda: power_as_graph(x, 100), number=10000))
    #
    #     print("Graph 2 execution:", timeit.timeit(lambda: exp_tf(x, 100, 10000), number=1))
    #
    #     x = x.numpy()
    #     print("Graph execution:", timeit.timeit(lambda: power_np(x, 100), number=10000))

    # with strategy.scope():
    #     start = time.time()
    #     for i in range(5):
    #         tss = GwTimeseries.load(fname, 2048)
    #         sps = []
    #
    #         v = gwu.tests(tss)
    #         print(v)
    #
    #         for ts in tss:
    #             f, Pxx = ts.psd(fftlength=ts.duration, nperseg=2048, overlap=0.75, window=('tukey', 0.5))
    #             ts.apply_window(window=('tukey', 0.1))
    #             ts.whiten(psd_val=(f, Pxx))
    #             ts.filter(frange=(50, 250),
    #                       psd_val=(f, Pxx),
    #                       outlier_threshold=3.0)
    #             # cqt_peak, cqt_tiles = gwu.tf_qsearch(
    #             #     data=ts.value, sample_rate=ts.sample_rate,
    #             #     qrange=(1, 64), mismatch=0.05)
    #
    #         print(i)
    #     end = time.time()
    #     print("tf.Tensor time elapsed: ", (end - start))