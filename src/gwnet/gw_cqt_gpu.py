import tensorflow as tf
import numpy as np
import math
import os
import random
import time

from scipy.interpolate import interp2d
from PIL import Image

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .gw_device_manager import TfDevice, init_processing_device
from .gw_timeseries import GwTimeseries


class CQT:
    def __init__(self, sample_rate, qrange, mismatch, time_range, freq_range):
        self._sample_rate = sample_rate
        self._qrange = qrange
        self._mismatch = mismatch
        self._x_step = time_range[2]
        self._xout = np.arange(time_range[0], time_range[1], time_range[2])
        self._fout = np.arange(freq_range[0], freq_range[1], freq_range[2])

        self._deltam = 2 * (self._mismatch / 3.) ** 0.5
        cumum = tf.math.log(self._qrange[1] / self._qrange[0]) / 2 ** 0.5

        self._nplanes = tf.floor(tf.maximum(tf.math.ceil(cumum / self._deltam), 1.0))

        dq = tf.cast(cumum / self._nplanes, dtype=tf.float32)
        self._qs = tf.range(self._nplanes, dtype=tf.float32)
        self._qs = tf.add(self._qs, 0.5)
        self._qs = tf.multiply(self._qs, 2 ** 0.5 * dq)
        self._qs = tf.exp(self._qs)
        self._qs = tf.multiply(self._qs, self._qrange[0])

        self._duration = None
        self._fstepmin = None
        self._num = None
        self._fs = None
        self._oshape = None

    def _set_batched(self, ts_batch):
        if not tf.is_tensor(ts_batch):
            ts_batch = tf.convert_to_tensor(ts_batch)

        self._oshape = ts_batch.shape
        if len(self._oshape) == 3:
            ts_batch = tf.reshape(ts_batch, [self._oshape[0] * self._oshape[1], self._oshape[2]])

        new_shape = ts_batch.shape
        self._duration = new_shape[1] / self._sample_rate
        self._fstepmin = 1.0 / self._duration
        self._num = new_shape[0]
        self._fs = tf.signal.rfft(ts_batch)

    def _multiply_qwindow(self, task):
        fi, q, frequency = task
        fdata = self._fs[fi]
        qprime = q / 11 ** 0.5
        windowsize = 2 * tf.floor(frequency / qprime * self._duration) + 1
        half = tf.floor((windowsize - 1) / 2)
        half = tf.cast(half, dtype=tf.int32)
        indeces = tf.range(-half, half + 1, 1, dtype=tf.float32)
        data_indeces = tf.round(indeces + (1 + frequency * self._duration))
        data_indeces = tf.cast(data_indeces, dtype=tf.int32)

        indeces = tf.multiply(indeces, qprime / frequency / self._duration)
        tcum_mismatch = self._duration * 2 * np.pi * frequency / q
        ntiles = 2 ** tf.math.ceil(tf.experimental.numpy.log2(tcum_mismatch / self._deltam))
        wins = tf.cast(windowsize, dtype=tf.float32)

        pad = ntiles - wins
        left_pad = (pad - 1.0) / 2.0
        left_pad = tf.cast(left_pad, dtype=tf.int32)
        right_pad = (pad + 1.0) / 2.0
        right_pad = tf.cast(right_pad, dtype=tf.int32)
        padding = tf.convert_to_tensor([[left_pad, right_pad]])

        norm = ntiles / (self._duration * self._sample_rate) * (315 * qprime / (128 * frequency)) ** 0.5

        indeces = tf.square(indeces)
        ones = tf.ones_like(indeces)
        indeces = tf.subtract(ones, indeces)
        indeces = tf.square(indeces)
        indeces = tf.multiply(indeces, norm)
        window = tf.cast(indeces, dtype=tf.complex128)

        values = tf.gather(fdata, data_indeces)
        values = tf.multiply(values, window)
        padded = tf.pad(values, padding)
        return padded

    def _get_freq(self, nfreq, fstep, minf, q):
        nfreq = tf.cast(nfreq, dtype=tf.int32)
        freqs = tf.range(nfreq, dtype=tf.float32)
        freqs = tf.add(freqs, 0.5)
        freqs = tf.multiply(freqs, 2 / (2 + q ** 2) ** 0.5 * fstep)
        freqs = tf.exp(freqs)
        freqs = tf.multiply(freqs, minf)
        freqs = tf.math.floordiv(freqs, self._fstepmin)
        freqs = tf.multiply(freqs, self._fstepmin)
        freqs, idx = tf.unique(freqs)
        return freqs

    def _get_freq_range(self, q):
        qprime = q / 11 ** 0.5
        minf = 50 * q / (2 * math.pi * self._duration)
        maxf = self._sample_rate / 2 / (1 + 1 / qprime)

        fcum_mismatch = tf.math.log(maxf / minf) * (2 + q ** 2) ** 0.5 / 2.
        fcum_mismatch = tf.cast(fcum_mismatch, dtype=tf.float32)
        fcum_count = tf.math.ceil(fcum_mismatch / self._deltam)
        fcum_count = tf.cast(fcum_count, dtype=tf.float32)
        nfreq = tf.floor(tf.maximum(tf.constant(1.0, dtype=tf.float32), fcum_count))
        fstep = fcum_mismatch / nfreq
        return nfreq, fstep, minf

    def _apply_qsearch(self):
        def _q_search(q):
            nfreq, fstep, minf = self._get_freq_range(q)
            freqs = self._get_freq(nfreq, fstep, minf, q)

            def _freq_search(task):
                q_l, freq_l = task
                q_l = tf.fill([self._num], tf.cast(q_l, dtype=tf.dtypes.float32))
                freq_l = tf.fill([self._num], tf.cast(freq_l, dtype=tf.dtypes.float32))
                fsi = tf.range(self._num, dtype=tf.dtypes.int32)
                padded = tf.vectorized_map(self._multiply_qwindow, (fsi, q_l, freq_l))
                tenergy = tf.signal.ifft(padded)
                energy = tf.abs(tenergy) ** 2
                energy = tf.multiply(energy, 1.0 / tf.reduce_mean(energy))
                energy = tf.reduce_max(energy, axis=1)
                return tf.cast(energy, dtype=tf.dtypes.float32)

            q_c = tf.fill([tf.size(freqs)], q)
            v = tf.map_fn(_freq_search, (q_c, freqs),
                          fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.dtypes.float32))
            v = tf.reduce_max(v, axis=0)
            return v

        energy = tf.map_fn(_q_search, self._qs,
                           fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.dtypes.float32))
        maxi = tf.argmax(energy, axis=0)
        return tf.gather(self._qs, maxi)

    def _interpolate_cubic1d(self, y):
        ysize = tf.cast(tf.size(y), dtype=tf.dtypes.float32)
        oldstep = self._duration / ysize
        k = tf.cast(0.5 * oldstep / self._x_step, dtype=tf.dtypes.int32)
        yext = y[tf.newaxis, tf.newaxis, ..., tf.newaxis]
        new_size = tf.floor(self._duration / self._x_step)
        new_size = tf.cast(new_size, dtype=tf.dtypes.int32)
        yext = tf.image.resize(yext, [1, new_size], method='bicubic')
        return tf.cond(tf.less(k, 1), lambda: yext[0, 0, :, 0], lambda: yext[0, 0, k:-k, 0])

    def _apply_qtransform(self, q):
        nfreq, fstep, minf = self._get_freq_range(q)

        def _apply_q(task):
            nfreq, fstep, minf, q, fsi = task
            freqs = self._get_freq(nfreq, fstep, minf, q)

            def _apply_freq(task):
                padded = self._multiply_qwindow(task)
                tenergy = tf.signal.ifft(padded)
                energy = tf.abs(tenergy) ** 2
                energy = tf.multiply(energy, 1.0 / tf.reduce_mean(energy))
                energy = tf.cast(energy, dtype=tf.dtypes.float32)
                return self._interpolate_cubic1d(energy)

            q_c = tf.fill([tf.size(freqs)], q)
            i_c = tf.fill([tf.size(freqs)], fsi)
            energy = tf.map_fn(
                _apply_freq, (i_c, q_c, freqs),
                fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.dtypes.float32))

            return tf.ragged.stack([energy, tf.reshape(tf.cast(freqs, dtype=tf.dtypes.float32), [tf.size(freqs), 1])])

        fsi = tf.range(self._num, dtype=tf.dtypes.int32)
        return tf.map_fn(_apply_q, (nfreq, fstep, minf, q, fsi),
                         fn_output_signature=tf.RaggedTensorSpec(shape=[None, None, None], dtype=tf.dtypes.float32))

    def _apply_interpolation(self, cat_ragged):
        cqt_interp = []
        for cqt in cat_ragged:
            tile, freqs = cqt
            tile = tile.numpy()
            tile_interp = tile

            # tile_interp = []
            # for row in tile:
            #     xrow = np.linspace(0.0, self._duration, len(row))
            #     interp = InterpolatedUnivariateSpline(xrow, row)
            #     tile_interp.append(interp(xout))

            tile_interp = np.array(tile_interp)
            freqs = np.squeeze(freqs.numpy())
            interp = interp2d(self._xout, freqs, tile_interp, kind='cubic')
            qspc = interp(self._xout, self._fout)
            cqt_interp.append(qspc)
        return np.array(cqt_interp)

    def _apply_normalization(self, cqt_interp, minmax=None, esp=1e-6):
        mean = np.mean(cqt_interp)
        std = np.std(cqt_interp)

        cqt_interp = (cqt_interp - mean) / (std + esp)

        if minmax is None:
            qspectr_min, qspectr_max = cqt_interp.min(), cqt_interp.max()
        else:
            qspectr_min, qspectr_max = minmax

        cqt_interp[cqt_interp < qspectr_min] = qspectr_min
        cqt_interp[cqt_interp > qspectr_max] = qspectr_max
        cqt_interp = 255 * (cqt_interp - qspectr_min) / (qspectr_max - qspectr_min)
        cqt_interp = cqt_interp.astype(np.uint8)
        cqt_interp = np.flip(cqt_interp, 2)
        return cqt_interp

    @tf.function
    def transform(self, ts_batch):
        self._set_batched(ts_batch)
        q = self._apply_qsearch()
        return self._apply_qtransform(q)

    def interpolate(self, data):
        cqti = self._apply_interpolation(data)

        if len(self._oshape) == 3:
            cur_shape = cqti.shape
            cqti = np.reshape(cqti, [self._oshape[0], self._oshape[1], cur_shape[1], cur_shape[2]])

        cqtn = self._apply_normalization(cqti)
        return cqtn

    def save(self, fn_batch, cqti_batch, out_path='./', size=None):
        for i, fn in enumerate(fn_batch):
            filename, file_extension = os.path.splitext(fn)
            fp = os.path.join(out_path, fn)
            os.makedirs(os.path.dirname(fp), exist_ok=True)

            result = cqti_batch[i]
            if len(self._oshape) == 3:
                res_img = np.zeros([result[0].shape[0], result[0].shape[1], result.shape[0]], dtype=np.uint8)
                for j in range(result.shape[0]):
                    res_img[:, :, j] = result[j, :, :]
                result = res_img

            if file_extension in ['.npy']:
                np.save(fp, result)
            elif file_extension in ['.png']:
                img = Image.fromarray(result)
                if size is not None:
                    img = img.resize(size)
                img.save(fp)
            else:
                raise ValueError('invalid file extension, use .png or .npy')


class CQTProcessor:
    def __init__(self, mode: Literal[TfDevice.TPU, TfDevice.GPU, TfDevice.TPU]):
        self._in_path = None
        self._task = []

        self._mode = mode
        self._strategy = init_processing_device(self._mode)

    def scan_directory(self, path):
        self._task = []
        self._in_path = path
        for root, dirs, files in os.walk(path):
            rel_path = root.replace(path, '')

            for fname in files:
                file_name, file_ext = os.path.splitext(fname)

                if file_ext != '.npy':
                    continue

                in_fn = os.path.join(rel_path, fname)
                out_fn = os.path.join(rel_path, file_name + '.png')
                self._task.append({'in': in_fn, 'out': out_fn})

        return dict(path=path, files_count=len(self._task))

    def _crop_tasks_batch(self, batch_size=1):
        task_count = len(self._task)
        for ndx in range(0, task_count, batch_size):
            yield self._task[ndx:min(ndx + batch_size, task_count)]

    def _batches_count(self, batch_size):
        task_count = len(self._task)
        return math.ceil(task_count / batch_size)

    def _shuffle_tasks(self):
        random.shuffle(self._task)

    def run(self, batch_size, out_path, sample_rate,
            qrange, mismatch, time_range, freq_range,
            img_size, shuffle_tasks, verbose):
        if shuffle_tasks:
            self._shuffle_tasks()

        with self._strategy.scope():
            cqt = CQT(
                sample_rate=sample_rate,
                qrange=qrange,
                mismatch=mismatch,
                time_range=time_range,
                freq_range=freq_range)

            index = 0
            batches_count = self._batches_count(batch_size)
            for tasks in self._crop_tasks_batch(batch_size):
                start = time.time()

                ts_batch = []
                fn_batch = []

                for task in tasks:
                    in_fn = os.path.join(self._in_path, task['in'])
                    out_fn = task['out']

                    tss = GwTimeseries.load(in_fn, sample_rate)
                    sps = [ts.value for ts in tss]
                    ts_batch.append(sps)
                    fn_batch.append(out_fn)
                ts_batch = np.array(ts_batch)

                data = cqt.transform(ts_batch)
                cqt_batch = cqt.interpolate(data)
                cqt.save(fn_batch=fn_batch, cqti_batch=cqt_batch,
                         out_path=out_path, size=img_size)

                if verbose:
                    end = time.time()
                    print(f'Batch {index}/{batches_count} processed during {end - start}s.')

                index += 1
