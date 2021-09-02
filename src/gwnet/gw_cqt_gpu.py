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

from .gw_device_manager import TfDevice, get_first_logical_device
from .gw_timeseries import GwTimeseries


def _apply_normalization(cqt_interp, minmax=None, esp=1e-6):
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


def _apply_interpolation(cat_ragged, time_range, freq_range):
    xout = np.arange(time_range[0], time_range[1], time_range[2])
    fout = np.arange(freq_range[0], freq_range[1], freq_range[2])

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
        interp = interp2d(xout, freqs, tile_interp, kind='cubic')
        qspc = interp(xout, fout)
        cqt_interp.append(qspc)
    return np.array(cqt_interp)


def _interpolate_cubic1d(y, x_step, duration):
    ysize = tf.cast(tf.size(y), dtype=tf.dtypes.float32)
    oldstep = duration / ysize
    k = tf.cast(0.5 * oldstep / x_step, dtype=tf.dtypes.int32)
    yext = y[tf.newaxis, tf.newaxis, ..., tf.newaxis]
    new_size = tf.floor(duration / x_step)
    new_size = tf.cast(new_size, dtype=tf.dtypes.int32)
    yext = tf.image.resize(yext, [1, new_size], method='bicubic')
    return tf.cond(tf.less(k, 1), lambda: yext[0, 0, :, 0], lambda: yext[0, 0, k:-k, 0])


def _get_freq_range(q, duration, sample_rate, mismatch):
    qprime = q / 11 ** 0.5
    deltam = 2 * (mismatch / 3.) ** 0.5
    minf = 50 * q / (2 * math.pi * duration)
    maxf = sample_rate / 2 / (1 + 1 / qprime)

    fcum_mismatch = tf.math.log(maxf / minf) * (2 + q ** 2) ** 0.5 / 2.
    fcum_mismatch = tf.cast(fcum_mismatch, dtype=tf.float32)
    fcum_count = tf.math.ceil(fcum_mismatch / deltam)
    fcum_count = tf.cast(fcum_count, dtype=tf.float32)
    nfreq = tf.floor(tf.maximum(tf.constant(1.0, dtype=tf.float32), fcum_count))
    fstep = fcum_mismatch / nfreq
    return nfreq, fstep, minf


def _get_freq(nfreq, fstep, minf, q, duration):
    fstepmin = 1.0 / duration
    nfreq = tf.cast(nfreq, dtype=tf.int32)
    freqs = tf.range(nfreq, dtype=tf.float32)
    freqs = tf.add(freqs, 0.5)
    freqs = tf.multiply(freqs, 2 / (2 + q ** 2) ** 0.5 * fstep)
    freqs = tf.exp(freqs)
    freqs = tf.multiply(freqs, minf)
    freqs = tf.math.floordiv(freqs, fstepmin)
    freqs = tf.multiply(freqs, fstepmin)
    freqs, idx = tf.unique(freqs)
    return freqs


def _multiply_qwindow(task):
    fdata, q, frequency, duration, sample_rate, mismatch = task
    qprime = q / 11 ** 0.5
    deltam = 2 * (mismatch / 3.) ** 0.5
    windowsize = 2 * tf.floor(frequency / qprime * duration) + 1
    half = tf.floor((windowsize - 1) / 2)
    half = tf.cast(half, dtype=tf.int32)
    indeces = tf.range(-half, half + 1, 1, dtype=tf.float32)
    data_indeces = tf.round(indeces + (1 + frequency * duration))
    data_indeces = tf.cast(data_indeces, dtype=tf.int32)

    indeces = tf.multiply(indeces, qprime / frequency / duration)
    tcum_mismatch = duration * 2 * np.pi * frequency / q
    ntiles = 2 ** tf.math.ceil(tf.experimental.numpy.log2(tcum_mismatch / deltam))
    wins = tf.cast(windowsize, dtype=tf.float32)

    pad = ntiles - wins
    left_pad = (pad - 1.0) / 2.0
    left_pad = tf.cast(left_pad, dtype=tf.int32)
    right_pad = (pad + 1.0) / 2.0
    right_pad = tf.cast(right_pad, dtype=tf.int32)
    padding = tf.convert_to_tensor([[left_pad, right_pad]])

    norm = ntiles / (duration * sample_rate) * (315 * qprime / (128 * frequency)) ** 0.5

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


def _q_roi(mismatch, qrange):
    deltam = 2 * (mismatch / 3.) ** 0.5
    cumum = tf.math.log(qrange[1] / qrange[0]) / 2 ** 0.5
    nplanes = tf.floor(tf.maximum(tf.math.ceil(cumum / deltam), 1.0))

    dq = tf.cast(cumum / nplanes, dtype=tf.float32)
    qs = tf.range(nplanes, dtype=tf.float32)
    qs = tf.add(qs, 0.5)
    qs = tf.multiply(qs, 2 ** 0.5 * dq)
    qs = tf.exp(qs)
    qs = tf.multiply(qs, qrange[0])
    return qs


def _get_duration(ts_batch, sample_rate):
    return ts_batch.shape[2] / sample_rate


def _fft_batch(ts_batch):
    ts_batch = tf.convert_to_tensor(ts_batch)
    ts_batch = tf.reshape(ts_batch, [ts_batch.shape[0] * ts_batch.shape[1], ts_batch.shape[2]])
    ts_batch = tf.signal.rfft(ts_batch)
    return ts_batch


def _apply_qsearch(fs_batch, qrange, duration, sample_rate, mismatch):
    fs_batch_count = fs_batch.shape[0]

    def _q_search(q):
        nfreq, fstep, minf = _get_freq_range(q, duration, sample_rate, mismatch)
        freqs = _get_freq(nfreq, fstep, minf, q, duration)

        def _freq_search(task):
            q_l, freq_l = task
            q_l = tf.fill([fs_batch_count], tf.cast(q_l, dtype=tf.dtypes.float32))
            freq_l = tf.fill([fs_batch_count], tf.cast(freq_l, dtype=tf.dtypes.float32))
            duration_l = tf.fill([fs_batch_count], tf.cast(duration, dtype=tf.dtypes.float32))
            sample_rate_l = tf.fill([fs_batch_count], tf.cast(sample_rate, dtype=tf.dtypes.float32))
            mismatch_l = tf.fill([fs_batch_count], tf.cast(mismatch, dtype=tf.dtypes.float32))

            padded = tf.vectorized_map(_multiply_qwindow,
                                       (fs_batch, q_l, freq_l, duration_l, sample_rate_l, mismatch_l))

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

    qs = _q_roi(mismatch, qrange)
    energy = tf.map_fn(_q_search, qs,
                       fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.dtypes.float32))
    maxi = tf.argmax(energy, axis=0)
    return tf.gather(qs, maxi)


def _apply_cqtransform(q, fs_batch, duration, sample_rate, mismatch, x_step):
    nfreq, fstep, minf = _get_freq_range(q, duration, sample_rate, mismatch)

    def _apply_q(task):
        nfreq, fstep, minf, q, fdata = task
        freqs = _get_freq(nfreq, fstep, minf, q, duration)

        def _apply_freq(task):
            task_c = (task[0], task[1], task[2], duration, sample_rate, mismatch)
            padded = _multiply_qwindow(task_c)
            tenergy = tf.signal.ifft(padded)
            energy = tf.abs(tenergy) ** 2
            energy = tf.multiply(energy, 1.0 / tf.reduce_mean(energy))
            energy = tf.cast(energy, dtype=tf.dtypes.float32)
            return _interpolate_cubic1d(energy, x_step, duration)

        freqs_size = tf.size(freqs)
        q_c = tf.fill([freqs_size], q)
        i_c = tf.reshape(tf.tile(fdata, [freqs_size]), [freqs_size, tf.size(fdata)])

        energy = tf.map_fn(
            _apply_freq, (i_c, q_c, freqs),
            fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.dtypes.float32))

        return tf.ragged.stack([energy, tf.reshape(tf.cast(freqs, dtype=tf.dtypes.float32), [freqs_size, 1])])

    return tf.map_fn(_apply_q, (nfreq, fstep, minf, q, fs_batch),
                     fn_output_signature=tf.RaggedTensorSpec(shape=[None, None, None], dtype=tf.dtypes.float32))


def _cqt_batch_save(fn_batch, cqti_batch, out_path='./', size=None):
    for i, fn in enumerate(fn_batch):
        filename, file_extension = os.path.splitext(fn)
        fp = os.path.join(out_path, fn)
        os.makedirs(os.path.dirname(fp), exist_ok=True)

        result = cqti_batch[i]
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


def _cqt_batch_interpolate(data, original_data_shape, time_range, freq_range):
    cqti = _apply_interpolation(data, time_range, freq_range)
    cur_shape = cqti.shape
    cqti = np.reshape(cqti, [original_data_shape[0], original_data_shape[1], cur_shape[1], cur_shape[2]])
    cqtn = _apply_normalization(cqti)
    return cqtn


@tf.function
def _cqt_batch_process(ts_batch, sample_rate, mismatch, qrange, time_step):
    fs_batch = _fft_batch(ts_batch)
    duration = _get_duration(ts_batch, sample_rate)
    q = _apply_qsearch(fs_batch, qrange, duration, sample_rate, mismatch)
    return _apply_cqtransform(q, fs_batch, duration, sample_rate, mismatch, time_step)


class CQTProcessor:
    def __init__(self, mode: Literal[TfDevice.TPU, TfDevice.GPU, TfDevice.CPU]):
        self._in_path = None
        self._task = []

        self._mode = mode
        self._device = get_first_logical_device(self._mode)

    def scan_directory(self, input_path, output_path, reject_if_exists=True, imitate_loaded=None):
        self._task = []
        self._in_path = input_path
        for root, dirs, files in os.walk(input_path):
            rel_path = root.replace(input_path, '')

            if imitate_loaded is not None:
                files_tmp = []
                for i in range(imitate_loaded):
                    files_tmp = [*files_tmp, *files]
                files = files_tmp

            for fname in files:
                file_name, file_ext = os.path.splitext(fname)

                if file_ext != '.npy':
                    continue

                in_fn = os.path.join(rel_path, fname)
                out_fn = os.path.join(rel_path, file_name + '.png')

                if reject_if_exists:
                    full_out_path = os.path.join(output_path, out_fn)
                    if os.path.exists(full_out_path):
                        continue

                self._task.append({'in': in_fn, 'out': out_fn})

        return dict(path=output_path, files_count=len(self._task))

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

        with tf.device(self._device.name):
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

                original_data_shape = ts_batch.shape
                time_step = time_range[2]
                data = _cqt_batch_process(ts_batch, sample_rate, mismatch, qrange, time_step)
                cqt_batch = _cqt_batch_interpolate(data, original_data_shape, time_range, freq_range)
                _cqt_batch_save(
                    fn_batch=fn_batch, cqti_batch=cqt_batch,
                    out_path=out_path, size=img_size)

                if verbose:
                    end = time.time()
                    print(f'Batch {index+1}/{batches_count} processed during {end - start}s.')

                index += 1
