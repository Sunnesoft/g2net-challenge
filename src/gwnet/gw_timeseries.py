import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import irfft, rfft, rfftfreq
from scipy.interpolate import interp1d

import tensorflow as tf
from . import gw_utils as gwu


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