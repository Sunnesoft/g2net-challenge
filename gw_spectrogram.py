import os
import matplotlib.pyplot as plt
import numpy as np
import pywt
import scaleogram as scg
from PIL import Image
from scipy.interpolate import interp2d

import gw_utils as gwu
import gw_timeseries as gwts


class GwSpectrogram:
    def __init__(self, timeseries):
        self._ts = timeseries
        self._time = None
        self._freq = None
        self._value = None

    def cqt(self, out_time_range, out_freq_range, qrange=(4, 64), qmismatch=0.2):
        cqt_peak, cqt_tiles = gwu.qsearch(
            data=self._ts.value, sample_rate=self._ts.sample_rate,
            qrange=qrange, mismatch=qmismatch)

        self._time, self._freq, self._value, q = gwu.interpolate_cqt_data(
            cqt_peak, out_time_range, out_freq_range)

    def cwt(self, out_time_range, out_freq_range, wavelete_name='shan0.5-2', prange=np.arange(1, 64)):
        scales = scg.periods2scales(prange)
        coefficients, self._freq = pywt.cwt(self._ts.value, scales, wavelete_name)
        self._freq *= self._ts.sample_rate
        self._value = abs(coefficients) ** 2
        self._time = np.copy(self._ts.time)

        xout = np.arange(out_time_range[0], out_time_range[1], out_time_range[2])
        fout = np.arange(out_freq_range[0], out_freq_range[1], out_freq_range[2])
        interp = interp2d(self._time, self._freq, self._value, kind='cubic')
        self._value = interp(xout, fout).astype(float, casting="same_kind", copy=False)
        self._time = xout
        self._freq = fout

    def normalize(self, minmax=None, esp=1e-6):
        mean = self._value.mean()
        std = self._value.std()

        self._value = (self._value - mean) / (std + esp)

        if minmax is None:
            qspectr_min, qspectr_max = self._value.min(), self._value.max()
        else:
            qspectr_min, qspectr_max = minmax

        self._value[self._value < qspectr_min] = qspectr_min
        self._value[self._value > qspectr_max] = qspectr_max
        self._value = 255 * (self._value - qspectr_min) / (qspectr_max - qspectr_min)
        self._value = self._value.astype(np.uint8)
        self._value = np.flip(self._value, 0)

    def show_value(self):
        plt.imshow(self._value, cmap=plt.cm.seismic, aspect='auto',
                   vmax=self._value.max(), vmin=self._value.min())
        plt.show()

    @property
    def value(self):
        return self._value

    @property
    def time(self):
        return self._time

    @property
    def frequencies(self):
        return self._freq

    @staticmethod
    def save(fp, data, mode='depth_stacked', size=None):
        if isinstance(data, GwSpectrogram):
            result = data.value
        elif isinstance(data, list):
            result = []
            for d in data:
                result.append(d.value)

            if mode == 'vert_stacked':
                result = np.vstack(result)
            elif mode == 'depth_stacked':
                res_img = np.zeros([result[0].shape[0], result[0].shape[1], len(result)], dtype=np.uint8)
                for i, v in enumerate(result):
                    res_img[:, :, i] = v
                result = res_img
        else:
            raise ValueError('unexpected type of spectrogram data')

        filename, file_extension = os.path.splitext(fp)
        if file_extension in ['.npy']:
            np.save(fp, result)
        elif file_extension in ['.png']:
            img = Image.fromarray(result)
            if size is not None:
                img = img.resize(size)
            img.save(fp)
        else:
            raise ValueError('invalid file extension, use .png or .npy')


if __name__ == '__main__':
    fname = './data/test/111012cee3.npy'
    OUT_PATH = './data/tmp/train/'

    for i in range(1):
        tss = gwts.GwTimeseries.load(fname, 2048)
        sps = []

        for ts in tss:
            f, Pxx = ts.psd(fftlength=ts.duration, nperseg=2048, overlap=0.75, window=('tukey', 0.5))
            ts.apply_window(window=('tukey', 0.1))
            ts.whiten(psd_val=(f, Pxx))
            ts.filter(frange=(50, 250),
                      psd_val=(f, Pxx),
                      outlier_threshold=3.0)
            sp = GwSpectrogram(ts)
            # sp.cwt(out_time_range=(0, ts.duration, 1e-2), out_freq_range=(50, 500, 1), prange=np.arange(1, 64, 0.5))
            sp.cqt(out_time_range=(0, ts.duration, 1e-2), out_freq_range=(50, 250, 1), qrange=(1, 64), qmismatch=0.05)
            sp.normalize()
            # sp.show_value()
            sps.append(sp)

        # GwTimeseries.save(OUT_PATH + fname, tss)
        GwSpectrogram.save(OUT_PATH + '111012cee3.png', sps, mode='depth_stacked', size=(512, 512))
        print(i)
