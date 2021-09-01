from src.gwnet import GwTimeseries, GwSpectrogram


if __name__ == '__main__':
    fname = '../data/tests/111012cee3.npy'
    OUT_PATH = '../data/tmp/train/'

    for i in range(1):
        tss = GwTimeseries.load(fname, 2048)
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
