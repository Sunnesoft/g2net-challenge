from src.gwnet import GwTimeseries, GwSpectrogram
import os
import time

if __name__ == '__main__':
    input_path = '../data/test/'
    output_path = '../data/tmp/train/'

    batch_size = 1024
    sample_rate = 2048
    qrange = (4, 64)
    mismatch = 0.05
    time_range = (0, 2.0, 1e-2)
    freq_range = (50, 250, 1)
    img_size = (512, 512)

    start = time.time()

    for root, dirs, files in os.walk(input_path):
        rel_path = root.replace(input_path, '')

        files_tmp = []
        for i in range(200):
            files_tmp = [*files_tmp, *files]
        files = files_tmp

        for fname in files:
            file_name, file_ext = os.path.splitext(fname)

            if file_ext != '.npy':
                continue

            in_fn = os.path.join(input_path, fname)
            out_fn = os.path.join(rel_path, file_name + '.png')

            sps = []
            tss = GwTimeseries.load(in_fn, sample_rate)
            for ts in tss:
                sp = GwSpectrogram(ts)
                # sp.cwt(out_time_range=(0, ts.duration, 1e-2), out_freq_range=(50, 500, 1), prange=np.arange(1, 64, 0.5))
                sp.cqt(out_time_range=time_range, out_freq_range=freq_range, qrange=qrange, qmismatch=mismatch)
                sp.normalize()
                sps.append(sp)

            out_fn = os.path.join(output_path, out_fn)
            GwSpectrogram.save(out_fn, sps, mode='depth_stacked', size=img_size)

    end = time.time()
    print(f'All data processed during {end - start}s.')