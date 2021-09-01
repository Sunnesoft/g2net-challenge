import math
import operator
from functools import reduce
from typing import Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import irfft, rfft, fft, ifft, fftfreq, rfftfreq
from scipy.interpolate import interp1d, interp2d, InterpolatedUnivariateSpline
from scipy.special import expit


def _as_float(x):
    try:
        return float(x.value)
    except AttributeError:
        return float(x)


def design_iir(wp, ws, sample_rate, gpass, gstop, ftype='cheby1'):
    nyq = sample_rate / 2.
    wp = np.atleast_1d(wp)
    ws = np.atleast_1d(ws)
    wp /= nyq
    ws /= nyq
    return signal.iirdesign(wp, ws, gpass, gstop, ftype=ftype, output='zpk')


def butter_bandpass(lowcut, highcut, sample_rate, order=5):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    return signal.butter(order, [low, high], analog=False, btype='band', output='zpk')


def bandpass(flow, fhigh, sample_rate, fstop=None, gpass=2, gstop=30):
    sample_rate = _as_float(sample_rate)
    flow = _as_float(flow)
    fhigh = _as_float(fhigh)
    if fstop is None:
        fstop = (flow * 2 / 3.,
                 min(fhigh * 1.5, sample_rate / 2.))
    fstop = (_as_float(fstop[0]), _as_float(fstop[1]))
    return design_iir((flow, fhigh), fstop, sample_rate, gpass, gstop)


def highpass(frequency, sample_rate, fstop=None, gpass=2, gstop=30):
    sample_rate = _as_float(sample_rate)
    frequency = _as_float(frequency)
    if fstop is None:
        fstop = frequency * 2 / 3.
    return design_iir(frequency, fstop, sample_rate, gpass, gstop)


def notch(frequency, sample_rate, **kwargs):
    nyq = 0.5 * sample_rate
    df = 1.0
    df2 = 0.1
    low1 = (frequency - df) / nyq
    high1 = (frequency + df) / nyq
    low2 = (frequency - df2) / nyq
    high2 = (frequency + df2) / nyq
    kwargs.setdefault('gpass', 1)
    kwargs.setdefault('gstop', 10)
    kwargs.setdefault('ftype', 'ellip')
    return signal.iirdesign([low1, high1], [low2, high2], output='zpk', **kwargs)


def concatenate_zpks(*zpks):
    zeros, poles, gains = zip(*zpks)
    return (np.concatenate(zeros),
            np.concatenate(poles),
            reduce(operator.mul, gains, 1))


def apply_filter(data, *filt):
    sos = signal.zpk2sos(*filt)
    return signal.sosfiltfilt(sos, data)


def psd(data, sample_rate: int, fftlength: float = None, overlap: float = 0,
        window: Union[str, np.ndarray, Tuple[str, float]] = 'hann'):
    nfft = int(sample_rate * fftlength)
    nwindow = window
    if not isinstance(window, np.ndarray):
        nwindow = signal.get_window(window, nfft)
    noverlap = int(overlap * nwindow.size)
    f, Pxx = signal.welch(data, fs=sample_rate,
                          window=nwindow, noverlap=noverlap, nfft=nfft)
    return f, Pxx


def asd(data, sample_rate: int, fftlength: float = None, overlap: float = 0,
        window: Union[str, np.ndarray, Tuple[str, float]] = 'hann'):
    f, Pxx = psd(data=data, sample_rate=sample_rate,
                 fftlength=fftlength, overlap=overlap, window=window)
    return f, Pxx ** (1 / 2.)


def planck(N, nleft=0, nright=0):
    w = np.ones(N)
    if nleft:
        w[0] *= 0
        zleft = np.array([nleft * (1. / k + 1. / (k - nleft))
                          for k in range(1, nleft)])
        w[1:nleft] *= expit(-zleft)
    if nright:
        w[N - 1] *= 0
        zright = np.array([-nright * (1. / (k - nright) + 1. / k)
                           for k in range(1, nright)])
        w[N - nright:N - 1] *= expit(-zright)
    return w


def interpolate(f, data, df_new):
    f0 = f[0]
    size = len(f)
    df = f[1] - f[0]
    N = (size - 1) * (df / df_new) + 1
    fsamples = np.arange(0, np.rint(N), dtype=float) * df_new + f0
    data_new = np.interp(fsamples, f, data)
    return fsamples, data_new


def truncate_transfer(transfer, ncorner=None):
    nsamp = transfer.size
    ncorner = ncorner if ncorner else 0
    out = transfer.copy()
    out[0:ncorner] = 0
    out[ncorner:nsamp] *= planck(nsamp - ncorner, nleft=5, nright=5)
    return out


def truncate_impulse(impulse, ntaps, window='hanning'):
    out = impulse.copy()
    trunc_start = int(ntaps / 2)
    trunc_stop = out.size - trunc_start
    window = signal.get_window(window, ntaps)
    out[0:trunc_start] *= window[trunc_start:ntaps]
    out[trunc_stop:out.size] *= window[0:trunc_start]
    out[trunc_start:trunc_stop] = 0
    return out


def fir_from_transfer(transfer, ntaps, window='hanning', ncorner=None):
    transfer = truncate_transfer(transfer, ncorner=ncorner)
    impulse = irfft(transfer)
    impulse = truncate_impulse(impulse, ntaps=ntaps, window=window)
    out = np.roll(impulse, int(ntaps / 2 - 1))[0:ntaps]
    return out


def convolve(data, fir, window='hanning'):
    pad = int(np.ceil(fir.size / 2))
    nfft = min(8 * fir.size, len(data))
    # condition the input data
    in_value = data.copy()
    window = signal.get_window(window, fir.size)
    in_value[:pad] *= window[:pad]
    in_value[-pad:] *= window[-pad:]
    # if FFT length is long enough, perform only one convolution
    if nfft >= len(data) / 2:
        conv = signal.fftconvolve(in_value, fir, mode='same')
    # else use the overlap-save algorithm
    else:
        nstep = nfft - 2 * pad
        conv = np.zeros(len(data))
        # handle first chunk separately
        conv[:nfft - pad] = signal.fftconvolve(in_value[:nfft], fir, mode='same')[:nfft - pad]
        # process chunks of length nstep
        k = nfft - pad
        while k < len(data) - nfft + pad:
            yk = signal.fftconvolve(in_value[k - pad:k + nstep + pad], fir, mode='same')
            conv[k:k + yk.size - 2 * pad] = yk[pad:-pad]
            k += nstep
        # handle last chunk separately
        conv[-nfft + pad:] = signal.fftconvolve(in_value[-nfft:], fir, mode='same')[-nfft + pad:]
    return conv


def whiten_conv(data, sample_rate, fftlength, overlap=0, window='hanning', fduration=2, highpass=None):
    duration = len(data) / sample_rate
    f_asd, v_asd = asd(data=data, sample_rate=sample_rate, fftlength=fftlength, overlap=overlap, window=window)
    f_asd, v_asd = interpolate(f_asd, v_asd, 1. / duration)
    df = f_asd[1] - f_asd[0]
    ncorner = int(highpass / df) if highpass else 0
    ntaps = int(fduration * sample_rate)
    tdw = fir_from_transfer(1 / v_asd, ntaps=ntaps, window=window, ncorner=ncorner)
    data_detr = signal.detrend(data, type='constant')
    out = convolve(data_detr, tdw, window=window)
    return out * np.sqrt(2 / sample_rate)


def apply_window(data, kind: Union[Tuple[str, float], None, float, np.ndarray] = ('tukey', 0.25)):
    if kind is None:
        return data

    nwindow = kind
    if not isinstance(nwindow, np.ndarray):
        nwindow = signal.get_window(nwindow, len(data))

    windowed = data * nwindow
    return windowed


def whiten(data, sample_rate: int, fftlength: float, nperseg: int, overlap: float = 0.5, window=None):
    psd_window = signal.windows.tukey(nperseg, alpha=0.25)
    f, Pxx = psd(data=data, sample_rate=sample_rate,
                 fftlength=fftlength, overlap=overlap, window=psd_window)
    psd_interp = interp1d(f, Pxx)

    windowed = apply_window(data, window)
    Nt = len(windowed)
    dt = 1.0 / sample_rate
    freqs = rfftfreq(Nt, dt)
    hf = rfft(windowed)
    norm = 1. / np.sqrt(1. / (dt * 2))
    white_hf = hf / np.sqrt(psd_interp(freqs)) * norm
    white_ht = irfft(white_hf, n=Nt)
    return white_ht


def spectrogram(data, sample_rate, fftlength, window=None, overlap=None):
    nfft = int(sample_rate * fftlength)
    noverlap = int(overlap * sample_rate)
    nwindow = signal.get_window(window, nfft) if window is not None else None

    if noverlap is None:
        noverlap = int(nfft / 2)

    starts = np.arange(0, len(data), nfft - noverlap, dtype=int)
    starts = starts[starts + nfft < len(data)]

    if nfft % 2:
        numFreqs = (nfft + 1) // 2
    else:
        numFreqs = nfft // 2 + 1

    xns = []
    for start in starts:
        data_piece = data[start:start + nfft].copy()
        if isinstance(nwindow, np.ndarray):
            data_piece *= nwindow

        data_piece_fr = fft(data_piece, axis=0)[:numFreqs]
        xns.append(data_piece_fr)
    result = np.array(xns).T
    result = np.conj(result) * result
    result /= sample_rate
    result /= (np.abs(nwindow) ** 2).sum()

    freqs = fftfreq(nfft, 1 / sample_rate)[:numFreqs]

    if not nfft % 2:
        freqs[-1] *= -1

    result = 10 * np.log10(result)

    return result, freqs, starts / sample_rate


def plot_spectrogram(spec, freq, time, spec_cmap='ocean', extent=None,
                     interpolation: Union[str, None] = 'bicubic',
                     plt_scene=None, label=None, mode='main', yscale=None):
    extent_v = time[0], time[-1], freq[0], freq[-1]

    if plt_scene is None:
        plt_scene = plt

    if extent is not None:
        b = (extent[3] - extent[2])
        if yscale == 'log':
            b = np.log(b)

        aspect = (extent[1] - extent[0]) / b
    else:
        b = (freq[-1] - freq[0])
        if yscale == 'log':
            b = np.log(b)

        aspect = (time[-1] - time[0]) / b
        extent = extent_v

    plt_spec = plt_scene.imshow(
        spec.real, origin='lower',
        cmap=spec_cmap, aspect=aspect, extent=extent_v,
        interpolation=interpolation)

    if mode == 'main':
        plt_scene.ylabel('Frequency (Hz)')
        plt_scene.xlabel('Time (s)')
        plt_scene.axis(extent)
        plt_scene.colorbar()
        if label is not None:
            plt.title(label)
    elif mode in ['subplot', 'subplot_left', 'subplot_bottom', 'subplot_bottom_left']:
        if mode in ['subplot_left', 'subplot_bottom_left']:
            plt_scene.set_ylabel('Frequency (Hz)')
        if mode in ['subplot_bottom', 'subplot_bottom_left']:
            plt_scene.set_xlabel('Time (s)')
        plt_scene.axis(extent)
        if label is not None:
            plt_scene.set_title(label)

    return plt_spec


def apply_qwindow(fdata, q, qprime, frequency, deltam, sample_rate, duration):
    # duration = len(fdata) / sample_rate

    windowsize = 2 * int(frequency / qprime * duration) + 1

    half = int((windowsize - 1) / 2)
    indeces = np.arange(-half, half + 1, 1.0, dtype=np.float64)
    data_indeces = np.round(indeces + (1 + frequency * duration)).astype(int)

    np.multiply(indeces, qprime / frequency / duration, out=indeces)
    tcum_mismatch = duration * 2 * np.pi * frequency / q
    ntiles = 2 ** math.ceil(math.log(tcum_mismatch / deltam, 2))

    pad = ntiles - windowsize
    padding = (int((pad - 1) / 2.), int((pad + 1) / 2.))

    norm = ntiles / (duration * sample_rate) * (315 * qprime / (128 * frequency)) ** 0.5

    np.square(indeces, out=indeces)
    np.subtract(1.0, indeces, out=indeces)
    np.square(indeces, out=indeces)
    np.multiply(indeces, norm, out=indeces)
    window = np.asarray(indeces, dtype=np.complex128)

    np.multiply(fdata[data_indeces], window, out=window)
    padded = np.zeros(ntiles, dtype=window.dtype)
    padded[padding[0]:-padding[1]] = window

    return padded


def qsearch(data, sample_rate, qrange, mismatch, output_all_tiles=False):
    duration = len(data) / sample_rate

    deltam = 2 * (mismatch / 3.) ** 0.5
    cumum = math.log(qrange[1] / qrange[0]) / 2 ** 0.5
    nplanes = int(max(math.ceil(cumum / deltam), 1))
    dq = cumum / nplanes
    qlist = [qrange[0] * math.exp(2 ** 0.5 * dq * (i + .5)) for i in range(nplanes)]

    nfft = len(data)
    fdata = rfft(data, n=nfft) / nfft
    fdata[1:] *= 2.0

    glob_peak = {'energy': 0, 'freqs': None, 'tile': None, 'duration': duration, 'q': 0}
    glob_tiles = []

    for q in qlist:
        qprime = q / 11 ** 0.5
        minf = 50 * q / (2 * math.pi * duration)
        maxf = sample_rate / 2 / (1 + 1 / qprime)

        fcum_mismatch = math.log(maxf / minf) * (2 + q ** 2) ** 0.5 / 2.
        nfreq = int(max(1, math.ceil(fcum_mismatch / deltam)))
        fstep = fcum_mismatch / nfreq
        fstepmin = 1.0 / duration

        freqs = []
        for i in range(nfreq):
            f = minf * math.exp(2 / (2 + q ** 2) ** 0.5 * (i + .5) * fstep) // fstepmin * fstepmin
            if f in freqs:
                continue
            freqs.append(f)

        peak_energy = 0.0
        tile = []

        for freq in freqs:
            padded = apply_qwindow(fdata, q, qprime, freq, deltam, sample_rate, duration)

            # wenergy = ifftshift(padded)
            tenergy = ifft(padded)
            energy = np.absolute(tenergy) ** 2
            np.multiply(energy, 1.0 / np.mean(energy), out=energy)

            maxe = energy.max()
            if maxe > peak_energy:
                peak_energy = maxe

            tile.append(energy)

        if peak_energy > glob_peak['energy']:
            glob_peak['energy'] = peak_energy
            glob_peak['tile'] = tile
            glob_peak['freqs'] = freqs
            glob_peak['q'] = q

        if output_all_tiles:
            glob_tiles.append({'tile': tile, 'freqs': freqs, 'duration': duration, 'q': q})

    return glob_peak, glob_tiles


def plot_cqt_tiles(tiles, time_range, freq_range, spec_cmap='ocean', cols=2):
    lines = int(math.ceil(len(tiles) / cols))
    fig, axs = plt.subplots(lines, cols)
    peak_im = None
    peak_tile = 0
    for i, data in enumerate(tiles):
        mode = 'subplot'
        if i // cols == lines - 1:
            mode = f'{mode}_bottom'
        if i % cols == 0:
            mode = f'{mode}_left'

        im = plot_cqt_spectrogram(data, time_range, freq_range,
                                  spec_cmap=spec_cmap, plt_scene=axs[i // cols, i % cols], mode=mode)

        peak_energy = im.norm.vmax
        if peak_tile < peak_energy:
            peak_tile = peak_energy
            peak_im = im

    if peak_im is not None:
        fig.colorbar(peak_im, ax=axs.ravel().tolist(), shrink=0.75)

    return fig


def interpolate_cqt_data(data, time_range, freq_range, t0=0):
    xout = np.arange(time_range[0], time_range[1], time_range[2])
    fout = np.arange(freq_range[0], freq_range[1], freq_range[2])

    tile = data['tile']
    freqs = data['freqs']
    duration = data['duration']
    q = data['q']

    tile_interp = []
    for row in tile:
        xrow = np.linspace(t0, duration, len(row))
        interp = InterpolatedUnivariateSpline(xrow, row)
        tile_interp.append(interp(xout).astype(float, casting="same_kind", copy=False))

    tile_interp = np.array(tile_interp)
    interp = interp2d(xout, freqs, tile_interp, kind='cubic')
    qspectr = interp(xout, fout).astype(float, casting="same_kind", copy=False)
    return xout, fout, qspectr, q


def plot_cqt_spectrogram(data, time_range, freq_range, spec_cmap='ocean', plt_scene=None, mode='main'):
    xout, fout, qspectr, q = interpolate_cqt_data(data, time_range, freq_range)
    return plot_spectrogram(qspectr, fout, xout, spec_cmap=spec_cmap, interpolation=None,
                            plt_scene=plt_scene, label=f'Q={str(round(q, 1))}', mode=mode)
