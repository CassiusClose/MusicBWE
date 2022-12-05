import torch
import scipy.signal as signal

from config import CONFIG



def filter_train_rand(waveform, fs):
    filt = torch.randint(0, 3, (1,)).item()
    match filt:
        case 0:
            return filter_butter_rand(waveform, fs)
        case 1:
            return filter_cheby_rand(waveform, fs)
        case 2:
            return filter_ellip_rand(waveform, fs)

    return None



def filter_butter_rand(waveform, fs):
    (order, cutoff, rp, rs) = choose_rand_gen_filt_params('butterworth')
    return filter_butter(waveform, fs, cutoff, order)

def filter_bessel_rand(waveform, fs):
    (order, cutoff, rp, rs) = choose_rand_gen_filt_params('bessel')
    return filter_bessel(waveform, fs, cutoff, order)

def filter_cheby_rand(waveform, fs):
    (order, cutoff, rp, rs) = choose_rand_gen_filt_params('chebyshev')
    return filter_cheby(waveform, fs, cutoff, order, rp)

def filter_ellip_rand(waveform, fs):
    (order, cutoff, rp, rs) = choose_rand_gen_filt_params('elliptic')
    return filter_ellip(waveform, fs, cutoff, order, rp, rs)

    


def filter_butter(waveform, fs, cutoff, order):
    sos = signal.butter(order, cutoff, output='sos', fs=fs)
    return torch.Tensor(signal.sosfilt(sos, waveform))


def filter_bessel(waveform, fs, cutoff, order):
    sos = signal.bessel(order, cutoff, 'low', output='sos', fs=fs)
    return torch.Tensor(signal.sosfilt(sos, waveform))


def filter_cheby(waveform, fs, cutoff, order, rp):
    sos = signal.cheby1(order, rp, cutoff, 'low', output='sos', fs=fs)
    return torch.Tensor(signal.sosfilt(sos, waveform))


def filter_ellip(waveform, fs, cutoff, order, rp, rs):
    sos = signal.ellip(order, rp, rs, [cutoff], 'low', output='sos', fs=fs)
    return torch.Tensor(signal.sosfilt(sos, waveform))




def choose_rand_gen_filt_params(filter_name):
    order_low = CONFIG['filters'][filter_name]['order_low']
    order_high = CONFIG['filters'][filter_name]['order_high'] + 1
    order = torch.randint(order_low, order_high, (1,)).item()

    cutoff_low = CONFIG['filters']['cutoff_low']
    cutoff_high = CONFIG['filters']['cutoff_high']
    cutoff = torch.randint(cutoff_low, cutoff_high, (1,)).item()

    rp = CONFIG['filters'][filter_name].get('rp')

    rs_low = CONFIG['filters'][filter_name].get('rs_low')
    rs_high = CONFIG['filters'][filter_name].get('rs_high')
    rs = None
    if rs_low != None and rs_high != None:
        rs = torch.randint(rs_low, rs_high, (1,)).item()

    return (order, cutoff, rp, rs)


