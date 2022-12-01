import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import scipy.signal as signal
import math
import numpy as np

from utils import *
from sinesweep import get_sine_sweep


BUTTER_ORDER = 50
BESSEL_ORDER = 10
CHEBY_ORDER = 12
CHEBY_RP = 0.1
ELLIP_ORDER = 12
ELLIP_RP = 0.01
ELLIP_RS = 100

TEST_FOLDER = "FilterTests/"
    

def test_filters(): 

    fs = 48000
    audio = torch.Tensor(get_sine_sweep(fs)).float()

    #audio, fs = torchaudio.load("DSD100/Mixtures/Train/067 - Georgia Wonder - Siren/mixture.wav")
    #audio = audio[:, fs*3:fs*10]

    cutoff = 5000

    plot_spect(audio, fs, 'Original Waveform', False, 0)
    torchaudio.save(TEST_FOLDER + "original.wav", audio, fs)


    (butter_filtered, sos) = filter_butter(audio, cutoff, fs)
    #plot_freqz(sos, fs, cutoff, 1)
    plot_spect(butter_filtered, fs, 'Butterworth Filter', False, 1)
    torchaudio.save(TEST_FOLDER + "butterworth.wav", butter_filtered, fs)


    (bessel_filtered, sos) = filter_bessel(audio, cutoff, fs)
    #plot_freqz(sos, fs, cutoff, 2)
    plot_spect(bessel_filtered, fs, 'Bessel Filter', False, 2)
    torchaudio.save(TEST_FOLDER + "bessel.wav", bessel_filtered, fs)


    (cheby_filtered, sos) = filter_cheby(audio, cutoff, fs)
    #plot_freqz(sos, fs, cutoff, 3)
    plot_spect(cheby_filtered, fs, 'Chebyshev Filter', False, 3)
    torchaudio.save(TEST_FOLDER + "chebyshev.wav", cheby_filtered, fs)


    (ellip_filtered, sos) = filter_ellip(audio, cutoff, fs)
    #plot_freqz(sos, fs, cutoff, 4)
    plot_spect(ellip_filtered, fs, 'Elliptic Filter', False, 4)
    torchaudio.save(TEST_FOLDER + "elliptic.wav", ellip_filtered, fs)

    plt.show()




def filter_butter(waveform, cutoff, fs):
    sos = signal.butter(BUTTER_ORDER, cutoff, output='sos', fs=fs)
    return (torch.Tensor(signal.sosfilt(sos, waveform)), sos)


def filter_bessel(waveform, cutoff, fs):
    sos = signal.bessel(BESSEL_ORDER, cutoff, 'low', output='sos', fs=fs)
    return (torch.Tensor(signal.sosfilt(sos, waveform)), sos)


def filter_cheby(waveform, cutoff, fs):
    sos = signal.cheby1(CHEBY_ORDER, CHEBY_RP, cutoff, 'low', output='sos', fs=fs)
    return (torch.Tensor(signal.sosfilt(sos, waveform)), sos)


def filter_ellip(waveform, cutoff, fs):
    sos = signal.ellip(ELLIP_ORDER, ELLIP_RP, ELLIP_RS, [cutoff], 'low', output='sos', fs=fs)
    return (torch.Tensor(signal.sosfilt(sos, waveform)), sos)




if __name__ == '__main__':
    test_filters()

