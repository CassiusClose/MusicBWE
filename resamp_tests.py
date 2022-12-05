import torch
import torchaudio
import torchaudio.transforms as T

from filters import *
from sinesweep import get_sine_sweep
from utils import *


BUTTER_ORDER = 50
BESSEL_ORDER = 10
CHEBY_ORDER = 20
CHEBY_RP = 0.1
ELLIP_ORDER = 12
ELLIP_RP = 0.1
ELLIP_RS = 100

TEST_FOLDER = "ResampTests/"


#fs = 48000
#audio = torch.Tensor(get_sine_sweep(fs)).float()

audio, fs = torchaudio.load("DSD100/Train/067 - Georgia Wonder - Siren/mixture.wav")
audio = audio[:, fs*3:fs*10]

plot_spect(audio, fs, 'Original Waveform', False, 0)
torchaudio.save(TEST_FOLDER + "original.wav", audio, fs)

new_fs = 8000

filt_downsampler = T.Resample(fs, new_fs, resampling_method='kaiser_window', lowpass_filter_width=30)
downsampler = T.Resample(fs, new_fs, lowpass_filter_width=1)


audio_ds = downsampler(audio)
plot_spect(audio_ds, new_fs, 'Downsampled', False, 1)
torchaudio.save(TEST_FOLDER + "ds_audio.wav", audio_ds, new_fs)

kaiser_ds = filt_downsampler(audio)
plot_spect(kaiser_ds, new_fs, 'Kaiser Downsampled', False, 2)
torchaudio.save(TEST_FOLDER + "ds_kaiser.wav", kaiser_ds, new_fs)

butter_ds = downsampler(filter_butter(audio, fs, new_fs//2-1, BUTTER_ORDER))
plot_spect(butter_ds, new_fs, 'Butterworth Downsampled', False, 3)
torchaudio.save(TEST_FOLDER + "ds_butter.wav", butter_ds, new_fs)

bessel_ds = downsampler(filter_bessel(audio, fs, new_fs//2-1, BESSEL_ORDER))
plot_spect(bessel_ds, new_fs, 'Bessel Downsampled', False, 4)
torchaudio.save(TEST_FOLDER + "ds_bessel.wav", bessel_ds, new_fs)

cheby_ds = downsampler(filter_cheby(audio, fs, new_fs//2-1, CHEBY_ORDER, CHEBY_RP))
plot_spect(cheby_ds, new_fs, 'Chebyshev Downsampled', False, 5)
torchaudio.save(TEST_FOLDER + "ds_cheby.wav", cheby_ds, new_fs)

ellip_ds = downsampler(filter_ellip(audio, fs, new_fs//2-1, ELLIP_ORDER, ELLIP_RP, ELLIP_RS))
plot_spect(ellip_ds, new_fs, 'Elliptic Downsampled', False, 6)
torchaudio.save(TEST_FOLDER + "ds_ellip.wav", ellip_ds, new_fs)

plt.show()



upsampler = T.Resample(new_fs, fs, resampling_method='kaiser_window', lowpass_filter_width=30)

audio_us = upsampler(audio_ds)
plot_spect(audio_us, fs, 'Upsampled', False, 1)
torchaudio.save(TEST_FOLDER + "us_audio.wav", audio_us, fs)

kaiser_us = upsampler(kaiser_ds)
plot_spect(kaiser_us, fs, 'Kaiser Upsampled', False, 2)
torchaudio.save(TEST_FOLDER + "us_kaiser.wav", kaiser_us, fs)

butter_us = upsampler(butter_ds)
plot_spect(butter_us, fs, 'Butterworth Upsampled', False, 3)
torchaudio.save(TEST_FOLDER + "us_butter.wav", butter_us, fs)

bessel_us = upsampler(bessel_ds)
plot_spect(bessel_us, fs, 'Bessel Upsampled', False, 4)
torchaudio.save(TEST_FOLDER + "us_bessel.wav", bessel_us, fs)

cheby_us = upsampler(cheby_ds)
plot_spect(cheby_us, fs, 'Chebyshev Upsampled', False, 5)
torchaudio.save(TEST_FOLDER + "us_cheby.wav", cheby_us, fs)

ellip_us = upsampler(ellip_ds)
plot_spect(ellip_us, fs, 'Elliptic Upsampled', False, 6)
torchaudio.save(TEST_FOLDER + "us_ellip.wav", ellip_us, fs)

plt.show()
