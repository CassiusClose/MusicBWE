import torch
import torchaudio.transforms as T

from config import CONFIG
from device import device


class Transforms:
    """
    Stores transform objects used in training & testing. Don't use the objects
    directly, call the functions associated with each one. They let you choose
    to use the CPU or GPU.

    The GPU versions of each object is not transferred to the GPU until the
    first time the caller uses it. This way, a transform that's never used is
    not taking up unnecessary space on the GPU.
    """

    # Downsamples training data to simulate low sample-rate recording
    downsampler = T.Resample(CONFIG['data_fs'],
                             CONFIG['lp_fs'],
                             lowpass_filter_width=1)
    downsampler_device = None


    # Downsamples and filters test data. We use the resampling function's
    # filter as the test filter
    test_downsampler = T.Resample(CONFIG['data_fs'],
                                  CONFIG['lp_fs'],
                                  resampling_method='kaiser_window',
                                  lowpass_filter_width=50)
    test_downsampler_device = None


    # Upsamples the input before passing it to the model
    upsampler = T.Resample(CONFIG['lp_fs'],
                           CONFIG['data_fs'],
                           resampling_method='kaiser_window',
                           lowpass_filter_width=30)
    upsampler_device = None


    # Takes the spectrogram of a signal with window length of 2048, and a hop size of
    # 1 quarter of that.
    spectrograms = [T.Spectrogram(n_fft=n_win, hop_length=n_win//CONFIG['spectrogram_hop_divider'], power=1) for n_win in CONFIG['spectrogram_win_lens']]
    spectrograms_device = [None for i in range(len(spectrograms))]


    @staticmethod
    def Downsample(x, use_device=False):
        if use_device:
            if Transforms.downsampler_device is None:
                Transforms.downsampler_device = Transforms.downsampler.to(device)

            return Transforms.downsampler_device(x)

        else:
            return Transforms.downsampler(x)


    @staticmethod
    def TestDownsample(x, use_device=False):
        if use_device:
            if Transforms.test_downsampler_device is None:
                Transforms.test_downsampler_device = Transforms.test_downsampler.to(device)

            return Transforms.test_downsampler_device(x)

        else:
            return Transforms.test_downsampler(x)


    @staticmethod
    def Upsample(x, use_device=False):
        if use_device:
            if Transforms.upsampler_device is None:
                Transforms.upsampler_device = Transforms.upsampler.to(device)

            return Transforms.upsampler_device(x)

        else:
            return Transforms.upsampler(x)


    @staticmethod
    def Spectrogram(x, index, use_device=False):
        if use_device:
            if Transforms.spectrograms_device[index] is None:
                n_win = CONFIG['spectrogram_win_lens'][index]
                n_hop = n_win//CONFIG['spectrogram_hop_divider']
                Transforms.spectrograms_device[index] = T.Spectrogram(n_fft=n_win, hop_length=n_hop, power=1).to(device)

            return Transforms.spectrograms_device[index](x)

        else:
            return Transforms.spectrograms[index](x)
            


    @staticmethod
    def Spectrograms(x, use_device=False):
        return [Transforms.Spectrogram(x, ind, use_device) for ind in range(len(Transforms.spectrograms))]
