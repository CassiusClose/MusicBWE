import torchaudio.transforms as T

from config import CONFIG
from device import device


class Transforms:
    downsampler = T.Resample(CONFIG['data_fs'],
                             CONFIG['lp_fs'],
                             lowpass_filter_width=1)
    downsampler_device = None


    upsampler = T.Resample(CONFIG['lp_fs'],
                           CONFIG['data_fs'],
                           resampling_method='kaiser_window',
                           lowpass_filter_width=30)
    upsampler_device = None


    spectrogram2048 = T.Spectrogram(n_fft=2048,
                                    hop_length=2048//4)
    spectrogram2048_device = None


    @staticmethod
    def Downsample(x, use_device=False):
        if use_device:
            if Transforms.downsampler_device is None:
                Transforms.downsampler_device = Transforms.downsampler.to(device)

            return Transforms.downsampler_device(x)

        else:
            return Transforms.downsampler(x)


    @staticmethod
    def Upsample(x, use_device=False):
        if use_device:
            if Transforms.upsampler_device is None:
                Transforms.upsampler_device = Transforms.upsampler.to(device)

            return Transforms.upsampler_device(x)

        else:
            return Transforms.upsampler(x)


    @staticmethod
    def Spectrogram2048(x, use_device=False):
        if use_device:
            if Transforms.spectrogram2048_device is None:
                #Transforms.spectrogram2048_device = Transforms.spectrogram2048.to(device)
                Transforms.spectrogram2048_device = T.Spectrogram(n_fft=2048,
                                    hop_length=2048//4).to(device)

            return Transforms.spectrogram2048_device(x)

        else:
            return Transforms.spectrogram2048(x)
