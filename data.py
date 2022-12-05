import pathlib
import torch
import torchaudio
from torch.utils.data import Dataset

from config import CONFIG
from filters import *
from transforms import Transforms



class DSD100Dataset(Dataset):
    def __init__(self, path, return_audio_segment=True):
        path = pathlib.Path(path)   
        files = path.rglob("*.wav")
        self.filepaths = [s.as_posix() for s in files]

        self.return_audio_segment = return_audio_segment


    def __getitem__(self, index):
        # Load audio
        # If stereo, sum to mono. Perhaps in the future expand to stereo.
        (audio, fs) = torchaudio.load(self.filepaths[index])
        audio = audio.mean(0)

        
        # Take a short section of the audio to train with
        if self.return_audio_segment:
            num_samps = CONFIG['training_length'] * fs
            startIndex = torch.randint(0, audio.shape[-1] - num_samps, (1,))
            audio = audio[startIndex : startIndex + num_samps]


        # Calculate the spectrogram for accuracy/losses
        # Could save these to the disk for quicker training.
        spect2048 = Transforms.Spectrogram2048(audio)


        # Perform random filtering, then downsample
        lp_audio = filter_train_rand(audio, fs)
        resamp_audio = Transforms.Downsample(lp_audio)


        return (self.filepaths[index], audio, resamp_audio, spect2048)



    def __len__(self):
        return len(self.filepaths)
