import torch
import torchaudio
from torch.utils.data import Dataset
import pathlib

import yaml
try:
    file = open('config.yml', 'r')
    config = yaml.safe_load(file)
except:
    print("Error opening config file: config.yml")
    exit()



class DSD100Dataset(Dataset):
    def __init__(self, path):
        path = pathlib.Path(path)   
        files = path.rglob("*.wav")
        self.filepaths = [s.as_posix() for s in files]

    def __getitem__(self, index):

        (audio, fs) = torchaudio.load(self.filepaths[index])
        audio = audio.mean(0)

        num_samps = config['training_length'] * fs

        startIndex = torch.randint(0, audio.shape[-1] - num_samps, (1,))
        audio = audio[startIndex : startIndex + num_samps]


           





    def __len__(self):
        return len(self.filepaths)

        



d = DSD100Dataset('DSD100/Mixtures/Train')
d[0]
