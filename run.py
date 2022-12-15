import torch
import torchaudio
from torch.utils.data import DataLoader

from models import UNetGenerator, DilationGenerator
from utils import *
from data import DSD100Dataset
from config import CONFIG
from transforms import Transforms
from device import device
from unet import Model
from hifi import BandwidthExtender


# TODO Make this work on any command-line argument file, not just in the dataset


fs = CONFIG['data_fs']
low_fs = CONFIG['lp_fs']

# Load saved model state
generator = DilationGenerator().to(device)
generator_params = torch.load(CONFIG['saved_model_folder'] + 'nodiscriminator_generator_last.pt')
#generator_params = torch.load(CONFIG['saved_model_folder'] + 'nospect_discriminator_last.pt')
generator.load_state_dict(generator_params)
generator.eval()



# Dataset & loader
test_data = DSD100Dataset('DSD100/Test', return_audio_segment=False, test_filter=True)
test_loader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=1)


n_iter = 0
# Pick one file to process
choose_iter=5
for filepath, audio, lp_audio, spect in test_loader:
    if n_iter < choose_iter:
        n_iter += 1
        continue
    if n_iter == choose_iter+1:
        break

    # Save a copy of the downsampled version for comparison
    torchaudio.save(CONFIG['samples_folder'] + "lp_output.wav", lp_audio, low_fs)

    # Upsample before passing to the model
    lp_audio = lp_audio.to(device)
    interp_audio = Transforms.Upsample(lp_audio, use_device=True)

    # Split into several segments for memory reasons
    lp_audio_segmented = interp_audio.split(low_fs * CONFIG['eval_segment_seconds'], 1)

    # Run each segment through the model, collect outputs
    reconst_audio = []
    for i, segment in enumerate(lp_audio_segmented):
        print(f"Segment {i}")
        reconst_audio.append(generator(segment)[0].detach().cpu())

    # Compile outputs into one result
    reconst_audio = torch.cat(reconst_audio, 1)[0]


    # Save result
    print(filepath)
    torchaudio.save(CONFIG['samples_folder'] + "output.wav", reconst_audio.unsqueeze(0), fs)


    # Compare spectrograms
    reconst_spect = Transforms.Spectrogram(reconst_audio, 0)
    n_win = CONFIG['spectrogram_win_lens'][0]
    n_hop = n_win//CONFIG['spectrogram_hop_divider']
    #plot_spect(spect[0][0].cpu(), fs, n_win, n_hop, 'Original', False, 0)
    #plot_calc_spect(interp_audio[0].cpu(), fs, 'Low Passed', False, 1)
    #plot_spect(reconst_spect.detach().cpu(), fs, 2048, 2048//4, 'High-Frequency Creation (Complete Method)', True, 2)
    

    n_iter += 1


