import torch
import torchaudio
from torch.utils.data import DataLoader

from models import UNetGenerator, DilationGenerator
from utils import *
from data import DSD100Dataset
from config import CONFIG
from transforms import Transforms
from device import device


# TODO Make this work on any command-line argument file, not just in the dataset


fs = CONFIG['data_fs']
low_fs = CONFIG['lp_fs']

# Load saved model state
generator = DilationGenerator(fs).to(device)
generator_params = torch.load(CONFIG['saved_model_folder'] + 'dilation_last.pt')
generator.load_state_dict(generator_params)
generator.eval()



# Dataset & loader
test_data = DSD100Dataset('DSD100/Test', return_audio_segment=False)
test_loader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=1)


# Just process one audio file for now
for filepath, audio, lp_audio, spect in test_loader:
    # Save a copy of the downsampled version for comparison
    torchaudio.save("lp_output.wav", lp_audio, low_fs)

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
    reconst_audio = torch.cat(reconst_audio, 0)


    # Save result
    print(filepath)
    torchaudio.save("output.wav", reconst_audio.unsqueeze(0), fs)


    # Compare spectrograms
    reconst_spect = Transforms.Spectrogram2048(reconst_audio)
    plot_spect(spect[0].cpu().log10(), fs, 2048, 2048//4, 'Original', False, 0)
    plot_spect(reconst_spect.detach().cpu().log10(), fs, 2048, 2048//4, 'Reconstructed', False, 1)
    plt.show()
    

    # For now, just evaluate
    break


