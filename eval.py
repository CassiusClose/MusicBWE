import torch
import torchaudio
import torchmetrics
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
generator = DilationGenerator().to(device)
#generator_params = torch.load(CONFIG['saved_model_folder'] + 'nodiscriminator_generator_last.pt')
generator_params = torch.load(CONFIG['saved_model_folder'] + 'fullmodel_discriminator_last.pt')
generator.load_state_dict(generator_params)
generator.eval()


# Dataset & loader
test_data = DSD100Dataset('DSD100/Test', return_audio_segment=False, test_filter=True)
test_loader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=1)


# Objective metrics
sdr = torchmetrics.SignalDistortionRatio()
snr = torchmetrics.SignalNoiseRatio()

sdrs = []
snrs = []
lsds = []

fad_len = 40

# Just process one audio file for now
for filepath, audio, lp_audio, spects in test_loader:
    from pathlib import Path
    p = Path(filepath[0])
    print(p.parts[-2])


    # Save a copy of the downsampled version for comparison
    torchaudio.save(CONFIG['results_folder'] + 'originals/' + p.parts[-2] + '_orig.wav', audio[:,:fad_len*fs], fs)

    # Upsample before passing to the model
    lp_audio = lp_audio.to(device)
    interp_audio = Transforms.Upsample(lp_audio, use_device=True)

    # Split into several segments for memory reasons
    lp_audio_segmented = interp_audio.split(low_fs * CONFIG['eval_segment_seconds'], 1)

    # Run each segment through the model, collect outputs
    reconst_audio = []
    for i, segment in enumerate(lp_audio_segmented):
        result = generator(segment)[0]
        reconst_audio.append(result.detach().cpu())
        del result

    # Compile outputs into one result
    reconst_audio = torch.cat(reconst_audio, 1)

    if reconst_audio.shape[-1] < audio.shape[-1]:
        audio = audio[:, :reconst_audio.shape[-1]]
    elif audio.shape[-1] < reconst_audio.shape[-1]:
        reconst_audio = reconst_audio[:, :audio.shape[-1]]

    sdrs.append(sdr(reconst_audio, audio))
    print(f"SDR: {sdrs[-1]}")
    snrs.append(snr(reconst_audio, audio))
    print(f"SNR: {snrs[-1]}")


    audio_spect = Transforms.Spectrogram(audio, 0)
    reconst_spect = Transforms.Spectrogram(reconst_audio, 0)

    lsd = (((audio_spect[0]**2).clamp(1e-5).log() - (reconst_spect[0]**2).clamp(1e-5).log())**2).mean(0).sqrt().mean(0)
    lsds.append(lsd)
    print(f"LSD: {lsd}\n")

    """
    plot_spect(audio_spect[0, :, fs*20//2048:fs*30//2048].log10(), fs, 2048, 2048//4, "Ground Truth Wideband Spectrogram", False, 1)
    plot_spect(reconst_spect[0, :, fs*20//2048:fs*30//2048].log10(), fs, 2048, 2048//4, "Reconstructed Wideband Spectrogram", True, 2)
    """


    # Save result
    torchaudio.save(CONFIG['results_folder'] + 'reconstructions/' + p.parts[-2] + "_reconst.wav", reconst_audio[:,:fad_len*fs], fs)

    torch.cuda.empty_cache()

# Average stats over all test data
mean_sdr = torch.Tensor(sdrs).mean()
print(f"Average SDR: {mean_sdr}")

mean_snr = torch.Tensor(snrs).mean()
print(f"Average SNR: {mean_snr}")

mean_lsd = torch.Tensor(lsds).mean()
print(f"Average LSD: {mean_lsd}")
