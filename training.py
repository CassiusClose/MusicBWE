import torch
import torchaudio
import matplotlib.pyplot as plt
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG
from data import DSD100Dataset
from models import UNetGenerator, DilationGenerator
from utils import *
from device import device
from transforms import Transforms



fs = CONFIG['data_fs']
low_fs = CONFIG['lp_fs']

# Training dataset
training_data = DSD100Dataset('DSD100/Train')
training_loader = DataLoader(training_data, shuffle=True, batch_size=2, num_workers=3)


# Model & training objects
generator = DilationGenerator(fs).to(device)
optimizer = optim.Adam(generator.parameters(), lr=1e-3)
loss_l1 = nn.L1Loss()


# Keep track of training progress
training_loss = []
loss_smoothing = 0.9
training_run_name = 'Current Run'
writer = SummaryWriter('tensorboard_data/' + training_run_name)

n_iter = 0

for epoch in range(15):
    for filepath, audio, lp_audio, spect in training_loader:
        audio = audio.to(device)
        lp_audio = lp_audio.to(device)
        spect = spect.to(device)

        interp_audio = Transforms.Upsample(lp_audio, use_device=True)
        reconst_audio = generator(interp_audio)

        reconst_spect = Transforms.Spectrogram2048(reconst_audio, use_device=True)

        loss_waveform = loss_l1(reconst_audio, audio)
        loss_spect = loss_l1(reconst_spect, spect)

        loss = loss_waveform + loss_spect

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print(f"{n_iter}: {loss.item()}")
        """
        if len(training_loss) == 0:
            training_loss.append(loss.detach().cpu())
        else:
            training_loss.append(loss_smoothing * training_loss[-1] + (1-loss_smoothing) * loss.detach().cpu())
        """
        writer.add_scalar('Loss/train', loss.detach().cpu().item(), n_iter)


        """
        if n_iter % 100 == 0:
            plot_spect(spect[0, :, :].detach().cpu().log10(), fs, 2048, 2048//4, "Original Spectogram", False, 0)
            plot_spect(reconst_spect[0, :, :].detach().cpu().log10(), fs, 2048, 2048//4, "Reconstructed Spectogram", False, 1)
            plt.show()
        """
        

        n_iter += 1


torch.save(generator.state_dict(), CONFIG['saved_model_folder'] + 'dilation_last.pt')
