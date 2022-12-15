import torch
import torchaudio
import matplotlib.pyplot as plt
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG
from data import DSD100Dataset
from models import DilationGenerator, WaveformDiscriminators, SpectralDiscriminators
from utils import *
from device import device
from transforms import Transforms
from unet import Model
from hifi import BandwidthExtender
from losses import *



fs = CONFIG['data_fs']
low_fs = CONFIG['lp_fs']

# Training dataset
training_data = DSD100Dataset('DSD100/Train')
training_loader = DataLoader(training_data, shuffle=True, batch_size=3, num_workers=3)


# Model & training objects
generator = DilationGenerator().to(device)
optimizer = optim.Adam(generator.parameters(), lr=1e-3)

# Keep track of training progress
training_run_name = 'No Discriminators'
writer = SummaryWriter('tensorboard_data/' + training_run_name)

n_iter = 0
for epoch in range(7):
    for filepath, audio, lp_audio, spects in training_loader:
        # Move data to GPU
        #audio = audio.to(device)
        lp_audio = lp_audio.to(device)
        #spects = [spect.to(device) for spect in spects]

        # Upsample the low resolution audio before inference
        interp_audio = Transforms.Upsample(lp_audio, use_device=True)

        # Get high-resolution audio from generator
        reconst_audio = generator(interp_audio)
        reconst_audio = reconst_audio[:, 0, :].cpu()

        # Take spectrograms
        reconst_spects = Transforms.Spectrograms(reconst_audio)

        # Loss functions
        loss_waveform = waveform_loss(reconst_audio, audio)
        loss_spect = spectrogram_loss(reconst_spects, spects)
        loss = loss_waveform + loss_spect

        # Update model params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print(f"{n_iter}: {loss.item()}")
        writer.add_scalar('Loss/train/generator_only', loss.detach().cpu().item(), n_iter)

        # Combat running out of memory
        del audio
        del lp_audio
        for spect in spects:
            del spect

        n_iter += 1

torch.save(generator.state_dict(), CONFIG['saved_model_folder'] + 'generator_last.pt')



"""
# Load in previous generator model
generator = DilationGenerator().to(device)
generator_params = torch.load(CONFIG['saved_model_folder'] + 'generator_last.pt')
generator.load_state_dict(generator_params)
"""


# Discriminator models
discriminator_time = WaveformDiscriminators().to(device)
discriminator_spect = SpectralDiscriminators().to(device)

# New learning rates
optimizer = optim.Adam(generator.parameters(), lr=1e-5)
optimizer_disc = optim.Adam(discriminator_time.parameters(), lr=1e-3)
optimizer_disc_spect = optim.Adam(discriminator_spect.parameters(), lr=1e-3)

n_iter = 0
for epoch in range(15):
    for filepath, audio, lp_audio, spects in training_loader:
        # Move data to GPU
        audio = audio.to(device)
        lp_audio = lp_audio.to(device)
        spects = [spect.to(device) for spect in spects]

    
        # Upsample the low resolution audio before inference
        interp_audio = Transforms.Upsample(lp_audio, use_device=True)

        # Get high-resolution audio from the generator
        reconst_audio = generator(interp_audio)

        # Take spectrograms
        reconst_spects = Transforms.Spectrograms(reconst_audio[:, 0, :], use_device=True)

        # DISCRIMINATOR TRAINING
        for i in range(2):
            ## WAVEFORM DISCRIMINATORS ##

            # Discriminate on the generated & real input
            disc_time_fake = discriminator_time(reconst_audio.detach())
            disc_time_real = discriminator_time(audio.unsqueeze(1).detach())

            # Get loss
            loss_disc_time = discriminator_loss(disc_time_fake, disc_time_real)

            # Update discrminator params
            optimizer_disc.zero_grad()
            loss_disc_time.backward()
            optimizer_disc.step()

            writer.add_scalar('Loss/train/discriminator', \
                    loss_disc_time.detach().cpu().item(), n_iter + 1 - (1/(i+1)))


            ## SPECTROGRAM DISCRIMINATORS ##

            # Discriminate on real & generated input
            disc_spect_fake = discriminator_spect([spect.detach() for spect in reconst_spects])
            disc_spect_real = discriminator_spect([spect.detach() for spect in spects])

            # Get loss
            loss_disc_spect = discriminator_loss(disc_spect_fake, disc_spect_real)

            # Update discriminator params
            optimizer_disc_spect.zero_grad()
            loss_disc_spect.backward()
            optimizer_disc_spect.step()

            writer.add_scalar('Loss/train/discriminator_spect', \
                    loss_disc_spect.detach().cpu().item(), n_iter + 1 - (1/(i+1)))

            # Delete unused memory
            for result in disc_time_fake:
                del result
            for result in disc_time_real:
                del result
            for result in disc_spect_fake:
                del result
            for result in disc_spect_real:
                del result


        
        # Discriminate on the reconstruction again for use in generator loss
        disc_time = discriminator_time(reconst_audio)
        disc_spect = discriminator_spect(reconst_spects)

        # Remove convolutional channels from audio
        reconst_audio = reconst_audio[:, 0, :]

        # Loss terms
        loss_waveform = waveform_loss(reconst_audio, audio)
        loss_spect = spectrogram_loss(reconst_spects, spects)
        loss_advers = adversarial_loss(disc_time, disc_spect)
        loss = loss_waveform + loss_spect + loss_advers

        # Update generator params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"{n_iter}: {loss.item()}")
        writer.add_scalar('Loss/train/generator', loss.detach().cpu().item(), n_iter)

        # Delete unneeded memory
        del audio
        del lp_audio
        for spect in spects:
            del spect

        n_iter += 1

torch.save(generator.state_dict(), CONFIG['saved_model_folder'] + 'discriminator_last.pt')
