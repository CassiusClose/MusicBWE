import torch
from torch import nn

"""
Calculates different losses for training the GAN
"""

loss_l1 = nn.L1Loss()


def discriminator_loss(disc_fake, disc_real):
    """
    Loss for training the discriminators. Takes discriminator output as input.
    """

    fake_loss = 0
    real_loss = 0

    for i in range(len(disc_fake)):
        fake_loss += torch.relu(1 + disc_fake[i]).mean()
        real_loss += torch.relu(1 - disc_real[i]).mean()

    return fake_loss + real_loss



def adversarial_loss(disc_time, disc_spect):
    """
    Loss for training the generator from the discriminators. Takes discriminator
    outputs on the generated sample as input.
    """

    loss_time = 0
    for result in disc_time:
        loss_time += -result.mean()

    loss_spect = 0
    for result in disc_spect:
        loss_spect += -result.mean()

    return loss_time + loss_spect


def spectrogram_loss(reconst_spects, spects):
    """
    For training the generator normally. L1 spectrogram loss.
    """
    loss = 0
    for i in range(len(reconst_spects)):
        loss += loss_l1(reconst_spects[i], spects[i])

    return loss


def waveform_loss(reconst_audio, audio):
    """
    For training the generator normally. L1 waveform loss.
    """
    return loss_l1(reconst_audio, audio)
