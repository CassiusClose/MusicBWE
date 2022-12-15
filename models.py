import torch
from torch import nn
import typing as T

from config import CONFIG


class DilationGenerator(nn.Module):
    """
    WaveNet-based generator, using dilated convolution layers.
    """

    def __init__(self):
        super().__init__()

        num_channels = 128
        num_layers = 5

        # Transform input to the desired number of channels
        self.inp_conv = nn.Conv1d(1, num_channels, 1)

        # Layers of dilated convolution, powers of 3 dilation 
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleList()
            layer.append(nn.Conv1d(num_channels, num_channels, 3, dilation=3**i, padding="same"))

            self.layers.append(layer)

        # Transform output back to 1 channel (mono audio)
        self.output_conv = nn.ModuleList()
        self.output_conv.append(nn.Conv1d(num_channels, num_channels, 1))
        self.output_conv.append(nn.Conv1d(num_channels, 1, 1))



    def forward(self, x):
        # Audio coming in doesn't have channel information, so provide it
        x = x.unsqueeze(1)

        x = self.inp_conv(x)

        skips = 0
        for layer in self.layers:
            inp = x

            # Perform convolution
            x = layer[0](x)
            x = nn.functional.leaky_relu(x)

            # Sum each layer's output 
            skips += x

            # Add to the original input to the layer, pass it on to the next
            # layer
            x = inp + x
            
        # Each layer's output gets summed here
        x = skips

        # Then pass everything through the output convolution layers
        for conv in self.output_conv:
            x = nn.functional.leaky_relu(x)
            x = conv(x)

        return x





class WaveformDiscriminator(nn.Module):
    """
    Discriminates a waveform using strided convolution layers
    """

    def __init__(self):
        super().__init__()

        stride = 2
        kernel = stride * 10 + 1

        chan = 16

        # Input convolution, transform to the desired channel numbers
        self.conv_inp = nn.Conv1d(1, chan, 15, padding="same")

        # Strided convolution to downsample the input
        num_layers = 5
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Max out channels at 1024
            new_chan = chan*stride
            if new_chan > 1024:
                new_chan = 1024

            layer = nn.Conv1d(chan, chan*stride, kernel, stride, groups=chan//4, padding=20)
            self.layers.append(layer)

            chan = new_chan

        # Output layers, convert down to 1 channel
        self.layer_out1 = nn.Conv1d(chan, chan, 5, 1, padding=2)
        self.layer_out2 = nn.Conv1d(chan, 1, 3, 1, padding=1)


    def forward(self, x):
        x = self.conv_inp(x)

        for layer in self.layers:
            x = layer(x)
            x = nn.functional.leaky_relu(x)

        x = self.layer_out1(x)
        x = nn.functional.leaky_relu(x)
        
        x = self.layer_out2(x)

        return x



class WaveformDiscriminators(nn.Module):
    """ 
    A collection of waveform discriminators, using increasingly downsampled
    versions of the waveform.
    """

    def __init__(self):
        super().__init__()

        self.discrims = nn.ModuleList()

        # Downsample with average pooling
        self.avgpool = nn.AvgPool1d(4, 2, padding=2, count_include_pad=False)

        for i in range(3):
            self.discrims.append(WaveformDiscriminator())


    def forward(self, x):
        results = []

        for discrim in self.discrims:
            # Save results from current discriminator
            x = discrim(x)
            results.append(x)

            # Then downsample before passing to the next discriminator
            x = self.avgpool(x)

        return results



class SpectralDiscriminator(nn.Module):
    """
    Discriminates a spectrogram using strided convolution layers
    """

    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList()

        # Several layers of strided convolution
        chans = [1, 32, 128, 512, 1024, 1024]
        for i in range(1, len(chans)):
            self.layers.append(nn.Conv2d(chans[i-1], chans[i], (5, 1), (3, 1), padding=(2,0)))

        # Reduce down to one channel
        self.out = nn.Conv2d(chans[-1], 1, (3, 1), 1, padding=(2,0))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.functional.leaky_relu(x)

        x = self.out(x)
        return x


class SpectralDiscriminators(nn.Module):
    """
    A collection of spectrogram discriminators 
    """

    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList()
        # Config determines how many spectrograms are taken
        for i in range(len(CONFIG['spectrogram_win_lens'])):
            self.layers.append(SpectralDiscriminator())


    def forward(self, x):
        # Input is a list of spectrogram data (shape: batch, frame, bins)
        if len(x) is not len(CONFIG['spectrogram_win_lens']):
            print("ERROR: Input to SpectralDiscriminators should be a list of batched spectrograms equal to the number of spectrogram discriminators.")
            return None

        results = []
        for i in range(len(self.layers)):
            y = self.layers[i](x[i].unsqueeze(1))
            results.append(y)

        return results






class UNetGenerator(nn.Module):
    """
    An initial test of the U-Net architecture. All I could get this to do was flip
    the low frequency content repeatedly into the higher frequencies.
    """

    def __init__(self, fs):
        super().__init__()

        self.fs = fs

        # Encoder 
        enc_channels = [1, 128, 256, 512, 512, 512]
        enc_kernels = [65, 33, 17, 9, 9]
        stride = 4
        self.enc_layers = nn.ModuleList()
        for i in range(len(enc_kernels)):
            layer = nn.Sequential(
                nn.Conv1d(enc_channels[i], enc_channels[i+1], enc_kernels[i], stride=stride),
                nn.BatchNorm1d(enc_channels[i+1]),
                nn.ReLU()
            )
            self.enc_layers.append(layer)

        
        # Transform the high-level features
        self.bottleneck_layers = nn.ModuleList()
        self.bottleneck_layers.append(nn.Conv1d(512, 512, 5, padding=(5-1)//2))
        self.bottleneck_layers.append(nn.BatchNorm1d(512))
        self.bottleneck_layers.append(nn.ReLU())

        
        # Decoder
        dec_channels = [512, 512, 512, 256, 128, 1]
        dec_kernels = [9, 9, 17, 33, 65]
        dec_padding = [0, 3, 0, 1, 3]
        self.dec_layers = nn.ModuleList()
        for i in range(len(dec_kernels)):
            layer = nn.Sequential(
                nn.Conv1d(2*dec_channels[i], dec_channels[i], dec_kernels[i], padding="same"),
                nn.ConvTranspose1d(dec_channels[i],
                                   dec_channels[i+1],
                                   dec_kernels[i],
                                   output_padding=dec_padding[i],
                                   stride=stride),
                nn.BatchNorm1d(dec_channels[i+1]),
                nn.ReLU()
            )
            self.dec_layers.append(layer)


    def forward(self, x):
        y = x.unsqueeze(1)

        skips = []

        for conv in self.enc_layers:
            y = conv(y)
            skips.append(y)
            
        for conv in self.bottleneck_layers:
            y = conv(y)
            
        for i, conv in enumerate(self.dec_layers):
            y = torch.cat((y, skips[len(skips)-1-i]), 1)
            y = conv(y)


        return y

