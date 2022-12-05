import torch
from torch import nn
import typing as T

from config import CONFIG


class DilationGenerator(nn.Module):
    def __init__(self, fs):
        super().__init__()
        
        self.fs = fs

        channels = 128

        self.layers = nn.ModuleList()
        for i in range(4):
            layer = nn.ModuleList()
            layer.append(nn.Conv1d(channels, channels, 3, dilation=3**i, padding="same"))

            self.layers.append(layer)


        self.inp_conv = nn.Conv1d(1, channels, 1)


        self.output_conv = nn.ModuleList()
        self.output_conv.append(nn.Conv1d(channels, channels, 1))
        self.output_conv.append(nn.Conv1d(channels, 1, 1))


    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.inp_conv(x)

        skips = 0
        for layer in self.layers:
            inp = x

            x = layer[0](x)

            x = torch.relu(x)

            skips += x

            x = inp + x
            
        x = skips

        for conv in self.output_conv:
            x = torch.relu(x)
            x = conv(x)

        x = torch.tanh(x)

        x = x[:, 0, :]

        return x




class UNetGenerator(nn.Module):

    def __init__(self, fs):
        super().__init__()

        self.fs = fs

        
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

    
        self.bottleneck_layers = nn.ModuleList()
        self.bottleneck_layers.append(nn.Conv1d(512, 512, 5, padding=(5-1)//2))
        self.bottleneck_layers.append(nn.BatchNorm1d(512))
        self.bottleneck_layers.append(nn.ReLU())

        #(inp + kernel_size - 2)/4 + 1

        dec_channels = [512, 512, 512, 256, 128, 1]
        dec_kernels = [9, 9, 17, 33, 65]
        dec_padding = [0, 3, 0, 1, 3]
        self.dec_layers = nn.ModuleList()
        for i in range(len(dec_kernels)):
            layer = nn.Sequential(
                nn.ConvTranspose1d(2*dec_channels[i],
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


        y = y[:, 0, :]

        return y

