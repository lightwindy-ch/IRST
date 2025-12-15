import torch
import torch.nn as nn
import torch.cuda
torch.cuda.set_device(0)

class UNet2Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet2Layer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.middle = nn.Sequential(   
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(64, out_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3
