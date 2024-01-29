import torch
from torch import nn
from torch.nn import functional as F

class EncoderBlock(nn.Module):
    def __init__(self, input_channels=1, latent_channels=16, ksize=3):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=ksize, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=ksize, padding=1)
        self.conv_mu = nn.Conv2d(32, latent_channels, kernel_size=ksize, stride=1, padding=0)
        self.conv_logvar = nn.Conv2d(32, latent_channels, kernel_size=ksize, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        return mu, logvar


class DecoderBlock(nn.Module):
    def __init__(self, latent_channels=16, output_channels=1, ksize=3):
        super(DecoderBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_channels, 32, kernel_size=ksize, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=ksize, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, output_channels, kernel_size=ksize, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # Using sigmoid for image reconstruction
        return x

