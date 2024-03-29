import pdb

import torch
import torchsummary
from torch import nn
from torch.nn import functional as F

class EncoderBlock(nn.Module):
    def __init__(self, input_channels, encoder_out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32) #[14, 2, 3, 0.5]     [n_out, j_out, r_out,
        # start_out ]

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32) #[7, 4, 7, 0.5]

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64) #[4, 8, 15, 0.5]

        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(encoder_out_channels) #[2, 16, 31, 0.5]

        self.conv5 = nn.Conv2d(32, encoder_out_channels, kernel_size=1,
                               stride=1, padding="valid") #[2, 16, 31, 0.5]

        # kernel_size, stride and padding for each layer
        self.conv_feat = ([3, 3, 3, 3, 1], [2, 2, 2, 2, 1], [1, 1, 1, 1, 0])


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # x = F.sigmoid(self.conv5(x))
        x = F.relu(self.conv5(x))
        return x

    def conv_info(self):
        return self.conv_feat


class DecoderBlock(nn.Module):
    def __init__(self, encoder_out_channels, latent_channels, output_channels):
        super(DecoderBlock, self).__init__()

        self.conv_transpose1 = nn.ConvTranspose2d(latent_channels,
                                                  encoder_out_channels,
                                                  kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(encoder_out_channels)

        self.conv_transpose2 = nn.ConvTranspose2d(encoder_out_channels, 64,
                                                  kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv_transpose3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                                                  padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv_transpose4 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2,
                                                  padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv_transpose5 = nn.ConvTranspose2d(32, output_channels,
                                                  kernel_size=3, stride=2,
                                                  padding=1, output_padding=1)



    def forward(self, x):
        x = F.relu(self.bn1(self.conv_transpose1(x)))
        x = F.relu(self.bn2(self.conv_transpose2(x)))
        x = F.relu(self.bn3(self.conv_transpose3(x)))
        x = F.relu(self.bn4(self.conv_transpose4(x)))
        x = torch.sigmoid(self.conv_transpose5(x))
        return x


# if __name__ == '__main__':
#     encoder = EncoderBlock(1, 32)
#     print(encoder)
#     torchsummary.summary(encoder, (1, 28, 28))
#
#     decoder = DecoderBlock(32, 16, 1)
#     print(decoder)
#     torchsummary.summary(decoder, (16, 2, 2))

