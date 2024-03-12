import pdb

import torch
import torchsummary
from torch import nn
from torch.nn import functional as F

class EncoderBlock(nn.Module):
    def __init__(self, input_channels, encoder_out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 1048, kernel_size=1, stride=1,
                               padding="valid")
        self.bn1 = nn.BatchNorm2d(1048) #[14, 2, 3, 0.5]     [n_out, j_out, r_out,
        # start_out ]

        self.conv2 = nn.Conv2d(1048, 512, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512) #[7, 4, 7, 0.5]

        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding="valid")
        self.bn3 = nn.BatchNorm2d(512) #[4, 8, 15, 0.5]

        self.conv4 = nn.Conv2d(512, 256, kernel_size=3,
                               stride=2, padding=1) #[2, 16, 31, 0.5]
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, encoder_out_channels, kernel_size=1,
                               stride=1, padding="valid") #[2, 16, 31, 0.5]

        # kernel_size, stride and padding for each layer
        self.conv_feat = ([1, 3, 1, 3, 1], [1, 2, 1, 2, 1], [0, 1, 0, 1, 0])


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
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

        self.conv_transpose2 = nn.ConvTranspose2d(encoder_out_channels, 512,
                                                  kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        self.conv_transpose3 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2,
                                                  padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(512)

        self.conv_transpose4 = nn.ConvTranspose2d(512, 1048, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm2d(1048)

        self.conv_transpose5 = nn.ConvTranspose2d(1048, output_channels,
                                                  kernel_size=3, stride=2,
                                                  padding=1, output_padding=1)



    def forward(self, x):
        x = F.relu(self.bn1(self.conv_transpose1(x)))
        x = F.relu(self.bn2(self.conv_transpose2(x)))
        x = F.relu(self.bn3(self.conv_transpose3(x)))
        x = F.relu(self.bn4(self.conv_transpose4(x)))
        x = F.relu(self.conv_transpose5(x))
        return x


if __name__ == '__main__':
    # encoder = EncoderBlock(256, 256)
    # print(encoder)
    # torchsummary.summary(encoder, (256, 28, 28))

    decoder = DecoderBlock(256, 128, 256)
    print(decoder)
    torchsummary.summary(decoder, (128, 7, 7))

