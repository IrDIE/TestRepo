import torch
import torch.nn as nn


# create blochk for discriminator
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)


# Patch_GAN_discriminator type
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[16, 32, 64, 128, 256]):
        super().__init__()
        self.initialBlock = nn.Sequential(

            nn.Conv2d(in_channels, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)

        )
        layers = []
        in_channels = features[0]
        for channels in features[1:]:
            layers.append(Block(in_channels, channels, 4, 2, 1))
            in_channels = channels
        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initialBlock(x)
        return torch.sigmoid(self.model(x))


def ntest():
    x_test = torch.rand(5, 3, 400, 600)
    discriminator = Discriminator()
    x_out = discriminator(x_test)
    print(x_out.shape)


if __name__ == "__main__":
    ntest()
