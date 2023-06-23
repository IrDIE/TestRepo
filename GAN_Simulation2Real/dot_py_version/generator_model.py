import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,in_ch, out_ch, is_up = False, identity = False, **kwargs ):
        super().__init__()
        self.conv = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, padding_mode = "reflect" , **kwargs) if not is_up else nn.ConvTranspose2d(in_ch, out_ch,  **kwargs),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2) if not identity else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)




class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.resBlock = nn.Sequential(

            ConvBlock(channels, channels, kernel_size = 3, stride = 1, padding = 1),
            ConvBlock(channels, channels, identity = True, kernel_size = 3, stride = 1, padding = 1)

        )

    def forward(self, x):
        return self.resBlock(x) + x


class Generator(nn.Module):
    def __init__(self, img_channels=3, channels = 32, num_residuals = 9):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(img_channels,channels, kernel_size= 7 ,padding =  3, stride= 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),

        )

        self.conv = nn.ModuleList([
            ConvBlock(channels, channels*2, kernel_size= 3 ,padding = 1, stride= 2),
            ConvBlock(channels*2, channels*4, kernel_size= 3 ,padding = 1, stride= 2)
        ])

        self.residuals_blocks = nn.Sequential(
            *[ResBlock(channels*4) for _ in range(num_residuals)]
        )

        self.up_conv = nn.ModuleList(
            [
                ConvBlock(channels*4, channels * 2, is_up=True, kernel_size=3, padding=1, stride=2, output_padding = 1),
                ConvBlock(channels * 2, channels *1,  is_up=True,kernel_size=3, padding=1, stride=2,output_padding = 1)
            ]
        )

        self.last = nn.Conv2d(channels,img_channels, 7, 1, 3, padding_mode="reflect" )


    def forward(self, x):
        x = self.initial(x)
        for downsample in self.conv:
            x = downsample(x)

        x = self.residuals_blocks(x)
        for upsampling in self.up_conv:
            x = upsampling(x)

        x = self.last(x)
        return torch.tanh(x)


def tryResBlock():
    x = torch.rand(5, 3, 256, 256)
    res_block = ResBlock(3)
    print(res_block(x).shape)

def tryGenerate():
    x = torch.rand(5, 3, 256, 256)
    gan = Generator(x.shape[1])
    x_out = gan(x)
    print("outpit size after Generator = ", x_out.shape)


if __name__ == "__main__":
    print("start testing ResBlock")
    tryResBlock()
    print("start testing Generate")
    tryGenerate()