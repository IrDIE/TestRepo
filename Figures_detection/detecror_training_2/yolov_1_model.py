import torch
import torch.nn as nn

# implementation of :
# https://arxiv.org/abs/1506.02640


class CNNBlock(nn.Module):
    def __init__(self , in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


architecture_config = [ # from paper
    (7, 64, 2, 3), # kernel / out_ch /  stride / padding
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class YOLO_model(nn.Module):
    def __init__(self, in_channels = 3, architecture_config = architecture_config,  **kwargs):
        super(YOLO_model, self).__init__()
        self.in_channels = in_channels
        self.arch = architecture_config
        self.conv_model = self._create_conv(self.arch)
        self.fc_model = self._create_fc(**kwargs)

    def _create_conv(self, arch):

        layers = []
        for layer in arch:
            if type(layer) == tuple:
                layers += [CNNBlock(in_channels = self.in_channels, out_channels = layer[1], kernel_size = layer[0],  stride=layer[2], padding=layer[3])]
                self.in_channels = layer[1]

            if type(layer) == str:
                layers += [nn.MaxPool2d(2, 2)]
            if type(layer) == list:
                n_repetitions = layer[-1]
                for _ in range(n_repetitions):
                    l1 = layer[0]
                    l2 = layer[1]  # [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
                    layers += [CNNBlock(in_channels=self.in_channels, out_channels=l1[1], kernel_size = l1[0],stride=l1[2], padding=l1[3])]
                    layers += [CNNBlock(in_channels=l1[1], out_channels=l2[1], kernel_size=l2[0], stride=l2[2], padding=l2[3])]
                    self.in_channels = l2[1]
        return nn.Sequential(*layers)

    def _create_fc(self, patch_size, n_boxes_per_img, input_img_width, n_classes):
        n_pathes = input_img_width // n_boxes_per_img
        fc_model = nn.Sequential(

            nn.Flatten(),
            nn.Linear(1024*patch_size*patch_size, n_pathes*n_pathes),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.Linear( n_pathes*n_pathes, patch_size*patch_size*(n_boxes_per_img*5  + n_classes))


        )

        return fc_model

    def forward(self, img):
        img = self.conv_model(img)
        res = self.fc_model(img)

        return  res


def try_model():
    yolo = YOLO_model(architecture_config = architecture_config, \
                      patch_size=4, n_boxes_per_img = 2, input_img_width = 256, \
                      n_classes = 4)

    x = torch.rand(1, 3, 256, 256)
    yolo(x) # output size =  torch.Size([1, 240]) = patch_size * patch_size * (10 + n_classes)

#try_model()