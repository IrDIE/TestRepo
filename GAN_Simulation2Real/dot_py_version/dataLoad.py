import torch
from PIL import Image
import os

from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms.functional import crop
import numpy as np
import torchvision.transforms as tf
import configparser as cfg


class SimRealDataset(Dataset):
    def __init__(self, root_img, transform = None):
        super().__init__()
        self.root_img = root_img

        self.transform = transform
        self.root_imgs = os.listdir(root_img)
        self.transforms = tf.Compose(
                          [
                              tf.ToTensor(),
                              tf.ToPILImage(),
                              RandomVerticalFlip(p=0.5)

                          ]   )


    def __len__(self):
        return len(self.root_imgs)

    def __getitem__(self, item):
        simreal_img = self.root_imgs[item % len(self.root_imgs)]
        simpeal_path = os.path.join(self.root_img, simreal_img)

        simreal_img =np.array(Image.open(simpeal_path).convert("RGB"))
        h, w, _ = simreal_img.shape
        # переворачиваем вертикально картинку до того, как поделим её на 2 части
        simreal_img = self.transforms(simreal_img)

        sim_img = crop(simreal_img,   0, w / 2, h, w / 2)
        sim_img_tensor = tf.ToTensor()(sim_img)
        real_img = crop(simreal_img, 0, 0, h, w / 2)
        real_img_tensor = tf.ToTensor()(real_img)

        if self.transform:

            sim_img_tensor = self.transform(sim_img_tensor)
            real_img_tensor = self.transform(real_img_tensor)

        return sim_img_tensor, real_img_tensor


def ntest():
    conf =cfg.ConfigParser()
    conf.read("conf.ini")
    path_img = conf['path']['root_img']
    transforms = tf.Compose([
                                tf.Resize((400, 600))
                            ])

    train_dataset = SimRealDataset(root_img = os.path.join(path_img) , transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    sim_img_tensor, real_img_tensor = next(iter(train_loader))
    print("Images shape = ", sim_img_tensor[0].shape)
    f, axarr = plt.subplots(1, 2, figsize=(15, 15))
    axarr[0].imshow(sim_img_tensor[0].permute(1, 2, 0))
    axarr[1].imshow(real_img_tensor[0].permute(1, 2, 0))

    plt.show()




if __name__ == "__main__":
    ntest()











