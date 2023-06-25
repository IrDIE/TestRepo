import warnings

import numpy as np
import torch
import torchvision.transforms as tf
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
import torch.nn as nn

from dataLoad import SimRealDataset
from discriminator_model import Discriminator
from display_plots import prot_imgs_trian
from generator_model import Generator
from save_load import load_GAN, save_GAN
import configparser as cfg
warnings.filterwarnings("ignore")


device= 'cuda' if torch.cuda.is_available() else 'cpu'

conf =cfg.ConfigParser()
conf.read("conf.ini")
path_img = conf['path']['root_img']
model_version=conf['hyperparameters']['model_version']
n_epochs=int(conf['hyperparameters']['n_epochs'])
lr = float(conf['hyperparameters']['lr'])
b1, b2 = float(conf['hyperparameters']['beta1']), float(conf['hyperparameters']['beta2'])
betas=(b1, b2)
lambda_cycle=float(conf['hyperparameters']['lambda_cycle'])
lambda_identity =float(conf['hyperparameters']['lambda_identity'])

root_path = path_img



# вспомогательная функция


def lr_linear_dacay(base_lr, iter, max_iter):
    return base_lr * ((1 - float(iter) / max_iter) )


def optimize_disc(opt1, opt2, type, epoch):
    if type == 'zero_grad':
        opt1.zero_grad()
        opt2.zero_grad()

    if type == 'step':
        opt1.step()
        opt2.step()
    if epoch > 20:
        reduced_lr = lr_linear_dacay(base_lr=lr, iter=epoch, max_iter=n_epochs)
        for g in opt1.param_groups:
            g['lr'] = reduced_lr

        for g in opt2.param_groups:
            g['lr'] = reduced_lr


def trainCycleGAN(n_epochs, model_version, model_SR, model_RS, opt_SR, opt_RS, lambda_cycle = lambda_cycle, lambda_identity = lambda_identity,
                  device= 'cuda' if torch.cuda.is_available() else 'cpu',
                  lr = lr, load = False):

    try:
        model_SR, opt_SR = load_GAN(MODEL_NAME = "SR_" + model_version, model = model_SR,optimizer = opt_SR)
        model_RS, opt_RS = load_GAN(MODEL_NAME = "RS_" + model_version, model = model_RS,optimizer = opt_RS)
    except:

        print("\n====================\n  looks like we gonna train from scratch  \n======================")

    gen_SR = model_SR['generator']
    gen_RS = model_RS['generator']

    discr_S = model_RS['discriminator']
    discr_R = model_SR['discriminator']
    # лосс дискриминатора, которому на вход фейковые или реальные симуляции
    loss_D_Sim_per_epochs = []
    # лосс дискриминатора, которому на вход реальные (сгенерированные или действительно) изображения
    loss_D_Real_per_epochs = []
    loss_G_Sim_per_epochs = []
    loss_G_Real_per_epochs = []
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(n_epochs):


        loss_D_Sim = []
        loss_D_Real = []

        loss_G_Sim =[]
        loss_G_Real =[]


        for i, (sim_img_tensor, real_img_tensor) in enumerate(train_loader):

            # TRAIN DISCRIMINATOR
            sim_img_tensor = sim_img_tensor.to(device)
            real_img_tensor = real_img_tensor.to(device)

            fake_real = gen_SR(sim_img_tensor)
            fake_sim = gen_RS(real_img_tensor)

            fake_sim_labels = discr_S(fake_sim.detach())
            real_sim_labels = discr_S(sim_img_tensor)

            fake_real_labels = discr_R(fake_real.detach())
            real_real_labels = discr_R(real_img_tensor)

            # лосс для дискриминатора, который получает на вход симуляцию
            loss_sim_d = criterias['mse']( fake_sim_labels, torch.zeros_like(fake_sim_labels) ) +\
                        criterias['mse'](real_sim_labels, torch.ones_like(real_sim_labels))
            # лосс для дискриминатора, который получает на вход real-images
            loss_real_d = criterias['mse']( real_real_labels, torch.ones_like(real_real_labels)) + \
             criterias['mse'](fake_real_labels, torch.zeros_like(fake_real_labels))

            loss_D_Sim.append(loss_sim_d.item())
            loss_D_Real.append(loss_real_d.item())

            loss_D_current = (loss_sim_d + loss_real_d) / 2
            #opt_discriminators.zero_grad()
            optimize_disc(opt_RS['discriminator'], opt_SR['discriminator'], type = 'zero_grad')

            loss_D_current.backward(retain_graph=True)
            #opt_discriminators.step()
            optimize_disc(opt_RS['discriminator'], opt_SR['discriminator'], type = 'step')

            #loss_D.append(loss_D_current.item())

            # TRAIN GENERATOR

            adversarial_loss_sim = criterias['mse'](fake_sim_labels.detach(), torch.ones_like(fake_sim_labels))
            adversarial_loss_real = criterias['mse'](fake_real_labels.detach(), torch.ones_like(fake_real_labels))
            adversarial_loss = (adversarial_loss_sim + adversarial_loss_real)

            cycle_sim_img = gen_RS(fake_real.detach())
            cycle_sim = criterias['l1'](cycle_sim_img, sim_img_tensor)

            cycle_real_img = gen_SR(fake_sim.detach())
            cycle_real = criterias['l1'](cycle_real_img, real_img_tensor)
            cycle_loss = lambda_cycle * (cycle_sim + cycle_real)

            sim_img_tensor_identity = gen_RS(sim_img_tensor.detach())
            real_img_tensor_identity = gen_SR(real_img_tensor.detach())
            identity_sim =  criterias['l1'](sim_img_tensor_identity, sim_img_tensor)
            identity_real = criterias['l1'](real_img_tensor_identity, real_img_tensor)
            identity_loss = lambda_identity *(identity_sim + identity_real )

            loss_G_Sim_ = adversarial_loss_sim + lambda_cycle*cycle_sim + lambda_identity*identity_sim
            loss_G_Real_ = adversarial_loss_real+lambda_cycle*cycle_real + lambda_identity*identity_real

            loss_G_current = adversarial_loss + cycle_loss + identity_loss
            optimize_disc(opt_RS['generator'], opt_SR['generator'], type = 'zero_grad')
            loss_G_current.backward(retain_graph=True)

            optimize_disc(opt_RS['generator'], opt_SR['generator'], type = 'step')

            loss_G_Sim.append(loss_G_Sim_.item())
            loss_G_Real.append(loss_G_Real_.item())

            #loss_G.append(loss_G_current.item())

            if i % 5 == 0:
                print(f"EPOCH = {epoch}\nloss d = {loss_D_current.item()} ---- loss g = {loss_G_current.item()}\n")
                prot_imgs_trian(sim_img_tensor, fake_real,cycle_sim_img, real_img_tensor,fake_sim, cycle_real_img )

        #loss_D_per_epochs.append(np.mean(loss_D))
        loss_D_Sim_per_epochs.append(np.mean(loss_D_Sim))
        loss_D_Real_per_epochs.append(np.mean(loss_D_Real))

        loss_G_Sim_per_epochs.append(np.mean(loss_G_Sim))
        loss_G_Real_per_epochs.append(np.mean(loss_G_Real))
        #loss_G_per_epochs.append(np.mean(loss_G))

        if epoch % 10 ==0:

            save_GAN(epoch, model_SR, optimizer = opt_SR,LOSS = [loss_D_Real_per_epochs, loss_G_Real_per_epochs ],\
                     MODEL_NAME = "SR_" + model_version)
            save_GAN(epoch, model_RS, optimizer = opt_RS,LOSS = [loss_D_Sim_per_epochs, loss_G_Sim_per_epochs ],  \
                     MODEL_NAME = "RS_" + model_version)


    return loss_D_Sim_per_epochs, loss_D_Real_per_epochs , loss_G_Sim_per_epochs, loss_G_Real_per_epochs




if __name__ == "__main__":

    transforms = tf.Compose(
        [
            tf.Resize((400, 600))

        ]
    )


    train_dataset = SimRealDataset(root_path, transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model_SR = {
        "discriminator": Discriminator().to(device),
        "generator": Generator().to(device)
    }

    model_RS = {
        "discriminator": Discriminator().to(device),
        "generator": Generator().to(device)
    }

    gen_SR = model_SR['generator']
    gen_RS = model_RS['generator']
    discr_S = model_RS['discriminator']
    discr_R = model_SR['discriminator']

    opt_SR = {
        "discriminator": torch.optim.Adam(params=discr_R.parameters(),
                                          lr=lr,
                                          betas=betas ),
        "generator": torch.optim.Adam(
            params=gen_SR.parameters(),
            lr=lr,
            betas=betas )
    }

    opt_RS = {
        "discriminator": torch.optim.Adam(params=discr_S.parameters(),
                                          lr=lr,
                                          betas=betas ),
        "generator": torch.optim.Adam(
            params=gen_RS.parameters(),
            lr=lr,
            betas=betas )
    }

    criterias = {
        "l1": nn.L1Loss(),
        "mse": nn.MSELoss()

    }


    loss_D_Sim_per_epochs, loss_D_Real_per_epochs, loss_G_Sim_per_epochs, loss_G_Real_per_epochs = \
        trainCycleGAN(n_epochs=n_epochs, model_version=model_version, model_SR=model_SR, model_RS=model_RS, opt_SR=opt_SR,
                      opt_RS=opt_RS)

    plt.plot(loss_D_Sim_per_epochs, label="loss Discr (Sim)")
    plt.plot(loss_D_Real_per_epochs, label="loss Discr (Real)")
    plt.legend()
    plt.show()


















