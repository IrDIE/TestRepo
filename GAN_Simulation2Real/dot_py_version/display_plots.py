from matplotlib import pyplot as plt


def plotf(image):
  return image[0].permute(1, 2, 0).cpu().detach().numpy()



def prot_imgs_trian(sim_img_tensor, fake_real, cycle_sim, real_img_tensor, fake_sim, cycle_real):
      fig = plt.figure(figsize=(20, 10))
      fig.add_subplot(2, 3, 1)

      plt.imshow(plotf(sim_img_tensor))
      fig.add_subplot(2, 3, 2)
      plt.imshow(plotf(fake_real))
      fig.add_subplot(2, 3, 3)
      plt.imshow(plotf(cycle_sim))

      fig.add_subplot(2, 3, 4)
      plt.imshow(plotf(real_img_tensor))
      fig.add_subplot(2, 3, 5)
      plt.imshow(plotf(fake_sim))
      fig.add_subplot(2, 3, 6)
      plt.imshow(plotf(cycle_real))

      plt.show()