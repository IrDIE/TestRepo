import torch
import configparser as cfg

conf = cfg.ConfigParser()
conf.read("conf.ini")
path_saved_models = conf['path']['saved_models']


def save_GAN(epoch, model, optimizer,LOSS,  MODEL_NAME, root_path = path_saved_models):

    PATH_D = f"{root_path}/D/{MODEL_NAME}.pt"
    PATH_G = f"{root_path}/G/{MODEL_NAME}.pt"

    torch.save({
            'epoch': epoch,
            'model_state_dict': model['discriminator'].state_dict(),
            'optimizer_state_dict': optimizer['discriminator'].state_dict(),
            'loss': LOSS[0],
            }, PATH_D)

    torch.save({
                'epoch': epoch,
                'model_state_dict': model['generator'].state_dict(),
                'optimizer_state_dict': optimizer['generator'].state_dict(),
                'loss': LOSS[1],
                }, PATH_G)



def load_GAN(MODEL_NAME, model,optimizer, root_path = path_saved_models):
        PATH_D = f"{root_path}/D/{MODEL_NAME}.pt"
        PATH_G = f"{root_path}/G/{MODEL_NAME}.pt"
        checkpoint_D = torch.load(PATH_D)
        model['discriminator'].load_state_dict(checkpoint_D['model_state_dict'])
        optimizer['discriminator'].load_state_dict(checkpoint_D['optimizer_state_dict'])

        checkpoint_G = torch.load(PATH_G)
        model['generator'].load_state_dict(checkpoint_G['model_state_dict'])
        optimizer['generator'].load_state_dict(checkpoint_G['optimizer_state_dict'])

        return model, optimizer