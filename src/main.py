import os
import copy
import time
import datetime
import argparse

import torch

from models import PPS_VAE
from dataloader import celeba_loader, tiny_imagenet_loader, clevr_loader, fer2013_loader
from utils import ELBO, Timer, save_model, load_config_file, get_model_number_param, get_inp_out_dirs

torch.backends.cudnn.benchmark = True



def get_data(data_dir, config_object, device):
    dataset_name = config_object["DATASET"]["name"]
    batch_size = int(config_object["GENERAL"]["batch_size"])

    if dataset_name == "CelebA":
        train, test = celeba_loader(
            data_dir, batch_size, shuffle=True, device=device)
    elif dataset_name == "Imagenet":
        train, test = tiny_imagenet_loader(
            data_dir, batch_size, shuffle=True, device=device)
    elif dataset_name == "CLEVR":
        train, test = clevr_loader(
            data_dir, batch_size, shuffle=True, device=device)
    elif dataset_name == "FER2013":
        train, test = fer2013_loader(
            data_dir, batch_size, shuffle=True, device=device)
    else:
        print("WRONG DATASET")
        exit()
    return train, test


def get_model(config_object, device):
    M = int(config_object["MODEL"]["M"])
    K = int(config_object["MODEL"]["K"])
    img_ch = int(config_object["MODEL"]["img_ch"])
    img_dim = int(config_object["MODEL"]["img_dim"])
    pps_enc_ch = int(config_object["MODEL"]["pps_enc_ch"])
    a_enc_ch = int(config_object["MODEL"]["a_enc_ch"])
    a_embed_dim = int(config_object["MODEL"]["a_embed_dim"])
    yM_dec_ch = int(config_object["MODEL"]["yM_dec_ch"])
    yT_dec_ch = int(config_object["MODEL"]["yT_dec_ch"])
    model = PPS_VAE(M, K, img_ch, img_dim, pps_enc_ch, a_enc_ch,
                    a_embed_dim, yM_dec_ch, yT_dec_ch, device).to(device)
    return model

def get_optimizer(config_object):
    lr = float(config_object["GENERAL"]["lr"])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, amsgrad=True)
    return optimizer


def train(model, dataset, optimizer, device):
    model.train()
    epoch_elbo = ELBO()
    for data in dataset:
        imgs, _ = data
        imgs = imgs.to(device)

        optimizer.zero_grad()

        yT_log_prob, yM_log_prob, kl, _ = model(imgs)
        loss = -(torch.mean(yT_log_prob) + torch.mean(yM_log_prob) - torch.mean(kl))
    
        loss.backward()
        optimizer.step()
        
        batch_elbo = ELBO(yT_log_prob.mean().item(), yM_log_prob.mean().item(), kl.mean().item())
        epoch_elbo = epoch_elbo + batch_elbo
    return epoch_elbo


@torch.no_grad()
def test(model, dataset, device):
    model.eval()
    epoch_elbo = ELBO()
    for data in dataset:
        imgs, _ = data
        imgs = imgs.to(device)

        yT_log_prob, yM_log_prob, kl, _ = model(imgs)
        loss = -(torch.mean(yT_log_prob) + torch.mean(yM_log_prob) - torch.mean(kl))

        batch_elbo = ELBO(yT_log_prob.mean().item(), yM_log_prob.mean().item(), kl.mean().item())
        epoch_elbo = epoch_elbo + batch_elbo

    return epoch_elbo



if __name__ == '__main__':

    descr = "implementation of the PPS-VAE model see https://arxiv.org/abs/2305.18485"
    epil = "None"
    parser = argparse.ArgumentParser(description=descr, epilog=epil)
    parser.add_argument("--config_file", required=True, type=str, default="", help="name of the config file")  
    args = parser.parse_args()

    # the name of used config file #
    config_file = args.config_file
    print("===> Config file:", config_file)
    
    # tells if the model is trained on a gpu or a cpu #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('===> The device to run the model is:', device)

    # load config file #
    config_object = load_config_file(config_file)
    print("===> Config params:",{section: dict(config_object[section]) for section in config_object})
    ##

    # the location of the data sets (data dir) and
    # the storage of the experimntal results e.g. saved model (result_dir) #
    data_dir, out_dirs = get_inp_out_dirs(config_object)
    print('===> Data dir:', data_dir)
    print('===> Available output dirs:', out_dirs)

    # load data #
    train_data, test_data = get_data(data_dir, config_object, device)
    ##

    # get model #
    model = get_model(config_object, device)
    ##

    # optimizer #
    optimizer = get_optimizer(config_object)
    ##

    # Training loop #
    start_epoch = 1
    epochs = int(config_object["GENERAL"]["epochs"])
    
    n_train_batches = len(train_data)    
    n_test_batches = len(test_data)
    print('[Start Training]')
    for epoch in range(start_epoch, epochs + 1):
        with Timer('train epoch') as t:
            train_epoch_elbo = train(model, train_data, optimizer, device)
            
        train_epoch_elbo = train_epoch_elbo / n_train_batches
        print(f'train epoch: {epoch} || {train_epoch_elbo}')
        
        with Timer('test epoch') as t:
            test_epoch_elbo = test(model, test_data, device)

        test_epoch_elbo = test_epoch_elbo / n_test_batches
        print(f'test epoch: {epoch} || {test_epoch_elbo}')
        
        if epoch % 10 == 0:
            print('saving model')
            save_model(epoch, model, optimizer, test_epoch_elbo.elbo.item(), out_dirs['model_dir'])

    ##

    # misc #
    get_model_number_param(model)
    ##
