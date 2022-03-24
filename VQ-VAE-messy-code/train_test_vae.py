import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import random_split, DataLoader
import autoencoder
import time
import logging
import os

# Some of this is from https://github.com/nadavbh12/VQ-VAE/blob/master/main.py

def main():
    fm_dir = '/playpen/fashion-mnist'

    save_path = os.path.join(os.path.abspath('./runs'), time.strftime("%d-%m-%H%M%S", time.localtime()))
    lr = 2e-4
    k = 10
    hidden = 256
    num_channels = 1
    writer = SummaryWriter(save_path)
    cuda = False
    epochs = 100
    bs_train = 128

    model = autoencoder.VAE(hidden, k=k, num_channels=num_channels)

    if cuda:
        torch.cuda.set_device('cuda:0')
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.5)

    transform = transforms.Compose([transforms.ToTensor()])

    train_val_dataset = datasets.FashionMNIST(fm_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(fm_dir, train=False, transform=transform)
    train_dataset, val_dataset = random_split(train_val_dataset, [57000, 3000])

    train_loader = DataLoader(train_dataset, batch_size=bs_train)
    val_loader = DataLoader(val_dataset, batch_size=bs_train)
    
    for epoch in range(1, epochs+1):
        train_losses = train(epoch, model, train_loader, optimizer, writer, cuda)
        val_losses = validate(epoch, model, val_loader, writer, cuda, save_path)
        for k in train_losses.keys():
            name = k.replace('_train', '')
            train_name = k
            val_name = k.replace('train', 'val')
            writer.add_scalars(name, {'train': train_losses[train_name],
                                      'val': val_losses[val_name],
                                      })

def train(epoch, model, train_loader, opt, writer, cuda, log_interval=10):
    model.train()
    loss_dict = model.latest_losses()
    losses = {f'{k}_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {f'{k}_train': 0 for k, v in loss_dict.items()}
    batch_idx, data = None, None
    start_time = time.time()
    for batch_idx, (data,_) in enumerate(train_loader):
        if cuda:
            data = data.cuda()
        opt.zero_grad()
        outs = model(data)  # (x^, mu, logvar)
        loss = model.loss_function(data, *outs)
        loss.backward()
        opt.step()
        latest_losses = model.latest_losses()
        
        for key in latest_losses:
            losses[f'{key}_train'] += float(latest_losses[key])
            epoch_losses[f'{key}_train'] += float(latest_losses[key])
        
        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[f'{key}_train'] /= log_interval
            loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            print(f'Train Epoch: {epoch} [{batch_idx * len(data):5d}/{len(train_loader) * len(data)} ({int(100. * batch_idx / len(train_loader)):2d}%)]   time:'
                         f' {time.time()-start_time:3.2f}   {loss_string}')
            start_time = time.time()
            for key in latest_losses:
                losses[f'{key}_train'] = 0
    
    for key in epoch_losses:
        epoch_losses[key] /= (len(train_loader.dataset)/train_loader.batch_size)
    # TODO end of epoch logging code
    return epoch_losses

def validate(epoch, model, val_loader, writer, cuda, save_path):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_val': 0 for k, v in loss_dict.items()}
    idx, data = None, None
    with torch.no_grad():
        for batch_idx, (data,_) in enumerate(val_loader):
            if cuda:
                data = data.cuda()
            outs = model(data)
            model.loss_function(data, *outs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[f'{key}_val'] +=  float(latest_losses[key])
            if batch_idx == 0:
                write_images(data, outs, writer, 'val')
                save_reconstructed_images(data, epoch, outs[0], save_path, 'reconstruction_val')
                os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
                checkpoint_path = os.path.join(save_path, 'checkpoints', f'model_{epoch}.pth')
                torch.save(model.state_dict(), checkpoint_path)

        for key in losses:
            losses[key] /= (len(val_loader.dataset)/val_loader.batch_size)
        # TODO validation logging
        return losses

def write_images(data, outputs, writer, suffix):
    original = data.mul(0.5).add(0.5)
    original_grid = make_grid(original[:6])
    writer.add_image(f'original/{suffix}', original_grid)
    reconstructed = outputs[0].mul(0.5).add(0.5)
    reconstructed_grid = make_grid(reconstructed[:6])
    writer.add_image(f'reconstructed/{suffix}', reconstructed_grid)

def save_reconstructed_images(data, epoch, outputs, save_path, name):
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    comparison = torch.cat([data[:n],
                            outputs.view(batch_size, size[1], size[2], size[3])[:n]])
    save_image(comparison.cpu(),
               os.path.join(save_path, name + '_' + str(epoch) + '.png'), nrow=n, normalize=True)

if __name__=='__main__':
    main()