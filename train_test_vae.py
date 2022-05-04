import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
import autoencoder
import time
import logging
import os
import ImageDataset
from PIL import Image
from PixelCNN.pixelcnn import PixelCNN
from argparse import Namespace
from torchsummary import summary
import torch.nn.functional as F

# Some of this is from https://github.com/nadavbh12/VQ-VAE/blob/master/main.py

prior_cfg = Namespace(hidden_fmaps=30,
                        color_levels=10,
                        causal_ksize=7,
                        hidden_ksize=7,
                        out_hidden_fmaps=10,
                        hidden_layers=6)

def main():
    fm_dir = '/playpen/fashion-mnist'

    save_path = os.path.join(os.path.abspath('./runs'), time.strftime("%d-%m-%H%M%S", time.localtime()))
    lr = 1e-4
    k = 10
    hidden = 64
    num_channels = 1
    writer = SummaryWriter(save_path)
    cuda = True
    epochs = 100
    bs_train = 128
    model_load_path = 'runs/07-04-192420/checkpoints/model_10.pth'
    load_model = False

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    model = autoencoder.VAE(hidden, k=k, num_channels=num_channels)

    if cuda:
        torch.cuda.set_device('cuda:0')
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.5)

    ############  This is where we set up the dataset.  ############
    fm_dir = '/playpen/Downloads/apple-orange'
    transform = transforms.Compose([
        transforms.Resize(int(224 * 1.12), Image.BICUBIC),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # train_dataset = ImageDataset.ImageDataset(fm_dir, transforms_=transform,
    #                 unaligned=False, mode='train')
    # val_dataset = ImageDataset.ImageDataset(fm_dir, transforms_=transform,
    #                 unaligned=False, mode='test')

    train_dataset = MNIST('/playpen', train=True, download=True, transform=transforms.ToTensor())
    val_dataset = MNIST('/playpen', train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=bs_train, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=bs_train, shuffle=True, **kwargs)
    ############  End of dataset setup.  ############
    
    if not load_model:
        for epoch in range(1, epochs+1):
            train_losses = train(epoch, model, train_loader, optimizer, writer, cuda)
            val_losses = validate(epoch, model, val_loader, writer, cuda, save_path)
            
            for k in train_losses.keys():
                writer.add_scalar(f'{k}/train', train_losses[k], epoch)
            for k in val_losses.keys():
                writer.add_scalar(f'{k}/val', val_losses[k], epoch)
    else:
        model.load_state_dict(torch.load(model_load_path))
    writer.close()
    
    '''
    prior_save_path = os.path.join(save_path, 'prior')
    pwriter = SummaryWriter(prior_save_path)
    prior = PixelCNN(prior_cfg)
    prior.cuda()
    prior_opt = optim.Adam(prior.parameters(), lr=3e-4)
    step = 0
    model.eval()
    for epoch in range(1,10):
        prior.train()
        for batch_idx, (images,_) in enumerate(train_loader):
            prior_opt.zero_grad()
            if cuda:
                images = images.cuda()
            _, z_train = model.emb(model.encode(images))
            z_train = z_train.unsqueeze(1)
            # print(z_train.unique())
            outputs = prior(z_train.float(), torch.zeros((len(images),),dtype=int).cuda()).squeeze(1)
            # print(outputs.shape)
            _labels = z_train.cpu()
            _out = outputs.cpu()
            # print(_labels)
            # print(_out)
            loss = F.cross_entropy(outputs, z_train.long())
            pwriter.add_scalar('loss/train', loss, step)
            loss.backward()
            prior_opt.step()
            step += 1
            if batch_idx % 10==0:
                print(f'Train batch {batch_idx} of epoch {epoch}: loss {loss:.6f}')
        with torch.no_grad():
            loss = 0
            for batch_idx, (images,_) in enumerate(val_loader):
                if cuda:
                    images = images.cuda()
                _, z_val = model.emb(model.encode(images))
                z_val = z_val.unsqueeze(1)
                outputs = prior(z_val.float(), torch.zeros((len(images),),dtype=int).cuda()).squeeze(1)
                loss += F.cross_entropy(outputs, z_val.long())
                if batch_idx==0:
                    print(images.shape)
                    print(outputs.shape)
                    original_grid = make_grid(images[:6])
                    pwriter.add_image(f'image/original', original_grid, step)
                    reconstructed_grid = make_grid(outputs[:6].max(dim=1)[1])
                    pwriter.add_image(f'image/prior_output', reconstructed_grid, step)
            loss /= len(val_loader)
            pwriter.add_scalar('loss/val', loss, step)

            print(f'Validation of epoch {epoch}: average loss {loss:.6f}')
            os.makedirs(os.path.join(prior_save_path, 'checkpoints'), exist_ok=True)
            checkpoint_path = os.path.join(prior_save_path, 'checkpoints', f'prior_{epoch}.pth')
            torch.save(prior.state_dict(), checkpoint_path)

    pwriter.close()
    '''


def train(epoch, model, train_loader, opt, writer, cuda, log_interval=10):
    model.train()
    loss_dict = model.latest_losses()
    losses = {f'{k}': 0 for k, v in loss_dict.items()}
    epoch_losses = {f'{k}': 0 for k, v in loss_dict.items()}
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
            losses[f'{key}'] += float(latest_losses[key])
            epoch_losses[f'{key}'] += float(latest_losses[key])
        
        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[f'{key}'] /= log_interval
            loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            print(f'Train Epoch: {epoch} [{batch_idx * len(data):5d}/{len(train_loader) * len(data)} ({int(100. * batch_idx / len(train_loader)):2d}%)]   time:'
                         f' {time.time()-start_time:3.2f}   {loss_string}')
            start_time = time.time()
            for key in latest_losses:
                losses[f'{key}'] = 0
    
    for key in epoch_losses:
        epoch_losses[key] /= (len(train_loader.dataset)/train_loader.batch_size)
    # TODO end of epoch logging code
    return epoch_losses

def validate(epoch, model, val_loader, writer, cuda, save_path):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k : 0 for k, v in loss_dict.items()}
    idx, data = None, None
    with torch.no_grad():
        for batch_idx, (data,_) in enumerate(val_loader):
            if cuda:
                data = data.cuda()
            outs = model(data)
            model.loss_function(data, *outs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[f'{key}'] +=  float(latest_losses[key])
            if batch_idx == 0:
                write_images(epoch, data, outs, writer, 'val')

                samples = model.sample(8)
                writer.add_image('generated', make_grid(samples), global_step=epoch)

                save_reconstructed_images(data, epoch, outs[0], save_path, 'reconstruction_val')
                os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
                checkpoint_path = os.path.join(save_path, 'checkpoints', f'model_{epoch}.pth')
                torch.save(model.state_dict(), checkpoint_path)

        for key in losses:
            losses[key] /= (len(val_loader.dataset)/val_loader.batch_size)
        # TODO validation logging
        return losses

def write_images(step, data, outputs, writer, suffix):
    original = data.mul(0.5).add(0.5)
    original_grid = make_grid(original[:6])
    writer.add_image(f'original/{suffix}', original_grid, step)
    reconstructed = outputs[0].mul(0.5).add(0.5)
    print(reconstructed.shape)
    reconstructed_grid = make_grid(reconstructed[:6])
    writer.add_image(f'reconstructed/{suffix}', reconstructed_grid, step)

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