import torch
import torch.nn as nn
import itertools
from util.image_pool import ImagePool
from .cycle_gan_model import CycleGANModel
from . import networks
import torch.nn.functional as F
import itertools

def kld(mu, log_var, kld_weight):
    return kld_weight * torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

class Conv2dDS(nn.Module):
    '''
    Represents a 2D depthwise separable convolution.
    '''
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2dDS, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(self.depth_conv, self.pointwise_conv)
    
    def forward(self, x):
        return self.conv(x)

class VAE(nn.Module):
    def __init__(self, enc, trans, dec, opt, device):
        super(VAE, self).__init__()
        self.opt = opt
        self.encoder = enc
        self.nlevels = opt.nlevels
        self.latent_dim = opt.latent_dim
        self.decoder = nn.Sequential(trans, dec)
        self.fc_mu = nn.Conv2d(opt.ngf*2**self.nlevels, self.latent_dim, opt.crop_size//(2**self.nlevels), device=device)
        self.fc_var = nn.Conv2d(opt.ngf*2**self.nlevels, self.latent_dim, opt.crop_size//(2**self.nlevels), device=device)
        self.decoder_input = nn.ConvTranspose2d(self.latent_dim, opt.ngf*2**self.nlevels, opt.crop_size//(2**self.nlevels), device=device)
    
    def encode(self, x):
        y = self.encoder(x)
        mu = self.fc_mu(y)
        log_var = self.fc_var(y)
        return [mu, log_var]
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def decode(self, z):
        y = self.decoder_input(z)
        y = self.decoder(y)
        return y

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return [self.decode(z), x, mu, logvar]
    
    def loss_function(self, *args, **kwargs):
        recons = args[0]
        x = args[1]
        mu = args[2]
        log_var = args[3]

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = 1
        recons_loss = F.mse_loss(recons, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return loss

class VarCycleGANModel(CycleGANModel):
    def __init__(self, opt):
        CycleGANModel.__init__(self, opt)
        self.loss_names.extend(['vae_A', 'vae_B'])

        self.encAB, self.transAB, self.decAB = [m for m in self.netG_A.module.children()][:3]
        self.encBA, self.transBA, self.decBA = [m for m in self.netG_B.module.children()][:3]
        self.netvae_A = VAE(self.encAB, self.transBA, self.decBA, opt, self.device)
        self.netvae_B = VAE(self.encBA, self.transAB, self.decAB, opt, self.device)

        self.model_names.extend(['vae_A', 'vae_B'])
        if self.isTrain:
            self.optimizer_VAE = torch.optim.Adam(itertools.chain(self.netvae_A.parameters(), self.netvae_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_VAE)
    def setup(self, opt):
        CycleGANModel.setup(self, opt)
        # All parameters are now loaded from disk
        if not self.isTrain:
            # "cross over" encoders so we can variationally translate A->B and B->A
            # netvae_A should *output* in domain A, and netvae_B should *output* in domain B            self.fc_mu_AB = self.netvae_A.fc_mu
            self.fc_mu_AB = self.netvae_A.fc_mu
            self.fc_mu_BA = self.netvae_B.fc_mu
            self.fc_var_AB = self.netvae_A.fc_var
            self.fc_var_BA = self.netvae_B.fc_var
            self.netvae_A.fc_mu = self.fc_mu_BA
            self.netvae_B.fc_mu = self.fc_mu_AB
            self.netvae_A.fc_var = self.fc_var_BA
            self.netvae_B.fc_var = self.fc_var_AB
            
    def forward(self):
        CycleGANModel.forward(self)
        if not self.isTrain:
            self.fake_A = self.netvae_A(self.real_B)[0]
            self.fake_B = self.netvae_B(self.real_A)[0]

    def backward_VAE(self, net, real):
        outs = net(real)
        loss = net.loss_function(*outs)
        loss.backward()
        return loss

    def optimize_parameters(self):
        CycleGANModel.optimize_parameters(self)
        self.set_requires_grad([self.netvae_A, self.netvae_B], True)
        self.optimizer_VAE.zero_grad()
        self.loss_vae_A = self.backward_VAE(self.netvae_A, self.real_A)
        self.loss_vae_B = self.backward_VAE(self.netvae_B, self.real_B)
        self.optimizer_VAE.step()