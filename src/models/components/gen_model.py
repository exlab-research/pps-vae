import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from .misc import ConvNeXtBlock


def pa(a: torch.Tensor, device: str):
    """Generative Model: factor - p(a)

    Parameters
    ----------
    a : torch.Tensor [batch_size, a_embed_dim]
        abstractive latent variable
    device : str
        cuda or cpu

    Returns
    -------
    pa_dist: td.Normal - Multivariate distribution with independent r.v's - (covariance has a diagonal matrix)
    """
    shape = a.shape
    pa_dist = td.Independent(td.Normal(torch.zeros(
        shape, device=device), torch.ones(shape, device=device)), 1)
    return pa_dist

 

class PxM_a(nn.Module):
    def __init__(self, M:int, a_embed_dim:int,  img_dim:int, device:str):
        """Generative Model: factor - p(xM|a)

        Parameters
        ----------
        M : int
            number of pixels in the latent representation xM
        a_embed_dim : int
            dimensionality of the embedding vector 'a'
        img_dim : int
            the resolution of image e.g. 64x64 then the img_dim is 64
        device : str
            gpu or cpu
        """
        super().__init__()

        self.img_dim = img_dim
        self.device = device

        self.a_to_xm_params = nn.Sequential(
            ConvNeXtBlock(7, a_embed_dim+2),
            ConvNeXtBlock(7, a_embed_dim+2),
            nn.Conv2d(a_embed_dim+2, M, 7, padding='same', bias=False))

    def forward(self, a:torch.Tensor):
        grid_x, grid_y = self.make_grid(a.shape[0])
        a = a.unsqueeze(2).unsqueeze(3)
        a = a.repeat(1, 1, self.img_dim, self.img_dim)
        grid_a = torch.concat([a, grid_x, grid_y], dim=1)
        xm_a_params = self.a_to_xm_params(grid_a)
        return xm_a_params

    def make_grid(self, batch_size: torch.Tensor):
        """The idea is taken from this paper: Spatial broadcast decoder:
            A simple architecture for learning disentangled representations in VAEs

        Parameters
        ----------
        batch_size : torch.Tensor
            batch size

        Returns
        -------
        grid_x: torch.Tensor [batch_size, 1, img_dim, img_dim]
        grid_y: torch.Tensor [batch_size, 1, img_dim, img_dim]
        """
        w = torch.linspace(-1, 1, steps=self.img_dim, device=self.device)
        h = torch.linspace(-1, 1, steps=self.img_dim, device=self.device)
        grid_x, grid_y = torch.meshgrid(w, h, indexing='ij')
        grid_x = grid_x.unsqueeze(0).unsqueeze(1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(1)
        grid_x = grid_x.repeat(batch_size, 1, 1, 1)
        grid_y = grid_y.repeat(batch_size, 1, 1, 1)
        return grid_x, grid_y




class PyM_xMa(nn.Module):
    def __init__(self, n_in_ch:int, n_latent_ch:int,  a_embed_dim:int, img_dim:int):
        """Generative Model: factor - p(yM|xM,a)

        Parameters
        ----------
        n_in_ch : int
            number of channels an image has (usually 1 or 3)
        n_latent_ch : int
            n filters
        a_embed_dim : int
            dimensionality of the embedding vector 'a'
        img_dim : int
            the resolution of image e.g. 64x64 then the img_dim is 64
        """
        super().__init__()
        self.n_in_ch = n_in_ch
        self.img_dim = img_dim
        self.coord_to_pxl_val = nn.Sequential(
            nn.Conv2d(a_embed_dim, n_latent_ch, 1),
            ConvNeXtBlock(7, n_latent_ch),
            ConvNeXtBlock(7, n_latent_ch),
            ConvNeXtBlock(7, n_latent_ch),
            nn.Conv2d(n_latent_ch, 2*self.n_in_ch, 1))

    def forward(self, a:torch.Tensor, xM:torch.Tensor, y:torch.Tensor):
        a = a.unsqueeze(2).unsqueeze(3)
        a = a.repeat(1, 1, self.img_dim, self.img_dim)
        a = xM * a

        mu_logscale = self.coord_to_pxl_val(a)
        mu, log_sigma = torch.split(
            mu_logscale, [self.n_in_ch, self.n_in_ch], dim=1)

        mu = mu.permute(0, 2, 3, 1)
        mu = torch.reshape(mu, [mu.shape[0], -1, self.n_in_ch])

        # Bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        sigma = sigma.permute(0, 2, 3, 1)
        sigma = torch.reshape(sigma, [sigma.shape[0], -1, self.n_in_ch])

        # Get the distribution
        normal = td.Normal(mu, sigma)
        dist = td.Independent(normal, 1)

        y = y.permute(0, 2, 3, 1)
        y = torch.reshape(y, [y.shape[0], -1, self.n_in_ch])

        log_prob = dist.log_prob(y)
        log_prob = log_prob * torch.reshape(xM.detach(), [xM.shape[0], -1])
        return log_prob, mu



class Conv(nn.Conv2d):
    def forward(self, input):
        return F.conv2d(input, self.weight.abs(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class CNP_CNNEncoder(nn.Module):
    def __init__(self, n_in_ch:int, n_latent_ch:int):
        """Encoder of ConvCNP

        Parameters
        ----------
        n_in_ch : int
            number of channels an image has (usually 1 or 3)
        n_latent_ch : int
            n filters
        """
        super().__init__()
        self.conv = Conv(n_in_ch, n_latent_ch, 11, stride=1,
                         bias=False, padding='same')

    def forward(self, ctx_signal:torch.Tensor, density:torch.Tensor):
        ctx_signal_ = self.conv(ctx_signal)
        density_ = self.conv(density.expand_as(ctx_signal))
        return ctx_signal_, density_


class CNP_CNNDecoder(nn.Module):
    def __init__(self, n_in_ch:int, n_latent_ch:int):
        """Decoder of ConvCNP

        Parameters
        ----------
        n_in_ch : int
            number of channels an image has (usually 1 or 3)
        n_latent_ch : int
            n filters
        """
        super().__init__()
        self.n_in_ch = n_in_ch
        self.pps_to_target = nn.Sequential(
            nn.Conv2d(2*n_latent_ch, n_latent_ch, 1),
            ConvNeXtBlock(7, n_latent_ch),
            ConvNeXtBlock(7, n_latent_ch),
            ConvNeXtBlock(7, n_latent_ch),
            ConvNeXtBlock(7, n_latent_ch),
            ConvNeXtBlock(7, n_latent_ch),
            nn.Conv2d(n_latent_ch, 2*self.n_in_ch, 1))

    def forward(self, ctx_signal_enc:torch.Tensor, density_enc:torch.Tensor):

        encoded = torch.concat([density_enc, ctx_signal_enc], dim=1)
        mu_logscale = self.pps_to_target(encoded)
        mu, log_sigma = torch.split(
            mu_logscale, [self.n_in_ch, self.n_in_ch], dim=1)

        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        mu = mu.permute(0, 2, 3, 1)
        mu = torch.reshape(mu, [mu.shape[0], -1, self.n_in_ch])

        sigma = sigma.permute(0, 2, 3, 1)
        sigma = torch.reshape(sigma, [sigma.shape[0], -1, self.n_in_ch])

        normal = td.Normal(mu, sigma)
        dist = td.Independent(normal, 1)

        return dist, mu, sigma


class PyT_yMxMxT(nn.Module):
    def __init__(self, n_in_ch:int, n_latent_ch:int):
        """Generative Model: factor - p(yT|xM,yM,xT) [ConvCNP model]

        Parameters
        ----------
        n_in_ch : int
            number of channels an image has (usually 1 or 3)
        n_latent_ch : int
            n filters
        """
        super().__init__()

        self.n_in_ch = n_in_ch
        self.encoder = CNP_CNNEncoder(self.n_in_ch, n_latent_ch)
        self.decoder = CNP_CNNDecoder(self.n_in_ch, n_latent_ch)

    def forward(self, xM:torch.Tensor, yM:torch.Tensor,  y:torch.Tensor):
        ctx_signal_enc, density_enc = self.encoder(yM, xM)
        dist, mu, _ = self.decoder(ctx_signal_enc, density_enc)

        y = y.permute(0, 2, 3, 1)
        y = torch.reshape(y, [y.shape[0], -1, self.n_in_ch])
        log_prob = dist.log_prob(y)
        log_prob = log_prob * \
            torch.reshape((1. - xM).detach(), [xM.shape[0], -1])
        return log_prob, mu

