import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from typing import List
from .misc import ConvNeXtBlock, GroupBlock


class SamplingStep(nn.Module):
    def __init__(self, n_in_ch:int, n_latent_ch:int,  K:int,  device:str):
        """ Sampling of K out of M points

        Parameters
        ----------
        n_in_ch : int
            number of channels an image has (usually 1 or 3)_
        n_latent_ch : int
            n filters
        K : int
            number of pixels to be sampled in parallel
        device : str
            gpu or cpu
        """
        super().__init__()
        self.K = K
        self.device = device

        self.localistaion = nn.Sequential(
            # map from  n_in_ch to n_latent_ch
            nn.Conv2d(n_in_ch + 1,  n_latent_ch, 7, padding='same'),
            ConvNeXtBlock(7, n_latent_ch),
            ConvNeXtBlock(7, n_latent_ch),
            nn.Conv2d(n_latent_ch, self.K, 7, padding='same'),
            GroupBlock(self.K, 7),
            GroupBlock(self.K, 7))

        self.temperature = torch.tensor([1.], device=device)

    def forward(self, y:torch.Tensor, xM:List[torch.Tensor]):
        """ A single sampling step of K points
        
        Parameters
        ----------
        y : torch.Tensor [batch_size, n_in_ch, img_dim, img_dim]
            an input image
        xM : List[torch.Tensor] [[batch_size, K, img_dim, img_dim]]
            list of currently sampled locations

        Returns
        -------
        xK : torch.Tensor [batch_size, K, img_dim, img_dim]
            sampled  K points (locations) at the current step
        xK_params : torch.Tensor [batch_size, K, img_dim, img_dim]
            the dist. parameters of the sampled  K points (locations) at the current step
        """
        batch_size, _, h, w = y.shape
        # detaching the autoregressive grad. path allows to make the training of the model stable
        xM_up_to_t = torch.sum(torch.hstack(xM), dim=1, keepdim=True).detach()
        # xK_params shape: [batch_size, K, h, w]
        xK_params = self.localistaion(torch.concat([y, xM_up_to_t], dim=1))
        # xK_params shape: [batch_size*K, h*w]
        # do the reshape for the parallel sampling of the K points
        xK_params = torch.reshape(xK_params, (batch_size*self.K, -1))
        # xK shape: [batch_size*K, h*w]
        xK = F.gumbel_softmax(xK_params, tau=self.temperature, hard=True)
        # xK shape: [batch_size, K, h, w]
        xK = torch.reshape(xK, (batch_size, self.K, h, w))
        # xK_params shape: [batch_size, K, h, w]
        xK_params = torch.reshape(xK_params, (batch_size, self.K, h, w))
        return xK, xK_params


class QxM_y(nn.Module):
    def __init__(self, n_in_ch:int, n_latent_ch:int,  M:int, K:int,  device:str):
        """Inference Model: factor - q(xM|y)

        Parameters
        ----------
        n_in_ch : int
            number of channels an image has (usually 1 or 3)
        n_latent_ch : int
            n filters
        M : int
            number of pixels in the latent representation xM
        K : int
            number of pixels to be sampled in parallel
        device : str
            gpu or cpu
        """
        super().__init__()
        self.M = M
        self.K = K
        assert self.M % self.K == 0
        self.device = device
        self.sampling_step_t = SamplingStep(n_in_ch, n_latent_ch,  self.K,  self.device)

    def forward(self, y:torch.Tensor):
        """Sampling procedure of the M points

        Parameters
        ----------
        y : torch.Tensor [batch_size, n_in_ch, img_dim, img_dims]
            an input image

        Returns
        -------
        xM : List[torch.Tensor] [[batch_size, K, img_dim, img_dim]]
            list of sampled M locations
        xM_params : List[torch.Tensor] [[batch_size, K, img_dim, img_dim]]
            list of  the dist. parameters of the sampled  M points (locations)
        """
        b, _, w, h = y.shape
        xm_0 = torch.zeros([b, 1, w, h], device=self.device, dtype=torch.float32)
        # keep the list of the sampled xM coordinates
        xM = [xm_0]
        # keep the list of the dist. params the sampled xM coordinates
        xM_params = []
        # the sampling loop
        for _ in torch.arange(self.M//self.K):
            xK_t, xK_param_t = self.sampling_step_t(y, xM)
            xM.append(xK_t)
            xM_params.append(xK_param_t)
        # xM[1:] - we do not need xm_0 (the tensor of all zeros) for further processing, so just drop it
        return xM[1:], xM_params


class Qa_xMyM(nn.Module):
    def __init__(self, n_in_ch:int, n_latent_ch:int, latent_embed_dims:int, img_dim:int):
        """Inference Model: factor - q(a|xM,yM)

        Parameters
        ----------
        n_in_ch : int
            number of channels an image has (usually 1 or 3)
        n_latent_ch : int
            n filters
        latent_embed_dims : int
            dimensionality of the embedding vector 'a'
        img_dim : int
            the resolution of image e.g. 64x64 then the img_dim is 64
        """
        
        super().__init__()
        self.img_dim = img_dim
        w_sz, stride = self.__get_conv_hyperparams()

        self.pps_to_a_param = nn.Sequential(
            nn.Conv2d(n_in_ch + 1, n_latent_ch, w_sz, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(n_latent_ch, n_latent_ch*2, w_sz, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(n_latent_ch*2, n_latent_ch*4, w_sz, padding=1, stride=2))

        self.mean = nn.Conv2d(
            n_latent_ch * 4, latent_embed_dims, w_sz, stride=stride)
        self.log_scale = nn.Conv2d(
            n_latent_ch * 4, latent_embed_dims, w_sz, stride=stride)

    def forward(self, yM:torch.Tensor, xM:torch.Tensor):
        """Samping the latent variable 'a'

        Parameters
        ----------
        yM : torch.Tensor [batch_size, n_in_ch, img_dim, img_dim]
            M pixel values
        xM : torch.Tensor [batch_size, 1, img_dim, img_dim]
            M coord locations

        Returns
        -------
        a : torch.Tensor [batch_size, latent_embed_dims]
            sampled latent variable 'a'
        log_prob: torch.Tensor [batch_size]
            log probability of the posterior (is used in the KL term)
        """
        # param shape: [batch_size, n_latent_ch*4, ?, ?]; (?) - depends on the resolution of the image
        param = self.pps_to_a_param(torch.concat([xM, yM], dim=1))
        # mean shape: [batch_size, latent_embed_dims, 1, 1]
        mean = self.mean(param).squeeze(2).squeeze(2) # removes the singelton dims
        # log_scale shape: [batch_size, latent_embed_dims, 1, 1]
        log_scale = self.log_scale(param).squeeze(2).squeeze(2) # removes the singelton dims
        scale = F.softplus(log_scale) + 1e-5 # adding small constant may aid stability during training

        post_dist = td.Independent(td.Normal(mean, scale), 1)
        #a shape: [batch_size, latent_embed_dims]
        a = post_dist.rsample()
        # log_prob shape: [batch_size]
        log_prob = post_dist.log_prob(a)
        return a, log_prob

    def __get_conv_hyperparams(self):
        if self.img_dim == 28:
            w_sz = 3
            stride = 2

        elif self.img_dim == 32:
            w_sz = 4
            stride = 1

        elif self.img_dim == 64:
            w_sz = 6
            stride = 1

        return w_sz, stride
