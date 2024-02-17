import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from .components import QxM_y, Qa_xMyM, pa, PxM_a, PyM_xMa, PyT_yMxMxT


class PPS_VAE(nn.Module):
    def __init__(self, M:int, K:int, img_ch:int,
                    img_dim:int, pps_enc_ch:int, a_enc_ch:int,
                    a_embed_dim:int, yM_dec_ch:int, yT_dec_ch:int, device:str):
        """ PPS-VAE Model

        Parameters
        ----------
        M : int
            number of pixels in the latent representation xM
        K : int
            number of pixels (in xM/yM) that are sampled in parallel
            (setting K > 1 allows to speed up training. Make sure that M is divisiable by K without a remainder)
        img_ch : int
            number of channels an image has (usually 1 or 3)
        img_dim : int
            the resolution of image e.g. 64x64 then the img_dim is 64
        pps_enc_ch : int
            n filters
        a_enc_ch : int
            n filters
        a_embed_dim : int
            dimensionality of the embedding vector 'a'
        yM_dec_ch : int
            n filters
        yT_dec_ch : int
            n filters
        device : str
            gpu or cpu
        """
        
        super().__init__()
        self.device = device
        self.M = M
        self.K = K
        self.img_dim = img_dim
        self.img_ch = img_ch
        
        # Inference Model: q(a, xM|y) =  q(a|xM,yM)q(xM|y)
        self.qxM_y = QxM_y(img_ch, pps_enc_ch,  self.M, self.K, self.device)
        self.qa_xMyM = Qa_xMyM(img_ch, a_enc_ch, a_embed_dim, img_dim)

        # Generative Model: p(a, xM, yM, yT) = p(yT|xM,yM,xT)p(yM|xM,a)p(xM|a)p(a)
        self.pa = lambda x, y: pa(x, y)
        self.pxM_a = PxM_a(self.M, a_embed_dim, img_dim, self.device)
        self.pyM_xMa = PyM_xMa(img_ch, yM_dec_ch,  a_embed_dim, img_dim)
        self.pyT_xMyMxT = PyT_yMxMxT(img_ch, yT_dec_ch)

    def forward(self, y: torch.Tensor):
        xM,  xM_M_ch, yM, a, xM_y_params, qa_xMyM_log_prob = self.inference(y)

        kl, yM_log_prob, yT_log_prob, yM_mu, yT_mu = self.scoring(
            a, xM, xM_M_ch, yM, y, xM_y_params, qa_xMyM_log_prob)

        plot_data = (yM, xM, yM_mu, yT_mu)

        return yT_log_prob, yM_log_prob, kl,  plot_data

    def inference(self, y:torch.Tensor):
        # xM_K_M_ch : List[torch.Tensor] [[batch_size, K, img_dim, img_dim]]
        # xM_K_M_y_params : List[torch.Tensor] [[batch_size, K, img_dim, img_dim]]
        xM_K_M_ch, xM_K_M_y_params = self.qxM_y(y)
        # join the list of M//K tensors [[batch_size, K, img_dim, img_dim]] into a single
        # tensor of [batch_size, M, img_dim, img_dim]
        xM_y_params = torch.hstack(xM_K_M_y_params)
        # reshape for the parallel computation of log ration in the KL term
        # xM_y_params [batch_size*M, img_dim*img_dim]
        xM_y_params = torch.reshape(xM_y_params, [y.shape[0]*self.M, -1])
        # join the list of M//K tensors [[batch_size, K, img_dim, img_dim]] into a single
        # tensor of [batch_size, M, img_dim, img_dim]
        xM_M_ch = torch.hstack(xM_K_M_ch)
        # reduce the chennel dim of batch_size, M, img_dim, img_dim] to batch_size, 1, img_dim, img_dim]
        xM = torch.sum(xM_M_ch, dim=1, keepdim=True)
        xM_norm_matrix = self.__get_target_mask(xM)
        # since we sample with replaement, a point may be sampled more than once
        # hence we use the normalisation procedure 
        xM = xM / xM_norm_matrix
        # a look up of M pixel values
        yM = y * xM

        a, qa_xMyM_log_prob = self.qa_xMyM(yM, xM)
        return xM, xM_M_ch, yM, a, xM_y_params, qa_xMyM_log_prob

    def scoring(self, a:torch.Tensor, xM:torch.Tensor,
                    xM_M_ch:torch.Tensor, yM:torch.Tensor,
                    y:torch.Tensor, xM_y_params:torch.Tensor,
                    qa_xMyM_log_prob:torch.Tensor):

        # kl:shape [B]
        kl = self.KL(a,  qa_xMyM_log_prob, xM_y_params, xM_M_ch)
        # yM_log_prob shape : [batch_size, img_dim*img_dim]
        # yM_mu shape : [batch_size, img_dim*img_dim, img_ch]
        yM_log_prob, yM_mu = self.pyM_xMa(a, xM, y)
        # yM_log_pro:  shape [B]
        yM_log_prob = torch.sum(yM_log_prob, dim=-1)
        # yT_log_prob shape : [batch_size, img_dim*img_dim]
        # yT_mu shape : [batch_size, img_dim*img_dim, img_ch]
        yT_log_prob, yT_mu = self.pyT_xMyMxT(xM, yM, y)
        # yT_log_prob shape : [B]
        yT_log_prob = torch.sum(yT_log_prob, dim=-1)

        return kl, yM_log_prob, yT_log_prob, yM_mu, yT_mu

    def KL(self, a:torch.Tensor,  qa_xMyM_log_prob:torch.Tensor,
                xM_y_params:torch.Tensor, xM_M_ch:torch.Tensor):

        pa_log_prob = self.pa(a, self.device).log_prob(a)
        # log_prob_ratio_a : shape [batch_size]
        log_prob_ratio_a = qa_xMyM_log_prob - pa_log_prob

        xM_a_params = self.pxM_a(a)
        # log_prob_ratio_xM : shape [batch_size]
        log_prob_ratio_xM  = self.__log_ratio_xM(xM_y_params, xM_a_params,  xM_M_ch)

        kl = log_prob_ratio_xM + log_prob_ratio_a
        return kl

    def __log_ratio_xM(self, xM_y_params:torch.Tensor, xM_a_params:torch.Tensor, xM:torch.Tensor):
        batch_size = xM.shape[0]
        xM_a_params = torch.reshape(xM_a_params, [batch_size*self.M, -1])
        
        pxM_a = F.softmax(xM_a_params, dim=-1)

        qxM_y = F.softmax(xM_y_params, dim=-1)

        xM = torch.reshape(xM, [batch_size*self.M, -1])

        log_q = torch.log(qxM_y + 1e-20)
        log_q = log_q * xM
        log_q = log_q.sum(-1)

        log_p = torch.log(pxM_a + 1e-20)
        log_p = log_p * xM
        log_p = log_p.sum(-1)

        log_ratio = log_q - log_p
        log_ratio = log_ratio.reshape([batch_size, self.M])
        log_ratio = torch.sum(log_ratio, dim=-1)
        return log_ratio

    @torch.no_grad()
    def __get_target_mask(self, densities: torch.Tensor):

        trgt_mask = torch.where(densities > 0., torch.zeros_like(
            densities, device=self.device), torch.ones_like(densities, device=self.device))
        norm_matrix = densities + trgt_mask
        return norm_matrix
