import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    def __init__(self, kernel_size, ch_in):
        super(ConvNeXtBlock, self).__init__()

        self.conv2_in = nn.Conv2d(
            ch_in, ch_in, kernel_size, stride=1, padding='same')
        # self.layer_norm = nn.LayerNorm([ch_in, img_dim, img_dim])
        self.layer_norm = LayerNorm(
            ch_in, eps=1e-6, data_format="channels_first")
        self.conv2_mid = nn.Conv2d(ch_in, ch_in*4, 1, stride=1)
        self.conv2_out = nn.Conv2d(ch_in*4, ch_in, 1, stride=1)

    def forward(self, input_tensor):
        x = self.conv2_in(input_tensor)
        x = self.layer_norm(x)
        x = self.conv2_mid(x)
        x = F.leaky_relu(x)
        x = self.conv2_out(x)

        x = x + input_tensor
        return x


class GroupBlock(nn.Module):
    def __init__(self, ch_in, kernel_size):
        super().__init__()
        self.l = nn.Conv2d(ch_in, ch_in, kernel_size,
                           padding='same', bias=False, groups=ch_in)
        self.act = nn.LeakyReLU()
        self.norm = nn.GroupNorm(ch_in, ch_in)

    def forward(self, x):
        out = self.l(x)
        out = self.norm(out)
        out = self.act(out)
        return x + out


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
