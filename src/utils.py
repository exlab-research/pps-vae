import os
import shutil
import sys
import time
from configparser import ConfigParser
from dataclasses import dataclass, field, asdict

import torch


@dataclass
class ELBO:
    """Expects values to be 0-order tensors"""

    yT_term: torch.float = field(default=torch.zeros(1).item())
    yM_term: torch.float = field(default=torch.zeros(1).item())
    kl_term: torch.float = field(default=torch.zeros(1).item())
    elbo: torch.float = field(init=False)

    def __post_init__(self):
        self.elbo = self.yM_term + self.yT_term - self.kl_term

    def __add__(self, other):
        if not isinstance(other, ELBO):
            return NotImplemented
        return ELBO(self.yT_term + other.yT_term,
                    self.yM_term + other.yM_term,
                    self.kl_term + other.kl_term)

    def __mul__(self, other):
        if isinstance(other, ELBO):
            return NotImplemented
        return ELBO(self.yT_term * other,
                    self.yM_term * other,
                    self.kl_term * other)

    def __truediv__(self, other):
        if isinstance(other, ELBO):
            return NotImplemented
        return ELBO(self.yT_term / other,
                    self.yM_term / other,
                    self.kl_term / other)

    def __repr__(self):
        return f'elbo: {self.elbo:.3f},' \
            f' yT_term: {self.yT_term:.3f},' \
            f' yM_term: {self.yM_term:.3f},' \
            f' kl_term: {self.kl_term:.3f}'

    def asdict(self):
        return {k: v.item() for k, v in asdict(self).items()}



class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))



def save_model(epoch, model, optimizer, loss, model_dir):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, model_dir+'epoch_'+str(epoch)+'.pth')
    return None


def load_config_file(exp_name):
    config_object = ConfigParser()
    config_object.read(exp_name)
    return config_object


def get_model_number_param(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The network has {params} trainable parameters")
    return None


def get_inp_out_dirs(config_object):
    data_dir = "../data"
    result_dir = "../results/"
    if os.path.exists(result_dir) == False:
        os.mkdir(result_dir)
    
    exp_name = config_object["DATASET"]["name"]+"-" + config_object["GENERAL"]["exp_name"] + \
        "-" + "M_" + config_object["MODEL"]["M"]

    base_dir = result_dir + exp_name
    if os.path.exists(base_dir) == False:
        os.mkdir(base_dir)
    
    model_dir = base_dir + "/Model/"
    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)
    
    out_dirs = {'result_dir':result_dir, 'base_dir':base_dir, 'model_dir':model_dir}
    
    return data_dir, out_dirs
