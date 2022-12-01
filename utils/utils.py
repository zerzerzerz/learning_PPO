import os
import shutil
from os.path import isfile, isdir, join
from os import makedirs
import json
import torch
import random
import numpy as np
import time
from matplotlib import pyplot as plt
def mkdir(d):
    if isdir(d):
        shutil.rmtree(d)
    makedirs(d)


def load_json(json_path):
    with open(json_path, 'r') as f:
        res = json.load(f)
    return res


def save_json(obj, json_path):
    with open(json_path, 'w') as f:
        json.dump(obj, f, indent=4)


def setup_seed(seed = 3407):
     torch.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.cuda.manual_seed(seed)
     torch.backends.cudnn.benchmark = True


class Logger():
    def __init__(self,log_path) -> None:
        self.log_path = log_path
        with open(log_path,'w') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))

    def log(self,content,print_log=False):
        with open(self.log_path, 'a') as f:
            f.write(str(content))
            if print_log:
                print(content)


def draw(rewards,fig_dir,episode,label,figsize=(10,5)):
    plt.figure(figsize=figsize)
    plt.plot(rewards,label=f'episode {label}')
    plt.grid(which='both')
    plt.title(f"Episode = {episode}, episode {label}")
    plt.legend()
    plt.xlabel('episode')
    plt.ylabel(f'episode {label}')
    plt.savefig(join(fig_dir,f'{label}_episode_{episode}.png'))
    plt.close()