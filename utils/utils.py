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
import pandas as pd

def get_datetime():
    t = time.localtime()
    t = time.strftime("%Y-%m-%d-%H-%M-%S", t)
    return t

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


def save_log(df:pd.DataFrame, new_line:dict, epoch, interval, csv_path):
    n = new_line.copy()
    n['epoch'] = epoch
    df = pd.concat([df, pd.DataFrame([n])])
    df = df.reset_index()
    if (epoch+1) % interval == 0:
        df.to(csv_path, index=None)
    return df


def save_plot(df:pd.DataFrame, epoch, interval, fig_dir, *keys):
    if (epoch+1) % interval == 0:
        for k in keys:
            plt.plot(df[k], label=k)
            plt.grid(which='both')
            plt.legend()
            plt.savefig(join(fig_dir, f'{k}-epoch={epoch}.png'))
        