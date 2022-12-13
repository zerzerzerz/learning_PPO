import torch
import gym
from argparse import ArgumentParser
from tqdm import tqdm
from agent.PPO import PPO
from utils.utils import mkdir, save_json, setup_seed, get_datetime, save_log, save_plot
from os.path import join
import pandas as pd

def get_args():
    parser = ArgumentParser()
    # parser.add_argument("--env_name", type=str, default="BipedalWalker-v3")
    parser.add_argument("--env_name", type=str, default="LunarLander-v2")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--episode_len", type=int, default=256)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--num_optim", type=int, default=100)
    parser.add_argument("--num_episode", type=int, default=10000)
    parser.add_argument("--log_episode_gap", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--output_dir", type=str, default="result-debug-discrete" + "-" + get_datetime())
    parser.add_argument("--weight_advantages", type=float, default=1.0)
    parser.add_argument("--weight_value_mse", type=float, default=0.5)
    parser.add_argument("--weight_entropy", type=float, default=0.01)

    args = parser.parse_args()
    return args


def post_process_args(args):
    return args


def main(args):
    setup_seed(0)
    mkdir(args.output_dir)
    fig_dir = join(args.output_dir,'fig')
    ck_dir = join(args.output_dir, 'checkpoint')
    mkdir(fig_dir)
    mkdir(ck_dir)

    save_json(vars(args), join(args.output_dir, 'args.json'))

    csv_path_train = join(args.output_dir, 'log_train.csv')
    csv_path_test = join(args.output_dir, 'log_test.csv')
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    ppo = PPO(args.env_name,args.hidden_dim,args.num_layers,args.lr_actor,args.lr_critic,args.num_optim,args.gamma,args.eps,args.device,args.weight_advantages,args.weight_value_mse,args.weight_entropy)
    for e in tqdm(range(1,1+args.num_episode)):
        ppo.collect_episode(args.episode_len)
        line_train = ppo.update()
        line_test = ppo.test_an_episode(args.episode_len)

        df_train = save_log(df_train, line_train, e, args.log_episode_gap, csv_path_train)
        df_test = save_log(df_test, line_test, e, args.log_episode_gap, csv_path_test)
        save_plot(df_train, e, args.log_episode_gap, fig_dir, *line_train.keys())
        save_plot(df_test, e, args.log_episode_gap, fig_dir, *line_test.keys())
        




if __name__ == "__main__":
    args = post_process_args(get_args())
    main(args)

