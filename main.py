import torch
import gym
from argparse import ArgumentParser
from tqdm import tqdm
from agent.PPO import PPO
from utils.utils import mkdir, save_json, draw, setup_seed
from os.path import join
import pandas as pd

def get_args():
    parser = ArgumentParser()
    # parser.add_argument("--env_name", type=str, default="BipedalWalker-v3")
    parser.add_argument("--env_name", type=str, default="LunarLander-v2")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--episode_len", type=int, default=300)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--num_optim", type=int, default=100)
    parser.add_argument("--num_episode", type=int, default=10000)
    parser.add_argument("--log_episode_gap", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--output_dir", type=str, default="result-debug")
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

    csv_path = join(args.output_dir, 'log.csv')
    df = pd.DataFrame()

    episode_rewards = []
    episode_losses = []
    losses_advantage = []
    losses_value_mse = []
    losses_entropy = []

    ppo = PPO(args.env_name,args.hidden_dim,args.num_layers,args.lr_actor,args.lr_critic,args.num_optim,args.gamma,args.eps,args.device,args.weight_advantages,args.weight_value_mse,args.weight_entropy)
    for e in tqdm(range(1,1+args.num_episode)):
        line = {"episode":e}
        episode_reward = ppo.collect_episode(args.episode_len)
        loss, loss_advantage, loss_value_mse, loss_entropy = ppo.update()

        episode_rewards.append(episode_reward)
        # episode_losses.append(loss)
        # losses_advantage.append(loss_advantage)
        # losses_value_mse.append(loss_value_mse)
        # losses_entropy.append(loss_entropy)

        line['loss'] = loss
        line['loss_advantage'] = loss_advantage
        line['loss_value_mse'] = loss_value_mse
        line['loss_entropy'] = loss_entropy
        df = pd.concat([df,pd.DataFrame([line])])
        df.to_csv(csv_path,index=None)

        if e % args.log_episode_gap == 0:
            draw(episode_rewards,fig_dir,e,'reward')
            # draw(episode_losses,fig_dir,e,'loss')
            # draw(losses_advantage,fig_dir,e,'advantage')
            # draw(losses_value_mse,fig_dir,e,'value_mse')
            # draw(losses_entropy,fig_dir,e,'entropy')



if __name__ == "__main__":
    args = post_process_args(get_args())
    main(args)

