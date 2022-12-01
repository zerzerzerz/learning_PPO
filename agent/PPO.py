import gym
import torch
from model.model import ActorCritic
from model.buffer import Buffer
from settings.settings import DISCRETE_ACTION_SPACE
from torch.optim import Adam
from torch.nn import functional as F
from einops import rearrange, reduce, repeat

class PPO():
    def __init__(self,env_name,hidden_dim,num_layers,lr_actor,lr_critic,num_optim,gamma,eps,device,weight_advantages,weight_value_mse,weight_entropy) -> None:
        self.env = gym.make(env_name)
        self.num_optim = num_optim
        self.gamma = gamma
        self.eps = eps
        self.device = device
        self.weight_advantages = weight_advantages
        self.weight_value_mse = weight_value_mse
        self.weight_entropy = weight_entropy

        action_type = DISCRETE_ACTION_SPACE[env_name]
        self.action_type = action_type
        if action_type == "continuous":
            act_dim = self.env.action_space.shape[0]
        elif action_type == "discrete":
            act_dim = self.env.action_space.n
        else:
            raise NotImplementedError(f'action_type = {action_type} is not implemented')
        obs_dim = self.env.observation_space.shape[0]
        self.buffer = Buffer()
        self.policy = ActorCritic(obs_dim,act_dim,hidden_dim,num_layers,action_type)
        self.policy_old = ActorCritic(obs_dim,act_dim,hidden_dim,num_layers,action_type)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = Adam([
            {
                "params": self.policy.actor.parameters(), 
                "lr": lr_actor
            },
            {
                "params": self.policy.critic.parameters(), 
                "lr": lr_critic
            }
        ])
        self.to_cuda()
    

    def to_cuda(self):
        self.policy.to(self.device)
        self.policy_old.to(self.device)
    

    def to_cpu(self):
        self.policy_old.cpu()
        self.policy.cpu()


    def collect_episode(self,episode_len):
        with torch.no_grad():
            episode_reward = 0
            # 1d np
            state, info = self.env.reset()
            for timestep in range(episode_len):
                state = torch.from_numpy(state).to(device=self.device, dtype=torch.float32)
                state = rearrange(state,'d -> 1 d')

                # [1,1] [1,1] [1,] [1,] torch, discrete
                # [1,1] [1,d] [1,] [1,] torch, continuous
                state_value, action, action_logprob, entropy = self.policy_old(state)
                if self.action_type == "continuous":
                    action = action.detach().cpu().numpy()
                    action = rearrange(action, '1 d -> d')
                    state_next, reward, terminate, truncated, info = self.env.step(action)
                    action = rearrange(torch.from_numpy(action), 'd -> 1 d' )
                elif self.action_type == "discrete":
                    state_next, reward, terminate, truncated, info = self.env.step(action.item())
                else:
                    raise NotImplementedError(f'action_type = {self.action_type} is not implemented')

                terminate = terminate or truncated
                self.buffer.add(state,reward,terminate,action_logprob,action)
                state = state_next
                episode_reward += reward
                if terminate:
                    # print(f'{timestep+1}/{episode_len}')
                    break
            return episode_reward
    

    def update(self):
        rewards = []
        episode_loss = 0
        sum_loss_advantage = 0
        sum_loss_value_mse = 0
        sum_loss_entropy = 0

        discounted_reward = 0.0
        for reward, terminate in zip(reversed(self.buffer.rewards), reversed(self.buffer.terminates)):
            if terminate:
                discounted_reward = 0.0
            discounted_reward = discounted_reward * self.gamma + reward
            rewards.insert(0,discounted_reward)
        
        states_old = torch.cat(self.buffer.states,dim=0).to(device=self.device,dtype=torch.float32).detach()
        logprobs_old = torch.cat(self.buffer.logprobs,dim=0).to(device=self.device,dtype=torch.float32).detach()
        actions_old = torch.cat(self.buffer.actions, dim=0).to(device=self.device,dtype=torch.float32)
        rewards_old = torch.Tensor(rewards).to(device=self.device,dtype=torch.float32)

        rewards_old = (rewards_old - rewards_old.mean()) / (rewards_old.std() + 1e-6)

        for epoch in range(self.num_optim):
            state_values, logprobs, entropy = self.policy.evaluate(states_old,actions_old)
            state_values = rearrange(state_values, 'b 1 -> b')

            ratio = torch.exp(logprobs - logprobs_old.detach())
            advantages = rewards_old - state_values.detach()
            surr1 = ratio * advantages
            surr2 = torch.clip(ratio, 1-self.eps, 1+self.eps) * advantages

            loss_advantage = -torch.min(surr1,surr2).mean()
            loss_value_mse = F.mse_loss(state_values,rewards_old,reduction="mean")
            loss_entropy = -entropy.mean()

            loss = self.weight_advantages * loss_advantage \
                + self.weight_value_mse * loss_value_mse \
                + self.weight_entropy * loss_entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            episode_loss += loss.item()
            sum_loss_advantage += loss_advantage.item()
            sum_loss_value_mse += loss_value_mse.item()
            sum_loss_entropy += loss_entropy.item()
        
        self.buffer.clear()
        self.policy_old.load_state_dict(self.policy.state_dict())

        return episode_loss / self.num_optim, \
            sum_loss_advantage / self.num_optim, \
            sum_loss_value_mse / self.num_optim, \
            sum_loss_entropy / self.num_optim
            


    
