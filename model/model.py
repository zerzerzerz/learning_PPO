import torch
from einops import rearrange, reduce, repeat
from torch import nn
from torch.distributions import Categorical, MultivariateNormal
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,num_layers) -> None:
        super().__init__()
        self.model = []
        self.add_linear(input_dim,hidden_dim)
        for _ in range(num_layers):
            self.add_linear(hidden_dim,hidden_dim)
        self.model.append(nn.Linear(hidden_dim,output_dim))
        self.model = nn.Sequential(*self.model)
    
    def add_linear(self,input_dim,output_dim):
        self.model.extend([
            nn.Linear(input_dim,output_dim),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm1d(output_dim)
        ])
    
    def forward(self,x):
        return self.model(x)


class ActorDiscrete(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_dim,num_layers) -> None:
        '''return action prob'''
        super().__init__()
        self.actor = nn.Sequential(
            MLP(obs_dim,act_dim,hidden_dim,num_layers),
            nn.Softmax(dim=-1)
        )
    
    def forward(self,state):
        return self.actor(state)


class ActorContinuous(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_dim,num_layers) -> None:
        '''return action prob mean and std for each dim'''
        super().__init__()
        self.actor = MLP(obs_dim,2*act_dim,hidden_dim,num_layers)
        self.act_dim = act_dim

    def forward(self,state):
        tmp = self.actor(state)
        mean = tmp[:,0:self.act_dim]
        std = F.softplus(tmp[:,self.act_dim:])
        return mean, std


class ActorCritic(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_dim,num_layers,action_type:str) -> None:
        """
        return
            state_value
            action
            action_logprob
            entropy
        """
        super().__init__()
        if action_type == "continuous":
            # mean and std for each action component
            self.actor = ActorContinuous(obs_dim,act_dim,hidden_dim,num_layers)
        elif action_type == "discrete":
            # act_dim is number of action
            self.actor = ActorDiscrete(obs_dim,act_dim,hidden_dim,num_layers)
        else:
            raise NotImplementedError(f'action_type = {action_type} is not implemented')

        self.critic = MLP(obs_dim,1,hidden_dim,num_layers)
        self.action_type = action_type
        self.act_dim = act_dim
    

    def forward(self,obs):
        state_value = self.critic(obs)
        if self.action_type == "continuous":
            mean, std = self.actor(obs)
            cov = torch.diag_embed(std)
            dist = MultivariateNormal(mean,cov)
        elif self.action_type == "discrete":
            action_prob = self.actor(obs)
            dist = Categorical(action_prob)
        else:
            raise NotImplementedError(f'action_type = {self.action_type} is not implemented')
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return state_value, action, action_logprob, entropy
    

    def evaluate(self, state, action):
        if self.action_type == "continuous":
            mean, std = self.actor(state)
            cov = torch.diag_embed(std)
            dist = MultivariateNormal(mean, cov)
        elif self.action_type == "discrete":
            action_prob = self.actor(state)
            dist = Categorical(action_prob)
        else:
            raise NotImplementedError(f'action_type = {self.action_type} is not implemented')

        action_logprob = dist.log_prob(action)
        state_value = self.critic(state)
        entropy = dist.entropy()

        return state_value, action_logprob, entropy
    

