import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():print("use gpu to train") 
else:print("use cpu to train")

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]

        self.actor = Actor(s_dim, 256, a_dim).to(device)
        self.actor_target = Actor(s_dim, 256, a_dim).to(device)
        self.critic = Critic(s_dim+a_dim, 256, a_dim).to(device)
        self.critic_target = Critic(s_dim+a_dim, 256, a_dim).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0).to(device)
        a0 = self.actor(s0).squeeze(0).detach().cpu().numpy()
        return a0
    
    def put(self, *transition): 
        if len(self.buffer)== self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 
        
        samples = random.sample(self.buffer, self.batch_size)
        
        s0, a0, r1, s1 = zip(*samples)
        
        s0 = torch.tensor(s0, dtype=torch.float).to(device)
        a0 = torch.tensor(a0, dtype=torch.float).to(device)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1).to(device)
        s1 = torch.tensor(s1, dtype=torch.float).to(device)
        
        def critic_learn():
            a1 = self.actor_target(s1).detach().to(device)
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach().to(device)
            
            y_pred = self.critic(s0, a0).to(device)
            
            loss_fn = nn.MSELoss().to(device)
            loss = loss_fn(y_pred, y_true).to(device)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
            
        def actor_learn():
            loss = -torch.mean( self.critic(s0, self.actor(s0)) ).to(device)
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
                                           
        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
    
    def save_model(self):
        torch.save(self.actor, "actor.pkl")
        torch.save(self.actor_target, "actor_target.pkl")
        torch.save(self.critic, "critic.pkl")
        torch.save(self.critic_target, "critic_target.pkl")
  
env = env.make(envname='Pendulum-v0')
# gym.make('Pendulum-v0')

env.reset()
# env.render()

params = {
    'env': env,
    'gamma': 0.99, 
    'actor_lr': 0.001, 
    'critic_lr': 0.001,
    'tau': 0.02,
    'capacity': 10000, 
    'batch_size': 32,
    }

agent = Agent(**params)
reward_list = []
for episode in range(200):
    s0 = env.reset()
    episode_reward = 0
    
    for step in range(500):
        # env.render()
        a0 = agent.act(s0)
        s1, r1, done, _ = env.step(a0)
        agent.put(s0, a0, r1, s1)

        episode_reward += r1 
        s0 = s1

        agent.learn()

    reward_list.append(episode_reward)
    print(episode, ': ', episode_reward)

agent.save_model()
plt.plot(reward_list)
plt.ylabel('reward')
plt.xlabel('episodes')
plt.savefig("result.png")
plt.show()
