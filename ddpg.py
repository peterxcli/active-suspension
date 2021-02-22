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
        action_value = x*100.0
        return action_value


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

        torch.autograd.set_detect_anomaly(True)
        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]
        self.hidden_dim = 256

        self.actor = Actor(s_dim, self.hidden_dim, a_dim).to(device)
        self.actor_target = Actor(s_dim, self.hidden_dim, a_dim).to(device)
        self.critic = Critic(s_dim+a_dim, self.hidden_dim, a_dim).to(device)
        self.critic_target = Critic(s_dim+a_dim, self.hidden_dim, a_dim).to(device)
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
            return loss.item()
            
        def actor_learn():
            loss = -torch.mean( self.critic(s0, self.actor(s0)) ).to(device)
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
                                           
        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        loss = critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

        return loss
    
    def save_model(self, aug=""):
        torch.save(self.actor, aug+"actor.pkl")
        torch.save(self.actor_target, aug+"actor_target.pkl")
        torch.save(self.critic, aug+"critic.pkl")
        torch.save(self.critic_target, aug+"critic_target.pkl")
  
# env = env.make(envname='Pendulum-v0')
env = env.make(envname='real')
# gym.make('Pendulum-v0')

# env.render()

params = {
    'env': env,
    'gamma': 0.99, 
    'actor_lr': 0.001, 
    'critic_lr': 0.001,
    'tau': 0.01,
    'capacity': 10000, 
    'batch_size': 32,
    }

agent = Agent(**params)
reward_list = []
loss_list = []
var = 3
MAX_EPI = 20
for episode in range(MAX_EPI):
    s0 = env.reset()
    episode_reward = 0
    
    for step in range(200):
        # env.render()
        
        a0 = agent.act(s0)
        a0 = np.clip(np.random.normal(a0, var), -100, 100)
        print('episode:', episode, ' :step:', step, ', act: ', a0, ", pos:", s0[0]*10.0)
        s1, r1 = env.step(a0)
        agent.put(s0, a0, r1, s1)

        episode_reward += r1 
        s0 = s1

        loss = agent.learn()
        loss_list.append(loss)

    reward_list.append(episode_reward)
    print(episode, ': ', episode_reward)
    # plt.plot(reward_list)
    # plt.ylabel('reward')
    # plt.xlabel('episodes')
    # plt.savefig("result" + str(episode) + ".png")

    # if (episode+1) % 10 == 0:
    #     agent.save_model(aug=str(episode+1))

    if episode == MAX_EPI-1:
        # print(reward_list)
        # print(loss_list)
        agent.save_model()
        loss_list = np.array(loss_list, dtype=np.float)
        reward_list = np.array(reward_list, dtype=np.float)
        np.savetxt("reward.csv", reward_list, delimiter=",")
        np.savetxt("loss.csv", loss_list, delimiter=",")

env.exit()
