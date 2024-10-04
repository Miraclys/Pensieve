import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import torch.optim as optim
from torch.distributions import Categorical

from network import ActorNet, CriticNet

GAMMA = 0.99
EPOCH = 100000

def get_entropy_weight(epoch):
    if epoch >= EPOCH:
        return 0.1
    else:
        return 1 - 0.9 * epoch / EPOCH

class A3C:
    def __init__(self, is_central, s_dim, a_dim, actor_lr=1e-4, critic_lr=1e-3):
        self.is_central = is_central
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor = ActorNet(s_dim, a_dim)
        if self.is_central:
            self.critic = CriticNet(s_dim, a_dim)

            self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr, alpha=0.99, eps=1e-10)
            self.critic_optim = optim.RMSprop(self.critic.parameters(),lr=critic_lr,alpha=0.99,eps=1e-10)
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
        else:
            self.actor.eval()

    def train(self, s_batch, a_batch, r_batch, end_of_video, epoch):

        self.entropy_weight = get_entropy_weight(epoch)

        a_batch = torch.argmax(a_batch, dim=-1)

        V = self.critic(s_batch)
        V_batch = V.detach()
        R_batch = torch.zeros_like(r_batch)
        
        if end_of_video:
            R_batch[-1] = 0
        else:
            R_batch[-1] = V[-1].detach()

        for t in reversed(range(len(r_batch) - 1)):
            R_batch[t] = r_batch[t] + GAMMA * R_batch[t + 1]

        td_batch = R_batch - V
        # need to change
        critic_loss = torch.mean(td_batch ** 2)

        td_batch = R_batch - V_batch

        probs = F.softmax(self.actor(s_batch), dim=-1)
        m_probs = Categorical(probs)
        log_probs = m_probs.log_prob(a_batch)
        # need to change
        entropy_loss = -self.entropy_weight * torch.mean(m_probs.entropy())
        # need to change
        td_loss = torch.mean(log_probs * (-td_batch))
        actor_loss = td_loss + entropy_loss

        actor_loss.backward()
        critic_loss.backward()

    def update(self):
        self.actor_optim.step()
        self.actor_optim.zero_grad()
        if self.is_central:
            self.critic_optim.step()
            self.critic_optim.zero_grad()

    def select_action(self, state):
        with torch.no_grad():
            logits = self.actor(state)
            probs = F.softmax(logits, dim=-1)
            m_probs = Categorical(probs)
            bit_rate = m_probs.sample().item()
            return bit_rate

    def hard_update_actor(self, actor_params):
        for target_param, source_param in zip(self.actor.parameters(), actor_params):
            target_param.data.copy_(source_param.data)

    def hard_update_critic(self, critic_params):
        for target_param, source_param in zip(self.critic.parameters(), critic_params):
            target_param.data.copy_(source_param.data)

    def get_actor_param(self):
        return list(self.actor.parameters())
    
    def get_critic_param(self):
        return self.critic.parameters()

