import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import cv2
import GPU_ILT_AP as gpuiltap

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        return x.view(-1, 128)

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super(ActorCritic, self).__init__()

        # actor
        self.actor = nn.Sequential(
            nn.AvgPool2d(4),  # 512 -> 128
            nn.Conv2d(1, 32, 3, padding=1),
            GroupNorm32(32, 32),
            SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            GroupNorm32(32, 64),
            SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            GroupNorm32(32, 128),
            SiLU(),
            nn.AvgPool2d(32),
            Reshape(),
            nn.Linear(128, 64),
            SiLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

        # critic
        self.critic = nn.Sequential(
            nn.AvgPool2d(4),  # 512 -> 128
            nn.Conv2d(1, 32, 3, padding=1),
            GroupNorm32(32, 32),
            SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            GroupNorm32(32, 64),
            SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            GroupNorm32(32, 128),
            SiLU(),
            nn.AvgPool2d(32),
            Reshape(),
            nn.Linear(128, 64),
            SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, mode):
        if mode == 'train':
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

            action = dist.sample()
            action_logprob = dist.log_prob(action)

            return action.detach(), action_logprob.detach()

        else:
            action_probs = self.actor(state)
            action = torch.unsqueeze(action_probs.argmax(), 0)

            return action

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):

        self.device = device

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, mode='train'):

        if mode == 'train':
            with torch.no_grad():
                state = torch.unsqueeze(state, 1)
                action, action_logprob = self.policy_old.act(state, mode)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

        else:
            with torch.no_grad():
                state = torch.unsqueeze(state, 1)
                action = self.policy_old.act(state, mode)

        return action.item()

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(
            self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(
            self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(
            self.buffer.logprobs, dim=0)).detach().to(self.device)

        old_states = torch.unsqueeze(old_states, 1)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()

            # loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage))


def Env_Reward_Update(input_layout_path, input, step_now, action, attn_kernel_selections, kernels, kernels_ct, kernels_def, kernels_def_ct, weight, weight_def, device, ilt_iter=50, mode='train', ref_attn_kernel=[5, 30, 50, 70]):

    if mode == 'train':
        if step_now == 1:
            attn_kernel_size = int(action / 5)
            attn_kernel_selections.append(attn_kernel_size)
            kernel = np.ones((attn_kernel_size, attn_kernel_size), np.uint8)
            state = cv2.erode(torch.squeeze(input, 0).cpu().numpy(), kernel)
            state = torch.from_numpy(state).view(1, 512, 512).to(device)
            reward = 0
            done = 0

        elif step_now == 2:
            attn_kernel_size = action
            attn_kernel_selections.append(attn_kernel_size)
            kernel = np.ones((attn_kernel_size, attn_kernel_size), np.uint8)
            state = cv2.dilate(torch.squeeze(input, 0).cpu().numpy(), kernel)
            state = torch.from_numpy(state).view(1, 512, 512).to(device)
            reward = 0
            done = 0

        elif step_now == 3:
            attn_kernel_size = action
            attn_kernel_selections.append(attn_kernel_size)
            kernel = np.ones((attn_kernel_size, attn_kernel_size), np.uint8)
            state = cv2.dilate(torch.squeeze(input, 0).cpu().numpy(), kernel)
            state = torch.from_numpy(state).view(1, 512, 512).to(device)
            reward = 0
            done = 0

        elif step_now == 4:
            attn_kernel_size = action
            attn_kernel_selections.append(attn_kernel_size)
            state = None

            l2, pvb = gpuiltap.gpu_ilt_ap(input_layout_path, attn_kernel_selections, kernels, kernels_ct,
                                          kernels_def, kernels_def_ct, weight, weight_def, device, ilt_iter)

            reward = -(l2+pvb)
            done = 1

        return state, reward, attn_kernel_selections, done

    else:

        if step_now == 1:
            attn_kernel_size = int(action / 5)
            attn_kernel_selections.append(attn_kernel_size)
            kernel = np.ones((attn_kernel_size, attn_kernel_size), np.uint8)
            state = cv2.erode(torch.squeeze(input, 0).cpu().numpy(), kernel)
            state = torch.from_numpy(state).view(1, 512, 512).to(device)

        elif step_now == 2:
            attn_kernel_size = action
            attn_kernel_selections.append(attn_kernel_size)
            kernel = np.ones((attn_kernel_size, attn_kernel_size), np.uint8)
            state = cv2.dilate(torch.squeeze(input, 0).cpu().numpy(), kernel)
            state = torch.from_numpy(state).view(1, 512, 512).to(device)

        elif step_now == 3:
            attn_kernel_size = action
            attn_kernel_selections.append(attn_kernel_size)
            kernel = np.ones((attn_kernel_size, attn_kernel_size), np.uint8)
            state = cv2.dilate(torch.squeeze(input, 0).cpu().numpy(), kernel)
            state = torch.from_numpy(state).view(1, 512, 512).to(device)

        elif step_now == 4:
            attn_kernel_size = action
            attn_kernel_selections.append(attn_kernel_size)
            state = None

        return state, attn_kernel_selections
