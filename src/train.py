from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient


import random
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(self.nb_actions)
        else:
            return greedy_action(self.model, observation)

    def save(self, path='model.pt'):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(
            torch.load('model.pt', map_location=device)
        )


####################################
    

    def __init__(self):
        self.nb_actions = 4
        self.gamma = 0.98
        self.batch_size = 1024
        buffer_size = 500000
        self.memory = ReplayBuffer(buffer_size)
        self.epsilon_max = 1.
        self.epsilon_min = 0.01
        self.epsilon_stop = 10000
        self.epsilon_delay = 500
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = DQN(6,4).to(device)
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = torch.nn.SmoothL1Loss()
        lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = 1
        self.update_target_strategy = 'ema'
        self.update_target_freq = 500
        self.update_target_tau = 0.001
        self.monitoring_nb_trials = 0

    def MC_eval(self, env, nb_trials):
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, max_episode):
        episode_return = []
        MC_avg_total_reward = []
        MC_avg_discounted_reward = []
        V_init_state = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials>0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)
                    MC_avg_total_reward.append(MC_tr)
                    MC_avg_discounted_reward.append(MC_dr)
                    V_init_state.append(V0)
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          sep='')

                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state


def greedy_action(network, state):
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
    

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        self.hidden_dim = 256

        self.fc1 = nn.Linear(input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc5 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc6 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.last_fc = nn.Linear(self.hidden_dim, output_size)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x)) + x
        x = self.activation(self.fc3(x)) + x
        x = self.activation(self.fc4(x)) + x
        x = self.activation(self.fc5(x)) + x
        x = self.activation(self.fc6(x)) + x
        x = self.last_fc(x)
        return x


if __name__ == '__main__':
    agent = ProjectAgent()

    # Pre-fill the replay buffer
    x, _ = env.reset()
    for t in range(10000):
        a = env.action_space.sample()
        y, r, d, tr, _ = env.step(a)
        agent.memory.append(x, a, r, y, d)
        if d:
            x, _ = env.reset()
        else:
            x = y
    
    agent.train(300)
    agent.save()
