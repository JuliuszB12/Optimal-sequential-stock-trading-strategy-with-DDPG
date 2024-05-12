import numpy as np
import random
import copy
from collections import namedtuple, deque

from actor_critic_networks import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # Rozmiar bufora powtórzeń
BATCH_SIZE = 128  # Rozmiar mini-wsadu
GAMMA = 0.99  # Stopa dyskontowa
TAU = 1e-3  # Hiperparametr delikatnej aktualizacji parametrów sieci celu Aktora i Krytyka
LR_ACTOR = 1e-4  # Współczynnik uczący Aktora
LR_CRITIC = 1e-3  # Współczynnik uczący Krytyka

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Agent():
    """Agent"""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Argumenty funkcji
        ======
            state_size (int): Wymiary Stanu
            action_size (int): Wymiary Akcji
            random_seed (int): Losowe ziarno dla powtarzalności rezultatów
        """
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Sieć Aktora (lokalna i celu)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Sieć Krytyka (lokalna i celu)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Noise
        self.noise = OUNoise(action_size, random_seed)

        # Bufor powtórzeń
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Zapisuje doświadczenie w buforze powtórzeń i wybiera losową próbkę z bufora do uczenia."""
        self.memory.add(state, action, reward, next_state, done)

        # Uczy się tylko, gdy w buforze powtórzeń jest wystarczająca liczba próbek
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Zwraca Akcję dla Stanu według Polityki."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Aktualizacja sieci.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        gdzie:
            actor_target(state) -> Akcja
            critic_target(state, action) -> Wartość Q(s,a)

        Argumenty funkcji:
        ======
            Doświadczenia (Tuple[torch.Tensor]): krotka (s, a, r, s', done) krotek
            gamma (float): stopa dyskontowa
        """
        states, actions, rewards, next_states, dones = doświadczenia

        # ---------------------------- Aktualizacja Krytyka ---------------------------- #
        # Zwraca następną Akcję i wartość Q(s+1,a) z sieci celu
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Oblicza cel wartości Q(s,a) dla danego s z użyciem Q(s+1,a)  (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Funkcja straty Krytyka
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Optymalizacja wag Krytyka
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- Aktualizacja Aktora ---------------------------- #
        # Funkcja straty Aktora
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Optymalizacja wag Aktora
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- Delikatna aktualizacja sieci celu ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Funkcja delikatnej aktualizacji dla sieci celu.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Argumenty funkcji
        ======
            local_model: model PyTorch do skopiowania wag
            target_model: model PyTorch do którego kopiuje się wagi
            tau (float): hiperparametr
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Proces Ornstein-Uhlenbeck. Usprawnia uczenie się Agenta"""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        """Zwracanie próbki noise"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Bufor do przechowywania dotychczasowych doświadczeń w postaci krotki."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Argumenty funkcji
        ======
            buffer_size (int): Maksymalna pojemność bufora
            batch_size (int): Rozmiar każdego wsadu treningowego
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Dodaje doświadczenie do bufora"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Wybiera losową próbkę doświadczeń z bufora"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Zwraca obecne zajęcie bufora doświadczeniami"""
        return len(self.memory)



########################################################################################################################
# References:
# [1] Udacity, Deep Reinforcement Learning, Github, 2020, online: https://github.com/udacity/deep-reinforcement-learning
########################################################################################################################