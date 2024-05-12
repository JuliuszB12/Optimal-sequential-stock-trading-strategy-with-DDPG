import random
import torch
import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent
from environment import StockTradingEnv

env = StockTradingEnv()
agent = Agent(state_size=env.state_dim, action_size=env.action_dim, random_seed=2)


def ddpg(n_episodes=1200, max_t=env.max_step, print_every=10):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores


scores = ddpg()
df = {'episode': np.arange(1, len(scores) + 1).tolist(), 'score': scores}
df = pd.DataFrame(df)
df.to_csv('scores.csv')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores) # color=''
plt.ylabel('Skumulowana nagroda')
plt.xlabel('Epizod')
plt.show()