import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
import csv
import torch

from ddpg_agent import Agent
from environment_1 import StockTradingEnv

for i in [('I_2019', 2e3, '', 'january'), ('II_2019', 2e3, '_july', 'july'), ('I_2020', 2e3, '_2020', 'january'), ('II_2020', 2e3, '_2020', 'july'), ('I_2021', 2e3, '_2021', 'january'), ('II_2021', 2e3, '_2021', 'july'), ('I_2022', 2e3, '_2022', 'january'), ('II_2022', 2e3, '_2022', 'july'), ('I_2023', 2e3, '_2023', 'january'), ('II_2023', 2e3, '_2023', 'july_2')]:
    env = StockTradingEnv(period=i[0], max_stock=i[1])
    agent = Agent(state_size=env.state_dim, action_size=env.action_dim, random_seed=2)
    agent.actor_local.load_state_dict(torch.load(f'{i[2]}_{i[3]}/checkpoint_actor_400_300_1200_2e2.pth'))
    agent.critic_local.load_state_dict(torch.load(f'{i[2]}_{i[3]}/checkpoint_critic_400_300_1200_2e2.pth'))

    cap = 5000000

    def smart_agent(max_t=env.max_step):
        state = env.reset()
        score = 0
        total_assets = []
        for t in range(max_t):
            total_assets.append(env.total_asset)
            action = agent.act(state, add_noise=False)
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        print(f"Score: {score}")
        print(f"final total asset: {env.total_asset}")
        total_assets.append(env.total_asset)
        return total_assets


    total_assets = smart_agent()
    # df_assets = {'step': np.arange(1, len(total_assets) + 1).tolist(), 'score': total_assets}
    # df_assets = pd.DataFrame(df_assets)
    # df_assets.to_csv('total_assets_blue.csv')

    var = i[0]

    df = pd.read_pickle(f'test_{var}.pkl')
    prices = []
    prices_c = []
    for i in range(1, len(total_assets)):
        avg_price = stat.mean(df[df.index == i]['close'])/stat.mean(df[df.index == (i-1)]['close'])
        prices.append(avg_price - 1)
    for i in range(0, len(total_assets)):
        avg_price = stat.mean(df[df.index == i]['close']) / stat.mean(df[df.index == 0]['close'])
        prices_c.append(avg_price*cap)

    total_assets_i = np.diff(total_assets) / total_assets[:-1]
    total_assets_i = [[x] for x in total_assets_i]
    prices_i = [[i] for i in prices]
    total_assets_i.insert(0, [0])
    prices_i.insert(0,  [0])
    fields = ["values"]
    with open(f'total_assets/total_assets_{var}.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(total_assets_i)
    fields = ["values"]
    with open(f'prices/prices_{var}.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(prices_i)

    # total_assets_blue = pd.read_csv('total_assets_blue.csv')['score'].tolist()
    for i in range(len(total_assets)):
        print(i, round(total_assets[i], 2), round(prices_c[i], 2), round(total_assets[i]/cap, 3),
    round(prices_c[i]/cap, 3), round(total_assets[i]/cap, 3) - round(prices_c[i]/cap, 3))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(total_assets) + 1), total_assets, color='green', label='Model zielony')
    # plt.plot(np.arange(1, len(total_assets_blue) + 1), total_assets_green, label='Model niebieski')
    plt.plot(np.arange(1, len(prices_c) + 1), prices_c, color='gray', label='Średnia cen')
    plt.ylabel('Wartość aktywów finansowych (w milionach dolarów)')
    plt.xlabel('Dzień')
    plt.legend(loc='lower right')
    plt.show()
    print(env.shares.astype(np.int64))
