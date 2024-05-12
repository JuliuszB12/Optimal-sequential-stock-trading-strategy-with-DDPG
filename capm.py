import statistics as stat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ddpg_agent import Agent
from environment_1 import StockTradingEnv

env = StockTradingEnv()
agent = Agent(state_size=env.state_dim, action_size=env.action_dim, random_seed=2)

# from test.py checkpoint files of weights in trained model for given half-year
agent.actor_local.load_state_dict(torch.load('2023_2/checkpoint_actor_400_300_1200_2e2.pth'))
agent.critic_local.load_state_dict(torch.load('2023_2/checkpoint_critic_400_300_1200_2e2.pth'))

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


for i in ['I_2019', 'II_2019', 'I_2020', 'II_2020', 'I_2021', 'II_2021', 'I_2022', 'II_2022', 'I_2023', 'II_2023']:
    if i == 'I_2019':
        df = pd.read_csv(f'total_assets/total_assets_{i}.csv')
        prices = pd.read_csv(f'prices/prices_{i}.csv')
        date = pd.read_pickle(f'test_{i}.pkl')
    else:
        df_temp = pd.read_csv(f'total_assets/total_assets_{i}.csv')
        df = pd.concat([df, df_temp])
        prices_temp = pd.read_csv(f'prices/prices_{i}.csv')
        prices = pd.concat([prices, prices_temp])
        date_temp = pd.read_pickle(f'test_{i}.pkl')
        date = pd.concat([date, date_temp])


df2 = pd.read_csv('DGS100.csv') # yearly risk-free rates for everyday as US10Y from internet
df3 = date.copy()
df3 = df3[['date', 'tic']]
df2["date"] = pd.to_datetime(df2["date"])
df3['date'] = pd.to_datetime(df3['date'])
df3 = df3.drop_duplicates(subset='date')
df3 = pd.merge(df3, df2, how='left', on='date')
df3 = df3.drop(columns=['tic'], axis=1)

risk_free_rates = np.array(df3["DGS10"].tolist()) / 100
date = df3["date"].tolist()

market_daily_returns = np.array([prices["values"].tolist()])
asset_daily_returns = np.array([df["values"].tolist()])

market_cumulative_returns = np.cumprod(1 + market_daily_returns) - 1
asset_cumulative_returns = np.cumprod(1 + asset_daily_returns) - 1

daily_risk_free_rates = (1 + risk_free_rates) ** (1 / 365) - 1
cumulative_risk_free_rate = np.cumsum(daily_risk_free_rates)*1.4

rolling_betas = np.empty(len(market_cumulative_returns))
expected_returns = np.empty(len(market_cumulative_returns))


for day in range(0, len(market_cumulative_returns)):
    if (day == 0) | (day == 1):
        rolling_betas[day] = np.nan
        expected_returns[day] = np.nan
        continue

    market_returns_subset = market_daily_returns[1:day+1]
    asset_returns_subset = asset_daily_returns[1:day+1]


    covariance = np.cov(market_returns_subset, asset_returns_subset)[0, 1]
    variance = np.var(market_returns_subset)

    rolling_betas[day] = covariance / variance if variance != 0 else np.nan

    if day > 1:
        expected_returns[day] = cumulative_risk_free_rate[day] + rolling_betas[day] * (
                    market_returns_subset[day-1] - cumulative_risk_free_rate[day])
    else:
        expected_returns[day+1] = np.nan
    print(day, date[day], "market", market_returns_subset[day-1], "expected", expected_returns[day], "asset", asset_returns_subset[day-1], "beta", rolling_betas[day])

# Display the results
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(asset_cumulative_returns) + 1), cap*(1 + asset_cumulative_returns), color='green', label='Model zielony')
plt.plot(np.arange(1, len(asset_cumulative_returns) + 1), cap*(1 + expected_returns), color='orange', label='Oczekiwana stopa zwrotu wg CAPM')
plt.plot(np.arange(1, len(asset_cumulative_returns) + 1), cap*(1 + market_cumulative_returns), color='gray', label='Średnia cen')
plt.plot(np.arange(1, len(asset_cumulative_returns) + 1), cap*(1 + cumulative_risk_free_rate), color='lightblue', label='Stopa wolna od ryzyka')
plt.ylabel('Wartość aktywów finansowych (w 10 milionach dolarów)')
plt.xlabel('Dzień')
ax.set_ylim(3500000, 12000000)
plt.legend(loc='upper left')
plt.show()
# env.shares[env.shares < 0] = 0
# print(env.shares.astype(np.int64)*env.close_ary[env.max_step])
# values = (env.shares.astype(np.int64)*env.close_ary[env.max_step]).tolist()
# values = list(filter(lambda x: x != 0, values))
# values = [round(i, 2) for i in values]
# fig, ax = plt.subplots(figsize=(9, 2))
# ax.barh('Udział', values[0], color=colors2[0])
# for i in range(1, len(values)):
#     ax.barh('Udział', values[i], left=sum(values[:i]), color=colors2[i])
# ax.set_title('Udział wartości wolumenów poszczególnych akcji w portfelu')
# plt.xticks([])
# plt.xlim([0, sum(values)])
# plt.show()
