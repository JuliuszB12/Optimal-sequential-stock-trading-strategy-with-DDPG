import os
import numpy as np
import numpy.random as rd
import pandas as pd


class StockTradingEnv:
    def __init__(self, initial_amount=5e6, max_stock=2e2, cost_pct=1e-3, gamma=0.99,
                 beg_idx=0, end_idx=1000):
        self.pd_data = './train.pkl' # './test.pkl' for test
        self.np_data = './train.numpy.npz' # './test.numpy.npz' for test

        self.close_prc, self.tech_idc = self.load_data()
        # self.close_prc = self.close_prc[beg_idx:end_idx]
        # self.tech_idc = self.tech_idc[beg_idx:end_idx]
        print(f"| StockTradingEnv: close_prc.shape {self.close_prc.shape}")
        print(f"| StockTradingEnv: tech_idc.shape {self.tech_idc.shape}")

        self.max_stock = max_stock
        self.cost_pct = cost_pct
        self.reward_scale = 2 ** -12
        self.initial_amount = initial_amount
        self.gamma = gamma

        # reset()
        self.day = None
        self.rewards = None
        self.total_asset = None
        self.cumulative_returns = 0
        self.if_random_reset = True # False for test

        self.amount = None
        self.shares = None
        self.shares_num = self.close_prc.shape[1]
        amount_dim = 1

        # environment information
        self.env_name = 'StockTradingEnv-v2'
        self.state_dim = self.shares_num + self.close_prc.shape[1] + self.tech_idc.shape[1] + amount_dim
        self.action_dim = self.shares_num
        self.if_discrete = False
        self.max_step = self.close_prc.shape[0] - 1
        self.target_return = +np.inf

    def reset(self):
        self.day = 0
        if self.if_random_reset:
            self.amount = self.initial_amount * rd.uniform(0.9, 1.1)
            self.shares = (np.abs(rd.randn(self.shares_num).clip(-2, +2)) * 2 ** 6).astype(int)
        else:
            self.amount = self.initial_amount
            self.shares = np.zeros(self.shares_num, dtype=np.float32)

        self.rewards = []
        self.total_asset = (self.close_prc[self.day] * self.shares).sum() + self.amount
        return self.get_state()

    def get_state(self):
        state = np.hstack((np.tanh(np.array(self.amount * 2 ** -16)),
                           self.shares * 2 ** -9,
                           self.close_prc[self.day] * 2 ** -7,
                           self.tech_idc[self.day] * 2 ** -6,))
        return state

    def step(self, action):
        self.day += 1

        action = action.copy()
        action[(-0.1 < action) & (action < 0.1)] = 0
        action_int = (action * self.max_stock).astype(int)
        # actions initially is scaled between -1 and 1
        # convert into integer because we can't buy fraction of shares

        for index in range(self.action_dim):
            stock_action = action_int[index]
            adj_close_price = self.close_prc[self.day, index]  # `adjcp` denotes adjusted close price
            if stock_action > 0:  # buy_stock
                delta_stock = min(self.amount // adj_close_price, stock_action)
                self.amount -= adj_close_price * delta_stock * (1 + self.cost_pct)
                self.shares[index] += delta_stock
            elif self.shares[index] > 0:  # sell_stock
                delta_stock = min(-stock_action, self.shares[index])
                self.amount += adj_close_price * delta_stock * (1 - self.cost_pct)
                self.shares[index] -= delta_stock

        total_asset = (self.close_prc[self.day] * self.shares).sum() + self.amount
        reward = (total_asset - self.total_asset) * self.reward_scale
        self.rewards.append(reward)
        self.total_asset = total_asset

        done = self.day == self.max_step
        if done:
            reward += 1 / (1 - self.gamma) * np.mean(self.rewards)
            self.cumulative_returns = total_asset / self.initial_amount * 100  # todo

        state = self.get_state()
        return state, reward, done, {}
    
    def load_data(self, tech_id_list=None):
        tech_id_list = [
                "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma", "aroon"
        ] if tech_id_list is None else tech_id_list

        if os.path.exists(self.np_data):
            ary_dict = np.load(self.np_data, allow_pickle=True)
            close_prc = ary_dict['close_prc']
            tech_idc = ary_dict['tech_idc']
        elif os.path.exists(self.pd_data):  # convert pandas.DataFrame to numpy.array
            df = pd.read_pickle(self.pd_data)

            tech_idc = []
            close_prc = []
            df_len = len(df.index.unique())  # df_len = max_step
            for day in range(df_len):
                item = df.loc[day]

                tech_items = [item[tech].values.tolist() for tech in tech_id_list]
                tech_items_flatten = sum(tech_items, [])
                tech_idc.append(tech_items_flatten)

                close_prc.append(item.close)

            close_prc = np.array(close_prc)
            tech_idc = np.array(tech_idc)

            np.savez_compressed(self.np_data, close_prc=close_prc, tech_idc=tech_idc, )
        else:
            raise FileNotFoundError("No data")
        return close_prc, tech_idc



#################################################################
# References:
# [1] Liu, Xiao-Yang and Li, Zechu and Zhu, Ming and Wang, Zhaoran and Zheng, Jiahao, ElegantRL: Massively Parallel Framework for Cloud-native Deep Reinforcement Learning, Github, 2021, # online: https://github.com/AI4Finance-Foundation/ElegantRL
# [2] Liu, Xiao-Yang and Li, Zechu and Yang, Zhuoran and Zheng, Jiahao and Wang, Zhaoran and Walid, Anwar and Guo, Jian and Jordan, Michael I, ElegantRL-Podracer: Scalable and elastic
# library for cloud-native deep reinforcement learning, NeurIPS, Workshop on Deep Reinforcement Learning, 2021
#################################################################