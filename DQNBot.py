from time import sleep
import gym
import gym.spaces
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.policy import BoltzmannQPolicy
from rl.policy import GreedyQPolicy
from rl.policy import BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory
from keras.initializers import TruncatedNormal

import rl.callbacks
import matplotlib.pyplot as plt

import pandas as pd
import pandas.io.sql as psql
import sqlite3

# 直線上を動く点の速度を操作し、目標(原点)に移動させることを目標とする環境
class DQNBot(gym.core.Env):
    def __init__(self):
        self.BUY = 0
        self.SELL = 1
        self.STAY = 2
        self.action_space = gym.spaces.Discrete(2)

        self.con = sqlite3.connect('board_2.db')
        self.cur = self.con.cursor()
        
        df_low = psql.read_sql('SELECT 0, min(ask_price_100), min(ask_price_200), min(ask_price_300), min(ask_price_500), min(ask_price_800), min(ask_price_1300), min(ask_price_2100), min(ask_price_3400), min(ask_price_5500), min(ask_price_8900), \
                    min(bid_price_100), min(bid_price_200), min(bid_price_300), min(bid_price_500), min(bid_price_800), min(bid_price_1300), min(bid_price_2100), min(bid_price_3400), min(bid_price_5500), min(bid_price_8900) FROM boards;', self.con) # DBからPandasデータフレーム取得

        df_high = psql.read_sql('SELECT 1, max(ask_price_100), max(ask_price_200), max(ask_price_300), max(ask_price_500), max(ask_price_800), max(ask_price_1300), max(ask_price_2100), max(ask_price_3400), max(ask_price_5500), max(ask_price_8900), \
                    max(bid_price_100), max(bid_price_200), max(bid_price_300), max(bid_price_500), max(bid_price_800), max(bid_price_1300), max(bid_price_2100), max(bid_price_3400), max(bid_price_5500), max(bid_price_8900) FROM boards;', self.con) # DBからPandasデータフレーム取得
        
        board_df = psql.read_sql('SELECT ask_price_100, ask_price_200, ask_price_300, ask_price_500, ask_price_800, ask_price_1300, ask_price_2100, ask_price_3400, ask_price_5500, ask_price_8900, \
                                        bid_price_100, bid_price_200, bid_price_300, bid_price_500, bid_price_800, bid_price_1300, bid_price_2100, bid_price_3400, bid_price_5500, bid_price_8900 FROM boards;', self.con) # DBからPandasデータフレーム取得
        
        board_mid_df = psql.read_sql('SELECT mid_price FROM boards;', self.con) # DBからPandasデータフレーム取得

        low = df_low.values.flatten()
        high = df_high.values.flatten()
        self.observation_space = gym.spaces.Box(low=low, high=high)

        self.board_array = board_df.values
        self.board_array_rows = len(self.board_array)
        self.board_mid = board_mid_df.values

    def get_state(self, count):
        return self.board_array[count].flatten()
    
    def get_midprice_list(self):
        return self.board_mid.flatten()

    def get_midprice(self, count):
        return self.board_mid[count].flatten()[0]
    
    # 各stepごとに呼ばれる
    # actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装
    def step(self, action):
        self.step_count += 1
        done = self.board_array_rows - 11 < self.step_count
        
        reward = 0
        '''
        if self.pos[0] != self.STAY:
            if self.pos[0] == self.BUY:
                reward = self.get_midprice(self.step_count) - self.pos[1]
            else:
                reward = self.pos[1] - self.get_midprice(self.step_count)
        '''

        if action == self.BUY:
            if self.pos[0] == self.STAY:
                self.pos = [self.BUY, self.get_midprice(self.step_count)]
            elif self.pos[0] == self.SELL:
                reward = self.pos[1] - self.get_midprice(self.step_count)
                self.pos = [self.STAY, 0]
                self.hold_step = 0
            '''
            elif self.pos[0] == self.BUY:
                self.hold_step += 1
                if 10 < self.hold_step:
                    reward = self.get_midprice(self.step_count) - self.pos[1]
                    self.pos = [self.STAY, 0]
                    self.hold_step = 0
            '''
            
        elif action == self.SELL:
            if self.pos[0] == self.STAY:
                self.pos = [self.SELL, self.get_midprice(self.step_count)]
            elif self.pos[0] == self.BUY:
                reward = self.get_midprice(self.step_count) - self.pos[1]
                self.pos = [self.STAY, 0]
                self.hold_step = 0
            '''
            elif self.pos[0] == self.SELL:
                self.hold_step += 1
                if 10 < self.hold_step:
                    reward = self.pos[1] - self.get_midprice(self.step_count)
                    self.pos = [self.STAY, 0]
                    self.hold_step = 0
            '''
        '''    
        if done:
            if self.pos[0] == self.BUY:
                reward = self.get_midprice(self.step_count) - self.pos[1]
            elif self.pos[0] == self.SELL:
                reward = self.pos[1] - self.get_midprice(self.step_count)
        '''

        # 次のstate、reward、終了したかどうか、追加情報の順に返す
        # 追加情報は特にないので空dict
        return np.insert(self.get_state(self.step_count), 0, self.pos[0]), reward, done, {}

    # 各episodeの開始時に呼ばれ、初期stateを返すように実装
    def reset(self):
        self.pos = [self.STAY, 0]
        self.profit = 0
        self.step_count = 0
        self.hold_step = 0
        return np.insert(self.get_state(self.step_count), 0, self.pos[0])

class EpisodeLogger(rl.callbacks.Callback):
    def __init__(self):
        self.observations = {}
        self.rewards = {}
        self.actions = {}

    def on_episode_begin(self, episode, logs):
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])

if __name__ == "__main__":
    env = DQNBot()
    nb_actions = env.action_space.n

    # DQNのネットワーク定義
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    
    # model.add(Dense(50, activation='relu', kernel_initializer=TruncatedNormal(stddev=0.0001), bias_initializer='ones'))
    # model.add(Dense(25, activation='relu', kernel_initializer=TruncatedNormal(stddev=0.0001), bias_initializer='ones'))
    
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('relu'))
    
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # experience replay用のmemory
    memory = SequentialMemory(limit=2000000, window_length=1)
    # 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
    # policy = GreedyQPolicy()
    # policy = BoltzmannQPolicy()
    policy = EpsGreedyQPolicy(eps=0.4)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    history = dqn.fit(env, nb_steps=8000, visualize=False, verbose=2, nb_max_episode_steps=100)
    #学習の様子を描画したいときは、Envに_render()を実装して、visualize=True にします,

    cb_ep = EpisodeLogger()
    dqn.test(env, nb_episodes=1, visualize=False, callbacks=[cb_ep])

    pre = -1
    ac_list = []
    for ep_action in list(cb_ep.actions.values())[0]:
        if pre != ep_action:
            ac_list.append(ep_action)
            pre = ep_action
        else:
            ac_list.append(-1)

    '''
    count = 0
    rw = 0
    rw_list = []
    for ep_rw in list(cb_ep.rewards.values())[0]:
        if ac_list[count] != -1:
            rw = rw + ep_rw
        rw_list.append(rw)
        count += 1
    plt.plot(rw_list)
    print("PL: " + str(rw))
    '''

    rw = 0
    rw_list = []
    for ep_rw in list(cb_ep.rewards.values())[0]:
        rw = rw + ep_rw
        rw_list.append(rw)
    plt.plot(rw_list)
    print("PL: " + str(rw))

    count = 0
    for ac in ac_list:
        if ac == 0:
            if 0 < rw:
                plt.axvline(x=count, ymin=0, ymax=rw, color='r', linewidth=1)
            else:
                plt.axvline(x=count, ymin=rw, ymax=0, color='r', linewidth=1)
        elif ac == 1:
            if 0 < rw:
                plt.axvline(x=count, ymin=0, ymax=rw, color='b', linewidth=1)
            else:
                plt.axvline(x=count, ymin=rw, ymax=0, color='b', linewidth=1)
        count += 1
    print("BUY : " + str(ac_list.count(0)))
    print("SELL: " + str(ac_list.count(1)))
    
    plt.xlabel("step")
    plt.ylabel("rw")
    plt.show()
    
    # plt.plot(env.get_midprice_list())
    # plt.show()