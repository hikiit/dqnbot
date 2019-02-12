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
from rl.memory import SequentialMemory

import rl.callbacks
import matplotlib.pyplot as plt

import pandas as pd
import pandas.io.sql as psql
import sqlite3

# 直線上を動く点の速度を操作し、目標(原点)に移動させることを目標とする環境
class DQNBot(gym.core.Env):
    def __init__(self):
        # self.STAY = 0
        self.BUY = 0
        self.SELL = 1
        self.STAY = -1
        self.action_space = gym.spaces.Discrete(2)

        self.con = sqlite3.connect('board_second.db')
        self.cur = self.con.cursor()
        
        df_low = psql.read_sql('SELECT min(mid_price), min(ask_price_100), min(ask_price_200), min(ask_price_300), min(ask_price_500), min(ask_price_800), min(ask_price_1300), min(ask_price_2100), min(ask_price_3400), min(ask_price_5500), min(ask_price_8900), \
                    min(bid_price_100), min(bid_price_200), min(bid_price_300), min(bid_price_500), min(bid_price_800), min(bid_price_1300), min(bid_price_2100), min(bid_price_3400), min(bid_price_5500), min(bid_price_8900) FROM boards;', self.con) # DBからPandasデータフレーム取得

        df_high = psql.read_sql('SELECT max(mid_price), max(ask_price_100), max(ask_price_200), max(ask_price_300), max(ask_price_500), max(ask_price_800), max(ask_price_1300), max(ask_price_2100), max(ask_price_3400), max(ask_price_5500), max(ask_price_8900), \
                    max(bid_price_100), max(bid_price_200), max(bid_price_300), max(bid_price_500), max(bid_price_800), max(bid_price_1300), max(bid_price_2100), max(bid_price_3400), max(bid_price_5500), max(bid_price_8900) FROM boards;', self.con) # DBからPandasデータフレーム取得

        low = df_low.values.flatten()
        high = df_high.values.flatten()

        self.observation_space = gym.spaces.Box(low=low, high=high)

        board_df = psql.read_sql('SELECT mid_price, ask_price_100, ask_price_200, ask_price_300, ask_price_500, ask_price_800, ask_price_1300, ask_price_2100, ask_price_3400, ask_price_5500, ask_price_8900, \
                                        bid_price_100, bid_price_200, bid_price_300, bid_price_500, bid_price_800, bid_price_1300, bid_price_2100, bid_price_3400, bid_price_5500, bid_price_8900 FROM boards;', self.con) # DBからPandasデータフレーム取得

        self.board_array = board_df.values
        self.board_array_rows = len(self.board_array)

    def get_state(self, count):
        return self.board_array[count].flatten()

    # 各stepごとに呼ばれる
    # actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装
    def step(self, action):
        # actionを受け取り、次のstateを決定
        self.step_count += 1
        done = self.board_array_rows - 10 < self.step_count
        
        reward = 0
        if action == self.BUY:
            if self.pos[0] == self.STAY:
                self.pos = [self.BUY, self.get_state(self.step_count)[0]]
            elif self.pos[0] == self.BUY:
                reward = 0 # self.get_state(self.step_count-1)[0] - self.get_state(self.step_count)[0]
            elif self.pos[0] == self.SELL:
                reward = self.pos[1] - self.get_state(self.step_count)[0]
                self.pos = [self.STAY, 0]
            
        elif action == self.SELL:
            if self.pos[0] == self.STAY:
                self.pos = [self.SELL, self.get_state(self.step_count)[0]]
            elif self.pos[0] == self.BUY:
                reward = self.get_state(self.step_count)[0] - self.pos[1]
                self.pos = [self.STAY, 0]
            elif self.pos[0] == self.SELL:
                reward = 0 # self.get_state(self.step_count)[0] - self.get_state(self.step_count-1)[0]

        if done:
            if self.pos[0] == self.BUY:
                reward = self.get_state(self.step_count)[0] - self.pos[1]
            elif self.pos[0] == self.SELL:
                reward = self.pos[1] - self.get_state(self.step_count)[0]

        # 次のstate、reward、終了したかどうか、追加情報の順に返す
        # 追加情報は特にないので空dict
        return self.get_state(self.step_count), reward, done, {}

    # 各episodeの開始時に呼ばれ、初期stateを返すように実装
    def reset(self):
        # 初期stateは、位置はランダム、速度ゼロ
        self.pos = [self.STAY, 0]
        self.profit = 0
        self.step_count = 0
        return self.get_state(self.step_count)

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
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # experience replay用のmemory
    memory = SequentialMemory(limit=10000, window_length=1)
    # 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
    # policy = BoltzmannQPolicy()
    policy =  EpsGreedyQPolicy(eps=0.1) 
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    history = dqn.fit(env, nb_steps=10000, visualize=False, verbose=2, nb_max_episode_steps=300)
    #学習の様子を描画したいときは、Envに_render()を実装して、visualize=True にします,

    cb_ep = EpisodeLogger()
    dqn.test(env, nb_episodes=1, visualize=False, callbacks=[cb_ep])

    sleep(2)

    '''
    count = 0
    rw = 0
    rw_list = []
    for obs in cb_ep.rewards.values():
        for o in obs:
            rw = rw + o
            rw_list.append(rw)
            print(str(count) + ": " + str(rw))
    
    plt.plot(rw, label="test")
            
    plt.xlabel("step")
    plt.ylabel("rewards")
    plt.show()

    '''
    rw = 0
    for obs in cb_ep.rewards.values():
        plt.plot([o for o in obs], '.')
    plt.xlabel("step")
    plt.ylabel("pos")
    plt.show()
