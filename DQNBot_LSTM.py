import sys
from time import sleep
import random
from datetime import datetime
from sklearn import preprocessing
import gym
import gym.spaces
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM
from keras.optimizers import Adam
from keras.models import model_from_json

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
        self.action_space = gym.spaces.Discrete(3)

        self.con = sqlite3.connect(sys.argv[1])
        self.cur = self.con.cursor()
        
        low_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        high_list = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        low = np.array(low_list)
        high = np.array(high_list)

        board_df = psql.read_sql('SELECT ask_price_100, ask_price_200, ask_price_300, ask_price_500, ask_price_800, ask_price_1300, ask_price_2100, ask_price_3400, ask_price_5500, ask_price_8900, \
                                        bid_price_100, bid_price_200, bid_price_300, bid_price_500, bid_price_800, bid_price_1300, bid_price_2100, bid_price_3400, bid_price_5500, bid_price_8900 FROM boards;', self.con) # DBからPandasデータフレーム取得
        
        board_mid_df = psql.read_sql('SELECT mid_price FROM boards;', self.con) # DBからPandasデータフレーム取得

        self.observation_space = gym.spaces.Box(low=low, high=high)

        self.board_array = preprocessing.minmax_scale(board_df.values, axis=1)
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
        done = self.board_array_rows - 20 < self.step_count
        
        reward = 0
        if action == self.BUY:
            if self.pos[0] == self.STAY:
                self.pos = [self.BUY, self.get_midprice(self.step_count)]
            elif self.pos[0] == self.SELL:
                reward = self.pos[1] - self.get_midprice(self.step_count)
                self.pos = [self.STAY, 0]
                if sys.argv[2] == 'train':
                    done = True

        elif action == self.SELL:
            if self.pos[0] == self.STAY:
                self.pos = [self.SELL, self.get_midprice(self.step_count)]
            elif self.pos[0] == self.BUY:
                reward = self.get_midprice(self.step_count) - self.pos[1]
                self.pos = [self.STAY, 0]
                if sys.argv[2] == 'train':
                    done = True

        # 次のstate、reward、終了したかどうか、追加情報の順に返す
        # 追加情報は特にないので空dict
        return np.insert(self.get_state(self.step_count), 0, self.pos[0]), reward, done, {}

    # 各episodeの開始時に呼ばれ、初期stateを返すように実装
    def reset(self):
        self.pos = [self.STAY, 0]
        self.profit = 0
        if sys.argv[2] == 'train':
            self.step_count = random.randint(0, self.board_array_rows - 2000)
        else:
            self.step_count = 0
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

    if sys.argv[2] == 'train':
        input_shape = (1,) + env.observation_space.shape

        # DQNのネットワーク定義
        model = Sequential()
        model.add(LSTM(units=16, return_sequences=False, input_shape=input_shape))
        # model.add(LSTM(units=16, return_sequences=False))
        model.add(Dense(units=nb_actions))
        print(model.summary())

        # experience replay用のmemory
        memory = SequentialMemory(limit=2000000, window_length=1)
        # 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
        # policy = GreedyQPolicy()
        # policy = BoltzmannQPolicy()
        policy = EpsGreedyQPolicy(eps=0.1)
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                    target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=0.002))
    
        history = dqn.fit(env, nb_steps=1800, visualize=False, verbose=2, nb_max_episode_steps=1440)

        now = datetime.now().strftime("%Y%m%d%H%M%S")
        dqn.save_weights('weight_' + str(now) + '.h5')
        model_json = model.to_json()
        with open('model_' + str(now) + '.json', "w") as json_file:
            json_file.write(model_json)
    
    elif sys.argv[2] == 'test':
        json_file = open(sys.argv[3], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        print(model.summary())

        memory = SequentialMemory(limit=2000000, window_length=1)
        policy = EpsGreedyQPolicy(eps=0.1)
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                    target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=0.002))
        dqn.load_weights(sys.argv[4])
    
        cb_ep = EpisodeLogger()
        dqn.test(env, nb_episodes=1, visualize=False, callbacks=[cb_ep])

        pre = 2
        ac_list = []
        for ep_action in list(cb_ep.actions.values())[0]:
            if pre != ep_action and ep_action != 2:
                ac_list.append(ep_action)
                pre = ep_action
            else:
                ac_list.append(2)
        print("BUY : " + str(ac_list.count(0)))
        print("SELL: " + str(ac_list.count(1)))
        count = 0
        for ac in ac_list:
            if ac == 0:
                plt.axvline(x=count, ymin=0, ymax=400000, color='r', linewidth=1)
            elif ac == 1:
                plt.axvline(x=count, ymin=0, ymax=400000, color='b', linewidth=1)
            count += 1
        plt.plot(env.get_midprice_list())
        plt.xlabel("step")
        plt.ylabel("price")
        plt.savefig('figure.png')
