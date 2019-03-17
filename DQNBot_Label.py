import sys
from time import sleep
import random
from datetime import datetime
from sklearn import preprocessing
import gym
import gym.spaces
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, Dropout
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras.initializers import TruncatedNormal

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.policy import BoltzmannQPolicy
from rl.policy import GreedyQPolicy
from rl.policy import BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory

import rl.callbacks
import matplotlib.pyplot as plt

import pandas as pd
import pandas.io.sql as psql
import sqlite3

# 直線上を動く点の速度を操作し、目標(原点)に移動させることを目標とする環境
class DQNBot(gym.core.Env):
    def __init__(self):
        self.term = 60
        self.BUY = 0
        self.SELL = 1
        self.STAY = 2
        self.action_space = gym.spaces.Discrete(3)

        self.con = sqlite3.connect(sys.argv[1])
        self.cur = self.con.cursor()
        
        low_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        high_list = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        low = np.array(low_list)
        high = np.array(high_list)

        board_df = psql.read_sql('SELECT ask_price_100, ask_price_200, ask_price_300, ask_price_500, ask_price_800, ask_price_1300, ask_price_2100, ask_price_3400, ask_price_5500, ask_price_8900, \
                                        bid_price_100, bid_price_200, bid_price_300, bid_price_500, bid_price_800, bid_price_1300, bid_price_2100, bid_price_3400, bid_price_5500, bid_price_8900 FROM boards;', self.con) # DBからPandasデータフレーム取得
        
        board_mid_df = psql.read_sql('SELECT mid_price FROM boards;', self.con) # DBからPandasデータフレーム取得

        self.observation_space = gym.spaces.Box(low=low, high=high)

        self.board_array = preprocessing.minmax_scale(board_df.values, axis=1)
        self.board_array_rows = len(self.board_array)
        self.board_mid = board_mid_df.values
        self.step_count = 0
        
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
        done = self.board_array_rows - 400 < self.step_count
                
        reward = 0
        if action == self.BUY:
            if sys.argv[2] == 'train':
                pl = self.get_midprice(self.step_count + self.term) - self.get_midprice(self.step_count)
                if 100 < pl:
                    reward = 1
                elif pl < 0:
                    reward = -1
                else:
                    reward = 0
                done = True
            else:
                reward = self.get_midprice(self.step_count + self.term) - self.get_midprice(self.step_count)
                # self.step_count += 60

        elif action == self.SELL:
            if sys.argv[2] == 'train':
                pl = self.get_midprice(self.step_count) - self.get_midprice(self.step_count + self.term)
                if 100 < pl:
                    reward = 1
                elif pl < 0:
                    reward = -1
                else:
                    reward = 0
                done = True
            else:
                reward = self.get_midprice(self.step_count) - self.get_midprice(self.step_count + self.term)
                # self.step_count += 60

        # 次のstate、reward、終了したかどうか、追加情報の順に返す
        # 追加情報は特にないので空dict
        return self.get_state(self.step_count), reward, done, {}

    # 各episodeの開始時に呼ばれ、初期stateを返すように実装
    def reset(self):
        self.pos = [self.STAY, 0]
        self.profit = 0
        if sys.argv[2] == 'train':
            self.step_count = random.randint(0, self.board_array_rows - 2000)
        else:
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

    if sys.argv[2] == 'train':
        input_shape = (1,) + env.observation_space.shape
        dropout = 0

        # DQNのネットワーク定義
        model = Sequential()
        model.add(LSTM(units=512, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(LSTM(units=512, return_sequences=False))
        model.add(Dense(units=nb_actions))
        print(model.summary())

        # experience replay用のmemory
        memory = SequentialMemory(limit=5000000, window_length=1)
        # 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
        # policy = GreedyQPolicy()
        # policy = BoltzmannQPolicy()
        policy = EpsGreedyQPolicy(eps=0.1)
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                    target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=0.001))

        tbcb = TensorBoard(log_dir='./graph', histogram_freq=0, write_grads=True)
    
        history = dqn.fit(env, nb_steps=100000, verbose=2, nb_max_episode_steps=1440, callbacks=[tbcb])

        now = datetime.now().strftime("%Y%m%d%H%M%S")
        dqn.save_weights('weight_' + str(now) + '.h5')
        model_json = model.to_json()
        with open('model_' + str(now) + '.json', "w") as json_file:
            json_file.write(model_json)
        with open("history.pickle", mode='wb') as f:
            pickle.dump(history.history, f)
    
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

        print("COUNT BUY  : " + str(list(cb_ep.actions.values())[0].count(0)))
        print("COUNT SELL : " + str(list(cb_ep.actions.values())[0].count(1)))
        print("COUNT STAY : " + str(list(cb_ep.actions.values())[0].count(2)))

        plt.subplot(211)
        plt.plot(env.get_midprice_list(), linewidth=0.1)

        for i in range(len(list(cb_ep.actions.values())[0])):
            if list(cb_ep.actions.values())[0][i] == 0:
                plt.scatter(i, env.get_midprice_list()[i], s=0.1, marker="2", c='red')
                plt.scatter(i+60, env.get_midprice_list()[i+60], s=0.1, marker="x", c='red')
            elif list(cb_ep.actions.values())[0][i] == 1:
                plt.scatter(i, env.get_midprice_list()[i], s=0.1, marker="1", c='blue')
                plt.scatter(i+60, env.get_midprice_list()[i+60], s=0.1, marker="x", c='blue')

        plt.subplot(212)
        rw_list = []
        reward = 0
        for ep_reward in list(cb_ep.rewards.values())[0]:
            reward += ep_reward
            rw_list.append(reward)
        plt.plot(rw_list)
        plt.xlabel("step")
        plt.ylabel("price")
        plt.savefig("figure.png",format = 'png', dpi=1200)
        
