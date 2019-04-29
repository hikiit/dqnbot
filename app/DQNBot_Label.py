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

class DQNBot(gym.core.Env):
    def __init__(self):
        self.term = 20
        self.margin = 10
        self.BUY = 0
        self.SELL = 1
        self.STAY = 2
        self.action_space = gym.spaces.Discrete(3)

        self.con = sqlite3.connect(sys.argv[2])
        self.cur = self.con.cursor()
        
        low_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        high_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        low = np.array(low_list)
        high = np.array(high_list)

        board_df = psql.read_sql('SELECT ask_price_100, ask_price_200, ask_price_300, ask_price_500, ask_price_800, ask_price_1300, ask_price_2100, ask_price_3400, ask_price_5500, ask_price_8900, \
                                        bid_price_100, bid_price_200, bid_price_300, bid_price_500, bid_price_800, bid_price_1300, bid_price_2100, bid_price_3400, bid_price_5500, bid_price_8900 FROM boards;', self.con) # DBからPandasデータフレーム取得
        
        board_mid_df = psql.read_sql('SELECT mid_price FROM boards;', self.con)

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
    def step(self, action):
        self.step_count += 1
        done = self.board_array_rows - 400 < self.step_count
                
        reward = 0
        if action == self.BUY:
            if sys.argv[1] == 'train':                
                pl = self.get_midprice(self.step_count + self.term) - self.get_midprice(self.step_count)
                if self.margin < pl:
                    reward = 1
                elif pl < 0:
                    reward = -1
                done = True
            else:
                reward = self.get_midprice(self.step_count + self.term) - self.get_midprice(self.step_count)
                # self.step_count += 60

        elif action == self.SELL:
            if sys.argv[1] == 'train':            
                pl = self.get_midprice(self.step_count) - self.get_midprice(self.step_count + self.term)
                if self.margin < pl:
                    reward = 1
                elif pl < 0:
                    reward = -1
                done = True
            else:
                reward = self.get_midprice(self.step_count) - self.get_midprice(self.step_count + self.term)
                # self.step_count += 60
        
        if sys.argv[1] == 'test':
            print('\r' + "step: " + str(self.step_count), end='')
            if done:
                print("")

        return self.get_state(self.step_count), reward, done, {}

    def reset(self):
        self.pos = [self.STAY, 0]
        self.profit = 0
        if sys.argv[1] == 'train':
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
    memory = SequentialMemory(limit=200000, window_length=1)
    warmup = 1000
    model_update = 1e-2
    policy = EpsGreedyQPolicy(eps=0.1)

    if sys.argv[1] == 'train':
        input_shape = (1,) + env.observation_space.shape
        dropout = 0.5

        # DQNのネットワーク定義
        model = Sequential()
        model.add(LSTM(units=512, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(LSTM(units=512, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(units=nb_actions))
        print(model.summary())
        
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=warmup,
                    target_model_update=model_update, policy=policy)
        dqn.compile(Adam(lr=0.001))

        tbcb = TensorBoard(log_dir='./graph', histogram_freq=0, write_grads=True)
    
        history = dqn.fit(env, nb_steps=5000000, verbose=2, nb_max_episode_steps=60, callbacks=[tbcb])

        now = datetime.now().strftime("%Y%m%d%H%M%S")
        dqn.save_weights('weight1_' + str(now) + '.h5')
        model_json = model.to_json()
        with open('model1_' + str(now) + '.json', "w") as json_file:
            json_file.write(model_json)
        with open("history.pickle", mode='wb') as f:
            pickle.dump(history.history, f)
    
    elif sys.argv[1] == 'test':
        json_file = open(sys.argv[3], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        print(model.summary())
        
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=warmup,
                    target_model_update=model_update, policy=policy)

        dqn.compile(Adam(lr=0.002))                    
        dqn.load_weights(sys.argv[4])
    
        cb_ep = EpisodeLogger()
        dqn.test(env, nb_episodes=1, visualize=False, callbacks=[cb_ep])

        print("COUNT BUY  : " + str(list(cb_ep.actions.values())[0].count(0)))
        print("COUNT SELL : " + str(list(cb_ep.actions.values())[0].count(1)))
        print("COUNT STAY : " + str(list(cb_ep.actions.values())[0].count(2)))

        plt.subplot(211)
        plt.plot(env.get_midprice_list(), linewidth=0.1)

        plt.subplot(212)
        rw_list = []
        reward = 0
        for ep_reward in list(cb_ep.rewards.values())[0]:
            reward += ep_reward
            rw_list.append(reward)
            print("\rStep: " + str(len(rw_list)), end='')
        print("")

        plt.plot(rw_list)
        plt.xlabel("step")
        plt.ylabel("price")
        plt.savefig("figure.png",format = 'png', dpi=1200)
