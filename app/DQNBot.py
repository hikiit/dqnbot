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
from rl.memory import SequentialMemory
from keras.initializers import TruncatedNormal

import rl.callbacks
import matplotlib.pyplot as plt

import pandas as pd
import pandas.io.sql as psql
import sqlite3

class DQNBot(gym.core.Env):
    def __init__(self):
        # Agentにさせる行動はBUY/SELL/STAY(何もしない)
        self.BUY = 0
        self.SELL = 1
        self.STAY = 2
        self.action_space = gym.spaces.Discrete(3)

        self.con = sqlite3.connect(sys.argv[1])
        self.cur = self.con.cursor()
        
        # ask/bidの観測範囲
        # 0番目は BUY SELL STAYで0-2
        # それ以外は0-1で正規化する
        low_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        high_list = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        low = np.array(low_list)
        high = np.array(high_list)

        # DBからask/bidのpandasデータフレームを取得
        board_df = psql.read_sql('SELECT ask_price_100, ask_price_200, ask_price_300, ask_price_500, ask_price_800, ask_price_1300, ask_price_2100, ask_price_3400, ask_price_5500, ask_price_8900, \
                                        bid_price_100, bid_price_200, bid_price_300, bid_price_500, bid_price_800, bid_price_1300, bid_price_2100, bid_price_3400, bid_price_5500, bid_price_8900 FROM boards;', self.con) # DBからPandasデータフレーム取得
        
        # DBからmidpriceのpandasデータフレームを取得
        board_mid_df = psql.read_sql('SELECT mid_price FROM boards;', self.con)

        # 観測範囲を定義
        self.observation_space = gym.spaces.Box(low=low, high=high)

        # 0-1で正規化して配列取得
        self.board_array = preprocessing.minmax_scale(board_df.values, axis=1)
        
        # DBの長さを取得
        self.board_array_rows = len(self.board_array)

        # midpriceの配列を用意
        self.board_mid = board_mid_df.values


        self.step_count = 0
        
    # 指定stepのask/bidを取得
    def get_state(self, count):
        return self.board_array[count].flatten()
    
    # 指定stepのmidpriceを取得
    def get_midprice(self, count):
        return self.board_mid[count].flatten()[0]

    # 視覚化の時にだけ使用 本来は多分いらない
    def get_midprice_list(self):
        return self.board_mid.flatten()
    
    # 各stepごとに呼ばれる
    def step(self, action):
        # stepのカウントアップ
        self.step_count += 1

        # 学習データが終わりそうならdoneをTrueにしてstepを0に戻す
        done = self.board_array_rows - 20 < self.step_count
        if done:
            self.step_count = 0
        
        reward = 0
        if action == self.BUY:
            if self.pos[0] == self.STAY:
                self.pos = [self.BUY, self.get_midprice(self.step_count)]
            elif self.pos[0] == self.SELL:
                reward = self.pos[1] - self.get_midprice(self.step_count)
                self.pos = [self.STAY, 0]
                # 学習時は売買が成立した時点で区切る(=結果を学習させる)
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
        # ask/bidの情報 + ポジション情報を渡す
        # 追加情報は特にないので空dict
        return np.insert(self.get_state(self.step_count), 0, self.pos[0]), reward, done, {}

    # 各episodeの開始時に呼ばれる
    # 初期情報を渡す
    def reset(self):
        self.pos = [self.STAY, 0]
        self.profit = 0
        if sys.argv[2] == 'train':
            pass
            # randomにしてもいいかもしれないですね
            # self.step_count = random.randint(0, self.board_array_rows - 2000)
        else:
            self.step_count = 0
        return np.insert(self.get_state(self.step_count), 0, self.pos[0])

# 学習状況の保存
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
        # とりあえずオプションはデフォルト
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(512))
        model.add(Dense(512))
        model.add(Dense(nb_actions))
        print(model.summary())

        # experience replay用のmemory
        # 各ステップごと順番に学習させるわけではく、一度メモリに保存してからランダムに抽出と学習するとか
        # 正直、完全には理解できていません
        memory = SequentialMemory(limit=40000, window_length=1)

        # 行動方策はオーソドックスなepsilon-greedyです。
        policy = EpsGreedyQPolicy(eps=0.1)
        
        # warmup = 文字通り準備運動のイメージ いきなり学習させずにある程度メモリに貯めると思ってる
        # update = 学習率 小さくすると時間がかかるし、高くすると過学習しやすくなる
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                    target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=0.001))

        # nb_steps = 何ステップ学習させるか 数値をめちゃくちゃ大きくして、一晩経ったらCtrl+Cで止めるとかでも別にいい
        # max_episode_steps = 1エピソードの最大ステップ
        history = dqn.fit(env, nb_steps=400000, visualize=False, verbose=2, nb_max_episode_steps=1440)

        # modelとweightの保存    
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        dqn.save_weights('weight_' + str(now) + '.h5')
        model_json = model.to_json()
        with open('model_' + str(now) + '.json', "w") as json_file:
            json_file.write(model_json)
    
    elif sys.argv[2] == 'test':
        # modelのロード
        json_file = open(sys.argv[3], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        print(model.summary())

        # 学習後のテストをしたいだけなのに以下宣言が必要なのかは不明 一応同じようにdqnを設定していく
        memory = SequentialMemory(limit=2000000, window_length=1)
        policy = EpsGreedyQPolicy(eps=0.1)
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                    target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=0.001))

        # weighのロード
        dqn.load_weights(sys.argv[4])
    
        cb_ep = EpisodeLogger()

        # テストを実行
        # データベースで一通り売買してもらう
        # 時間がかかるので、consoleに状況を出すようにstepメソッド内で実装してもいいかも
        dqn.test(env, nb_episodes=1, visualize=False, callbacks=[cb_ep])

        # 結果の視覚化
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
            print("\rReward: " + str(len(rw_list)), end='')
        print("")

        plt.plot(rw_list)
        plt.xlabel("step")
        plt.ylabel("price")
        
        # dpiが低いと荒すぎる
        plt.savefig("figure.png",format = 'png', dpi=1200)
