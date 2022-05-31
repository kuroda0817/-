# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import preprocessing
import pickle



# 理想のQを手で作成
def make_risouQ(Q_name, state_name, action_name):
    green, yellow_green, cream, orange = 3, 2, 1, 0

    state_size = len(state_name)      # row
    action_size = len(action_name)    # col
    reward_map = np.zeros((state_size, action_size))

    Q = {}
    for s in range(state_size):   #state_size
        if s in range(0,6):
            Q[s] = [orange] * 2 + [green] * 1 + [orange] * 4 + [yellow_green] * 1
        if s in range(6,12):
            Q[s] = [cream] * 2 + [yellow_green] * 1 + [cream] * 4 + [yellow_green] * 1
        if s in range(12,15):
            Q[s] = [yellow_green] * 3 + [cream] * 4 + [yellow_green] * 1
        if s in range(15,18):
            Q[s] = [green] * 2 + [yellow_green] * 1 + [cream] * 4 + [yellow_green] * 1
        if s in range(18,21):
            Q[s] = [yellow_green] * 3 + [cream] * 1 + [yellow_green] * 4
        if s in range(21,24):
            Q[s] = [yellow_green] * 3 + [green] * 1 + [yellow_green] * 4

    for k, v in Q.items():
        print(k, v)

    with open(Q_name, mode='wb') as f:
        pickle.dump(Q, f) 



# 定義したstateのうち，学習されたものを表示
def show_q_value(Q, state_name, action_name, Q_name, draw_trained_state=False):

    state_size = len(state_name)      # row
    action_size = len(action_name)    # col
    reward_map = np.zeros((state_size, action_size))

    # 辞書を2次元リストに変換
    for s in range(state_size):   #state_size
        for a in range(action_size):   #action_size
            if s in Q.keys():
                reward_map[s][a] = Q[s][a]

    if draw_trained_state:
        # 学習されていない箇所を削除
        state_name = np.array(state_name)[list(sorted(Q.keys()))]
        reward_map = np.array(reward_map)[list(sorted(Q.keys()))]
        state_size = len(state_name)      # 振り直し

    # 正規化
    reward_map_normed = preprocessing.minmax_scale(reward_map, axis=1)

    fig = plt.figure()
    plt.subplots_adjust(wspace=0.7, hspace=0.7)
    #(suggested defaults : wspace = 0.2, hspace = 0.2)
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.2, top=0.95)
    #(suggested defaults : left = 0.125, right = 0.9, bottom = 0.1, top = 0.9)

    for i, (r_map, title) in enumerate(zip([reward_map, reward_map_normed], ['reward_map', 'reward_map_normed'])):
        ax = fig.add_subplot(1, 2, i+1)
        plt.imshow(r_map, cmap=cm.RdYlGn, interpolation="bilinear",
                   vmax=abs(r_map).max(), vmin=-abs(r_map).max())
        # 表示する値の範囲
        ax.set_xlim(-0.5, action_size - 0.5)
        ax.set_ylim(-0.5, state_size - 0.5)
        # 表示するメモリの値
        ax.set_xticks(np.arange(action_size))
        ax.set_yticks(np.arange(state_size))
        # メモリにラベルを貼る
        ax.set_xticklabels(action_name, rotation=-90, fontsize=8)
        ax.set_yticklabels(state_name, fontsize=8)
        # 軸のラベル
        ax.set_xlabel('action')
        ax.set_ylabel('state')
        ax.set_title(title)
        ax.grid(which="both")

    #plt.show()
    plt.savefig(Q_name)















