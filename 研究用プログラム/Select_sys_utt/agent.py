# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import pickle
import os
import itertools
from sklearn import preprocessing
from virtual_user import UserModel
from collections import defaultdict
from theme import HistoryTheme
from params import params
import copy
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import random
import scipy
import statistics
from asari.api import Sonar
import time
class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.l1=L.Linear(obs_size, n_hidden_channels)
            self.l2=L.Linear(n_hidden_channels, n_hidden_channels)
            self.l3=L.Linear(n_hidden_channels, n_actions)
    def __call__(self, x, test=False):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))


        y = chainerrl.action_value.DiscreteActionValue(self.l3(h2))

        return y





class Agent(params):
    def __init__(self, epsilon):
        super().__init__()
        self.Q = {}
        self.Q2 = {}
        self.epsilon = epsilon
        self.reward_log = []
        self.dialogue_log = []
        self.max_n_exchg = 10
        #self.df2 = pd.read_csv(self.get('path_dialogue_Q'))
        self.decidedtheme = 'sports'
        self.df = pd.read_csv(self.get('path_utterance_by_class_named'))
    # epsilon以下でランダムな行動，それ以外はQに従った行動
    # softmax=Trueで確率的に選択するようにできます
    def policy(self, env, s, actions, theme, selection='argmax'):

        if selection == 'argmax':
            if theme == 'スポーツ':
                Q2_theme = copy.copy(env.sports_key)
            if theme == '音楽':
                Q2_theme = copy.copy(env.music_key)
            if theme == '食事':
                Q2_theme = copy.copy(env.eating_key)
            if theme == '旅行':
                Q2_theme = copy.copy(env.travel_key)
            t = 0

            actions_Q2_theme = copy.copy(Q2_theme)
            for i in range(10):
                actions_Q2_theme.append(14)

            if np.random.random() < self.epsilon:
                t = 1
                random_theme = int(np.random.choice(actions_Q2_theme,size=1))
                return random_theme
            if t == 0:
                Q2prob_theme=[self.Q2[s][int(i)] for i in actions_Q2_theme]

                if s in self.Q2 and sum(Q2prob_theme) != 0:

                    return int(actions_Q2_theme[np.argmax(Q2prob_theme)])
                else:
                    return int(np.random.choice(actions_Q2_theme,size=1))

        elif selection == 'softmax':

            if theme == 'スポーツ':
                Q2_theme = copy.copy(env.sports_key)
            if theme == '音楽':
                Q2_theme = copy.copy(env.music_key)
            if theme == '食事':
                Q2_theme = copy.copy(env.eating_key)
            if theme == '旅行':
                Q2_theme = copy.copy(env.travel_key)
            t = 0
            #actions_Q2_theme = []
            actions_Q2_theme = copy.copy(Q2_theme)
            
            for i in range(20):
                actions_Q2_theme.append(14)
                return np.random.randint(len(actions))

            if np.random.random() < self.epsilon:
                t = 1
                random_theme = int(np.random.choice(actions_Q2_theme,size=1))
                return random_theme
            if t == 0:
                Q2prob_theme=[self.Q2[s][int(i)] for i in Q2_theme]
                if s in self.Q and sum(Q2prob_theme) != 0:
                    Q2prob_theme=[self.Q2[s][int(i)] for i in Q2_theme]
                    #return np.argmax(self.softmax(preprocessing.minmax_scale(self.Q2[s])))
                    ret = np.random.choice(self.softmax(preprocessing.minmax_scale(Q2prob_theme)))
                    return ret
                else:
                    return int(np.random.choice(actions_Q2_theme,size=1))
        else:
            print('invalid "selection"')
            exit(0)


    def init_log(self):
        self.reward_log = []
        self.dialogue_log = []

    def append_log_reward(self, reward):
        self.reward_log.append(reward)

    def append_log_dialogue(self, exchgID, state, action, theme, impression, s_utte, u_utte):
        self.dialogue_log.append([exchgID+'_S', state, action, theme, '-', s_utte])
        self.dialogue_log.append([exchgID+'_U', '-', '-', '-', impression, u_utte])

    def show_reward_log(self, interval=50, episode=-1, filename='sample.png'):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(
                   episode, mean, std))
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds,
                             alpha=0.1, color="g")
            plt.plot(indices, means, "o-", color="g",
                     label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.savefig(filename)
            #plt.show()

    def write_dialogue_log(self, filename):

        # ファイルが既に存在する場合，代わりの名前を振ってあげる．
        def search_and_rename_filename(oldpath):
            if os.path.exists(oldpath):
                print('file "{}" already exists.'.format(oldpath))
                #dirpath:ディレクトリのパス, filename:対象のファイルまたはディレクトリ
                #name:対象のファイルまたはディレクトリ（拡張子なし）, ext:拡張子
                dirpath, filename = os.path.split(oldpath)
                name, ext = os.path.splitext(filename)

                for i in itertools.count(1):
                    newname = '{}_{}{}'.format(name, i, ext)
                    newpath = os.path.join(dirpath, newname)
                    if not os.path.exists(newpath):
                        return newpath
                    else:
                        print('file "{}" already exists.'.format(newpath))
            else:
                return oldpath
                
        df = pd.DataFrame(data=self.dialogue_log, columns=['exchgID', 'state', 'action', 'theme', 'UI', 'utterance'])
        filename_new = search_and_rename_filename(filename)
        df.to_csv(filename_new, index=None)
        print('finished making file "{}".'.format(filename_new))

    def saveR(self, filename):
        np.save(filename, np.array(self.reward_log))

    def saveQ(self, table, filename):
        with open(filename, mode='wb') as f:
            pickle.dump(dict(table), f)
    # ソフトマックス関数
    # coefは推定値の振れ幅を調整するためのもの．（デフォルトは1）
    def softmax(self, a, coef=1):
        c = np.max(a)
        exp_a = np.exp(coef * (a - c))
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y


# システム側
class QlearningAgent(Agent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)
        self.df = pd.read_csv(self.get('path_utterance_by_class_named'))
        self.r_cons_history = []
    # rewardを定義
    
    def get_reward(self, env, usermodel, state, n_state, action_name, s2, n_state2, action2):
        # state(id)からstate_name(str)をreturn
        
        def get_state_name(d, state, index):
            values = [k for k, v in d.items() if v == state]
            if index == None:
                return values[0]
            else:
                return values[0].split('_')[index]

        # 現在の心象に応じて決定
        if get_state_name(env.stateIndex, state, index=2) == 'h':
            r_UI = float(self.get('R_oneUI'))
        elif get_state_name(env.stateIndex, state, index=2) == 'n':
            r_UI = 0
        elif get_state_name(env.stateIndex, state, index=2) == 'l':
            r_UI = -float(self.get('R_oneUI'))

        # N連続で心象を見る
        r_PUI = 0
        if len(usermodel.log_UI_1theme) >= env.persist_UI_exchgs:
            persist_UI = np.array(usermodel.log_UI_1theme[-env.persist_UI_exchgs:])
            if np.count_nonzero(persist_UI >= env.thres_high_UI) == env.persist_UI_exchgs:
                r_PUI = float(self.get('R_persistUI'))
            elif np.count_nonzero(persist_UI <= env.thres_low_UI) == env.persist_UI_exchgs:
                r_PUI = -float(self.get('R_persistUI'))

        # 対話行為のbigramで報酬
        da1 = get_state_name(env.stateIndex, state, index=0)
        da2 = get_state_name(env.stateIndex, n_state, index=0)
        if da2 == 'ct':
            r_DA = 0
        else:
            r_DA = env.reward_da_df.loc[da1, da2]

        # 固有名詞に適切に反応できたら正の報酬
        properNoun_response = ['qs_o_d','qs_o_s','re_o_m']
        noun_presence = get_state_name(env.stateIndex, state, index=1)
        if action_name in properNoun_response:
            if noun_presence == 'No':
                r_NOUN = float(self.get('R_noun'))
            elif noun_presence == 'Nx':
                r_NOUN = -float(self.get('R_noun'))
        else:
            r_NOUN = 0

        #ユーザ発話がネガティブなときに深堀したら負の報酬
        if get_state_name(env.stateIndex,state,index=3)=='ne':
            if action_name=="qs_o_s" or action_name=="qs_o_d" or action_name == "re_o_m" or action_name=="thank" or action_name=="io":
                r_nega=-10
                r_NOUN = 0
            else:
                r_nega=0
        else:
            r_nega=0
        # 感謝に対しては取り締まる
        if (action_name == 'thank') and (action_name != 'change_theme'):
            r_TNK = float(self.get('R_thank'))
        else:
            r_TNK = 0
            
        #blacklist報酬
        if len(s2.split('_')) > 9:
            pre_state_da = s2.split("_")[0:4]
            pre_state_da = '_'.join(pre_state_da)
        else:
            if s2 == 'ct_0':
                pre_state_da = s2
            if 'thank' in s2:
                pre_state_da = s2.split('_')[0:2]
                pre_state_da = '_'.join(pre_state_da)
            else:
                pre_state_da = s2.split('_')[0:4]
                pre_state_da = '_'.join(pre_state_da)

        prelist_da = self.df[self.df['cls_ID_action'] == action2]
        r_cons = 0
        prelist_da_b0 = prelist_da['precanb_0']
        prelist_da_w0 = prelist_da['precan_0']
        
        if not prelist_da_b0.isnull().any():
            prelist_da_b = prelist_da.loc[:, prelist_da.columns.str.startswith('precanb_')]
            prelist_da_b = prelist_da_b.values.tolist()
            prelist_da_b = list(map(lambda s: str(s).strip('*'), prelist_da_b[0]))
            if pre_state_da in prelist_da_b:
                r_cons += -20

        if not prelist_da_w0.isnull().any():
            prelist_da_w = prelist_da.loc[:, prelist_da.columns.str.startswith('precan_')]
            prelist_da_w = prelist_da_w.values.tolist()
            prelist_da_w = list(map(lambda x: str(x).strip('*'),prelist_da_w[0]))
            if not (pre_state_da in prelist_da_w):
                r_cons += -20
        
        # 報酬の総和
        reward = r_cons + r_UI+ r_PUI+ float(self.get('Rc_bigram')) * r_DA + r_NOUN + r_TNK+r_nega


        #テスト用
        # print('unigram:',r_cons)
        # print('ネガティブ',r_nega)
        # print('心象:',r_UI)
        # print('心象連続:',r_PUI)
        # print('簡易対話行為:',float(self.get('Rc_bigram'))*r_DA)
        # print('特定名詞:',r_NOUN)
        # print('感謝:',r_TNK)
        # print('報酬:',reward)
        # print('-------------------------------------------')
        
        return reward

    def negative(self,user_utterance,sonar):#ユーザ発話がネガティブかどうか判定
        #sonar=Sonar()
        nega = sonar.ping(text=user_utterance)
        if nega['top_class'] == 'negative':
            return "ne"
        else:
            return "po"

    def random_action(self):

        kodosu = list(np.arange(0,self.action_length,1))
        ran = np.random.choice(kodosu)
        #if ran == 33 or ran ==34:

        return ran

    #DQNで学習
    def learn(self, env, episode_count=179000, gamma=0.5,
              learning_rate=0.1, report_interval=10, coef_epsilon=0.99995, name=0, reward_seiki=0, m_node=0, bigramhosyu=0, learning_theme=0):
              #DQNの構成
        seiki = list(np.arange(-56,21.1,0.5))
        sonar=Sonar()
        seiki_zscore = scipy.stats.zscore(seiki)
        seiki_norm_plus = preprocessing.minmax_scale(seiki)
        self.action_length=len(env.actions2)
        gamma = 0.5
        alpha = 0.1
        max_number_of_steps = episode_count  #1試行のstep数
        num_episodes = episode_count  #総試行回数


        q_func = QFunction(len(env.states2_ID)+len(env.states_noun_presence)+len(env.states_negative)+1, len(env.actions2), m_node) #(状態, 行動, 中間層)
        q_func.to_gpu(0)
        optimizer = chainer.optimizers.Adam(eps=1e-2)
        optimizer.setup(q_func)
        print(len(env.actions2))
        explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0, end_epsilon=0.1, decay_steps=num_episodes*2.5, random_action_func=self.random_action)
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
        phi = lambda x: x.astype(np.float32, copy=False)
        agent_DQN = chainerrl.agents.DQN(
            q_func, optimizer, replay_buffer, gamma, explorer,
            replay_start_size=500, update_interval=1, target_update_interval=100, phi=phi)

        self.init_log()
        actions = list(env.actionIndex.keys())
        actions2 = list(env.actionIndex2.keys())
        self.Q2 = defaultdict(lambda: [0] * len(actions2))

        ep = self.epsilon

        for e in range(episode_count):#メインループ
            self.epsilon = max(coef_epsilon ** e, ep)

            done = True
            s = env.reset()#reset
            s2 = "qs_x_s_0_qs_x_s_0_No_l_ne"
            themeHis = HistoryTheme(random_choice=True)#reset
            usermodel = UserModel()#reset
            s_dim = list(np.arange(0,len(env.states2_ID)+len(env.states_noun_presence)+len(env.states_negative)+1,1))
            s_dim = np.array(s_dim)
            rewards = []
            
            state_history = []
            state2_history = []
            sys_utterance_history = []
            ENtoJP={"sports":"スポーツ", "music":"音楽", "eating":"食事", "travel":"旅行"}
            for n_exchg in range(self.max_n_exchg):


                if n_exchg == 0:
                    #chg_theme, theme = themeHis.decideNextTheme(None,False)
                    chg_theme = True
                    theme = ENtoJP[learning_theme]
                else:
                    #chg_theme, theme = themeHis.decideNextTheme(impression)
                    chg_theme = False
                    theme = ENtoJP[learning_theme]
                # システム発話決定

                if chg_theme:
                    a_name = 'change_theme'

                    a_name_dialogue = 'change_theme'
                else:
                    a = agent_DQN.act_and_train(s_dim, reward)

                    a_name = env.actionIndex2[a]

                    a_name_dialogue = a_name.split('_')
                    del a_name_dialogue[-1]
                    a_name_dialogue ='_'.join(a_name_dialogue)


                sys_utterance = env.utterance_selection(a_name, theme)# 発話選択
                sys_utterance_history.append(sys_utterance)

                
                user_utterance, impression = usermodel.getResponse(sys_utterance)# 応答選択
                negative=self.negative(user_utterance,sonar)
                #print(user_utterance)
                n_state = env.get_next_state(impression, sys_utterance, user_utterance,negative)# 次のstateを決める
                n_state_name = [k for k, v in env.stateIndex.items() if v == n_state][0]

                if a_name == 'change_theme':
                    if "スポーツ" in sys_utterance:
                        n_state2 = 'ct_0'
                    if "音楽" in sys_utterance:
                        n_state2 = 'ct_0'
                    if "食事" in sys_utterance:
                        n_state2 = 'ct_0'
                    if "旅行" in sys_utterance:
                        n_state2 = 'ct_0'

                else:
                    hozon = 0

                    
                    if a_name_dialogue == 'qs_o_d':
                        for i in state2_history:
                            if len(i.split('_'))<9 and ('qs_o_s' in i or 'qs_x_s' in i):
                                hozon = i.split('_')[0:4]
                                hozon = '_'.join(hozon)
                    
                    if hozon != 0:
                        n_state2 = hozon + '_' + env.get_next_state2(impression, sys_utterance, user_utterance,negative)
                    else:
                        n_state2 = env.get_next_state2(impression, sys_utterance, user_utterance,negative)


                #簡易対話行為
                if a_name == 'change_theme':
                    n_state_states2_ID = env.stateIndex_states2_ID['ct_0']

                    n_state_states2_ID_onehot = list(np.zeros(len(env.stateIndex_states2_ID)))

                    n_state_states2_ID_onehot[n_state_states2_ID] = 1
                    n_state_noun_presence_onehot = [1,0]
                    
                    n_state_impression = 0.5
                    n_state_negative_onehot = [1,0]
                    
                else: 
                    n_state_states2_ID_name = n_state2.split('_')[0:-3]

                    n_state_states2_ID_name = '_'.join(n_state_states2_ID_name)
                    n_state_states2_ID = env.stateIndex_states2_ID[n_state_states2_ID_name]
                    #n_state_states2_ID_his1 = env.stateIndex_states2_ID_his1[n_state2_name_his1]

                    n_state_states2_ID_onehot = list(np.zeros(len(env.stateIndex_states2_ID)))
                    n_state_states2_ID_onehot[n_state_states2_ID] = 1

                    #特定名詞の有無
                    n_state_noun_presence_name = n_state_name.split('_')[-3]
                    n_state_noun_presence = env.stateIndex_noun_presence[n_state_noun_presence_name]
                    n_state_noun_presence_onehot = [0,0]
                    n_state_noun_presence_onehot[n_state_noun_presence] = 1
                    #心象
                    n_state_impression_name = n_state_name.split('_')[-2]
                    n_state_impression = env.stateIndex_impression[n_state_impression_name]
                    #ポジネガ
                    n_state_negative_name = n_state_name.split('_')[-1]
                    n_state_negative = env.stateIndex_negative[n_state_negative_name]
                    n_state_negative_onehot = [0,0]
                    n_state_negative_onehot[n_state_negative] = 1


                if a_name == 'change_theme':
                    reward = 0
                    if reward_seiki == 'True':
                        reward = seiki_zscore[seiki.index(reward)]

                    else:
                        reward = reward# 報酬計算

                else:
                    reward = self.get_reward(env, usermodel, s, n_state, a_name_dialogue, s2, n_state2, a_name)# 報酬計算
                    #reward = 0
                    if reward_seiki == 'True':
                        reward = seiki_zscore[seiki.index(reward)]

                    else:
                        reward = reward# 報酬計算


                s_dim_list = n_state_states2_ID_onehot + n_state_noun_presence_onehot
                s_dim_list.append(n_state_impression)
                s_dim_list+=n_state_negative_onehot
                #s_dim = np.array([n_state_sys_da_onehot[0],n_state_sys_da_onehot[1],n_state_sys_da_onehot[2],n_state_sys_da_onehot[3],n_state_noun_presence[0],n_state_impression])
                s_dim = np.array(s_dim_list)

                s = n_state
                s2 = n_state2
                s2_number = env.stateIndex2[s2]

                rewards.append(reward)
                state_history.append(n_state_name)
                state2_history.append(n_state2)
            else:
                self.append_log_reward(np.mean(rewards))
            agent_DQN.stop_episode_and_train(s_dim, reward, done)
            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


        agent_DQN.save('agent_'+name)
        
        #Q関数からQテーブルへ
        for state_number,state_index_list in enumerate(env.states2_index_list):


            state_index_list = copy.copy(state_index_list)
            state_index_list = list(state_index_list)

            state_index_states2_ID_onehot = list(np.zeros(len(env.stateIndex_states2_ID)))
            state_index_noun_presence_onehot = [0,0]
            state_index_states2_ID_onehot[state_index_list[0]] = 1
            state_index_noun_presence_onehot[state_index_list[1]] = 1
            state_index_negative_onehot=[0,0]
            state_index_negative_onehot[state_index_list[3]] = 1
            state_index_impression = list(env.stateIndex_impression.values())

            state_index_onehot = state_index_states2_ID_onehot + state_index_noun_presence_onehot
            state_index_onehot.append(state_index_impression[state_index_list[2]])
            state_index_onehot+=state_index_negative_onehot

            pre_q_values = copy.copy(agent_DQN._evaluate_model_and_update_recurrent_states([np.array(state_index_onehot)], test=True))

            q_values_tmp = pre_q_values.q_values.array
            q_values_tmp = q_values_tmp.tolist()
            q_values = list(itertools.chain.from_iterable(q_values_tmp))




            for action_number,q_value in enumerate(q_values):


                self.Q2[state_number][action_number] = q_value




# システム側（学習済み）
class TrainedQlearningAgent(Agent):
    def __init__(self, filename):
        super().__init__(epsilon=0)
        #Q_name = [sports,music,eating,travel]
        #for i in filename:
        self.df_sports = pd.read_csv(self.get('path_utterance_by_class_named_sports'))
        self.df_music = pd.read_csv(self.get('path_utterance_by_class_named_music'))
        self.df_eating = pd.read_csv(self.get('path_utterance_by_class_named_eating'))
        self.df_travel = pd.read_csv(self.get('path_utterance_by_class_named_travel'))

        #self.df_{}.format(i) = pd.read_csv(self.get(path_utterance_by_class_named_{}.format(i)))
        #self.df = pd.read_csv(self.get('path_utterance_by_class_named'))
        # 学習すみQテーブルの読み込み
        with open(filename[0], mode='rb') as f:
            self.Q_sports = pickle.load(f)
        with open(filename[1], mode='rb') as f:
            self.Q_music = pickle.load(f)
        with open(filename[2], mode='rb') as f:
            self.Q_eating = pickle.load(f)
        with open(filename[3], mode='rb') as f:
            self.Q_travel = pickle.load(f)

    # Qの学習されていないところを埋める
    def fillQ(self, env_sports, env_music, env_eating, env_travel):
        for k in range(len(env_sports.states2)):
            if k not in self.Q_sports.keys():
                self.Q_sports[k] = [0] * len(env_sports.actions2)
            else:
                pass
        for k in range(len(env_music.states2)):
            if k not in self.Q_music.keys():
                self.Q_music[k] = [0] * len(env_music.actions2)
            else:
                pass
        for k in range(len(env_eating.states2)):
            if k not in self.Q_eating.keys():
                self.Q_eating[k] = [0] * len(env_eating.actions2)
            else:
                pass
        for k in range(len(env_travel.states2)):
            if k not in self.Q_travel.keys():
                self.Q_travel[k] = [0] * len(env_travel.actions2)
            else:
                pass

    # システム発話を入力として，(class, theme)を出力する
    def getUtteranceClassTheme(self, utterance):
        classFile = self.get('path_utterance_by_class_named')
        themeFile = self.get('path_theme_info')
        CLSdf = pd.read_csv(classFile)
        THEMEdf = pd.read_csv(themeFile)

        if '***' in utterance:
            return '-', '-'
        else:
            clsInfo = CLSdf[CLSdf['agent_utterance'] == utterance]['cls'].values.astype('str')
            clsInfo = '-'.join(clsInfo)
            themeInfo = THEMEdf[THEMEdf['agent_utterance'] == utterance]['theme'].values[0]
            return clsInfo, themeInfo
    
    def negative(self,user_utterance,sonar):#ユーザ発話がネガティブかどうかonehot
        #sonar=Sonar()
        nega = sonar.ping(text=user_utterance)
        if nega['top_class'] == 'negative':
            return 1
        else:
            return 0

    def negative_name(self,user_utterance,sonar):#ユーザ発話がネガティブかどうかの状態名
        #sonar=Sonar()
        nega = sonar.ping(text=user_utterance)
        if nega['top_class'] == 'negative':
            return "ne"
        else:
            return "po"
    
    def append_keyword(self,sys_utterance,confident_noun,action_dialogue):#もとの発話に復唱を追加
        if confident_noun!="" and (action_dialogue=="re_o_m" or action_dialogue=="qs_o_d"):
            keyword=confident_noun+"ですか。"
            utterance=keyword+sys_utterance
            return utterance
        else:
            return sys_utterance
    
    def conversation(self, env_sports, env_music, env_eating, env_travel):#学習済みQテーブルを用いた対話
        self.init_log()
        sonar=Sonar()
        s2 = "0"
        themeHis = HistoryTheme(random_choice=False)#reset
        state2_history = []
        #self.fillQ(env)
        len_chg_theme=0#話題変更から何ターン
        user_utt_renzoku7_onehot=0#7文字以下のユーザ発話が連続するかどうか
        tokutei_machigai=0#特定名詞なしのユーザ発話が連続するかどうか
        user_utterance=""#キーワード
        confident_noun=""
        sys_utterance_history = []
        user_utterance_history=[]
        len_user_utt_history=[]
        for n_exchg in range(100):
            #start=time.time()
            if n_exchg == 0:
                chg_theme, theme = themeHis.decideNextTheme(None,None,0,0,0,False)
                #theme="音楽"
                #chg_theme=True
                s2_number = 28
                if theme=="END":
                    break
            else:
                tsukaitsukushi = qs_x_s_utterance_list.issubset(env.history_sysutte)#指示語なし質問が残っているかどうか
                chg_theme, theme = themeHis.decideNextTheme(user_utt_negative_onehot, len_user_utterance,len_chg_theme,user_utt_renzoku7_onehot,tokutei_machigai,tsukaitsukushi)
                if theme=="END":
                    break
            if chg_theme:
                len_chg_theme = 0
                len_user_utt_history=[]
                user_utterance_history=[]
                #話題に沿った学習済みモデルに変更
                if theme == 'スポーツ':
                    env = copy.deepcopy(env_sports)
                    Q = copy.deepcopy(self.Q_sports)
                    df = copy.deepcopy(self.df_sports)
                    modelname = 'path_utterance_by_class_named_sports'
                if theme == '音楽':
                    env = copy.deepcopy(env_music)
                    Q = copy.deepcopy(self.Q_music)
                    df = copy.deepcopy(self.df_music)
                    modelname = 'path_utterance_by_class_named_music'
                if theme == '食事':
                    env = copy.deepcopy(env_eating)
                    Q = copy.deepcopy(self.Q_eating)
                    df = copy.deepcopy(self.df_eating)
                    modelname = 'path_utterance_by_class_named_eating'
                if theme == '旅行':
                    env = copy.deepcopy(env_travel)
                    Q = copy.deepcopy(self.Q_travel)
                    df = copy.deepcopy(self.df_travel)
                    modelname = 'path_utterance_by_class_named_travel'
                qs_x_s_utterance_list = env.ID_df[env.ID_df['cls_ID_state'].str.contains('qs_x_s')]
                qs_x_s_utterance_list = set(qs_x_s_utterance_list['agent_utterance'])
                action_dialogue="ct"
            else:
                len_chg_theme+=1

            sys_utterance = env.utterance_selection_softmax(chg_theme, theme, Q[int(s2_number)], modelname, coef=5)# 発話選択
            sys_utterance_history.append(sys_utterance)
            if not chg_theme:
                action_dialogue = env.ID_df[env.ID_df["agent_utterance"]==sys_utterance]["cls"].values[0]

            sys_utterance_keyword=self.append_keyword(sys_utterance,confident_noun,action_dialogue)


            #変更禁止
            print("system utterance :",sys_utterance_keyword)
            user_utterance = input('your utterance >> ')# 発話入力
            print()
            impression = float(input('your impression >> '))# 心象入力
            print()
            confident_noun =input('confident_noun >> ')# キーワード入力
            print()

            
            negative=self.negative_name(user_utterance,sonar)
            n_state2 = env.get_state_name(chg_theme, df, impression, sys_utterance, user_utterance, state2_history,negative)
            len_user_utterance=len(user_utterance)
            if not chg_theme:
                len_user_utt_history.append(len_user_utterance)
                user_utterance_history.append(user_utterance)
            user_utt_negative_onehot=self.negative(user_utterance,sonar)#ユーザ発話がネガティブかどうか
            states2 = [k for k, v in env.stateIndex2.items() if v == s2_number]
            self.append_log_dialogue(str(n_exchg).zfill(2),
                states2[0],
                env.history_sysutte_class[-1],
                self.getUtteranceClassTheme(sys_utterance)[1],
                impression,
                sys_utterance_keyword,
                user_utterance)

            # 更新
            s2 = n_state2
            s2_number = env.stateIndex2[s2]
            state2_history.append(n_state2)
            #特定名詞なしのユーザ発話が連続するかどうか
            if len(user_utterance_history)>0 and env.getSpecificNoun(user_utterance_history[-1])=="Nx":
                if len(user_utterance_history)>1 and env.getSpecificNoun(user_utterance_history[-2])=="Nx":
                    tokutei_machigai=1
                else:
                    tokutei_machigai=0
            else:
                tokutei_machigai=0
            #7文字以下のユーザ発話が連続するかどうか
            if len(len_user_utt_history)>0 and len_user_utt_history[-1]<8:
                if len(len_user_utt_history)>1 and len_user_utt_history[-2]<8:

                    user_utt_renzoku7_onehot=1
                else:
                    user_utt_renzoku7_onehot=0
            else:
                user_utt_renzoku7_onehot=0
        print("End of dialogue")




