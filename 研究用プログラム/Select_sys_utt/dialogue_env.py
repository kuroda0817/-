# coding:utf-8
import pandas as pd
import sys
import numpy as np
from params import params
import itertools
from optparse import OptionParser
import copy
import MeCab
from sklearn import preprocessing
import scipy
import pickle
import heapq

def fillQ2(Q,state_size,action_size):
    for k in range(state_size):
        if k not in Q.keys():
            Q[k] = [0] * action_size
        else:
            pass
    return Q


# 対話環境
class DialogueEnv(params):
    """docstring for DialogueEnv"""
    def __init__(self, path_utterance_by_class_named):
        super().__init__()
        self.history_sysutte = []
        self.history_sysutte_class = []

        # action
        
        self.action_df = pd.read_csv(self.get('path_class_name'))
        actions = self.action_df['clsname'].values.tolist()
        self.actions = actions
        
        self.ID_df = pd.read_csv(self.get(path_utterance_by_class_named))

        self.actions2 = list(self.ID_df['cls_ID_action'])

        self.actions2 = list(dict.fromkeys(self.actions2))#重複とる

        self.actions2.remove('ct_0')


        # actionにindex付け
        self.actionIndex = {}
        for i, val in enumerate(self.actions):
            self.actionIndex[i] = val
        
        self.actionIndex2 = {}
        for i, val in enumerate(self.actions2):
            self.actionIndex2[i] = val

        print(self.actionIndex2)
        self.sports_key = []
        self.music_key = []
        self.eating_key = []
        self.travel_key = []

        # 状態は「心象」「直前のシステム対話行為」「対話の位置」の組み合わせ
        
        self.states_sys_da = ['ct','io','re','qs']
        self.states_noun_presence = ['Nx','No']
        self.states_impression = ['l','n','h']
        self.states_negative = ['ne','po']
        self.states = list(itertools.product(self.states_sys_da, self.states_noun_presence, self.states_impression,self.states_negative))
        self.states = ['_'.join(x) for x in self.states]


        prehis = self.ID_df[(self.ID_df['cls_ID_state'].str.contains('qs_o_s')) | (self.ID_df['cls_ID_state'].str.contains('qs_x_s'))]
        self.pre_states_history = list(prehis['cls_ID_state'])
        prehis2 = self.ID_df[self.ID_df['cls_ID_state'].str.contains('qs_o_d')]
        self.pre_states_history2 = list(prehis2['cls_ID_state'])
        self.states_history = list(itertools.product(self.pre_states_history, self.pre_states_history2))
        self.states_history = ['_'.join(x) for x in self.states_history]
        self.states2_ID = list(self.ID_df['cls_ID_state'])
        self.states2_ID = self.states2_ID + self.states_history
        self.states2_ID = list(dict.fromkeys(self.states2_ID))

        self.states2 = list(itertools.product(self.states2_ID, self.states_noun_presence,self.states_impression,self.states_negative))

        self.states2=['_'.join(x) for x in self.states2]
        self.states2_index_list = list(itertools.product(list(range(len(self.states2_ID))), list(range(len(self.states_noun_presence))), list(range(len(self.states_impression))), list(range(len(self.states_negative)))))

        
        del_list=[]
        for i in self.states2:

            if 'ct' in i:
                del_list.append(i)
        for i in del_list:
            self.states2.remove(i)

        self.states2.append('ct_0')
        # stateにindex付け
        self.stateIndex = {}
        for i, val in enumerate(self.states):
            self.stateIndex[val] = i

        self.stateIndex2 = {}
        for i, val in enumerate(self.states2):
            self.stateIndex2[val] = i

        print(self.stateIndex2)

        #生データを正規化(心象)
        self.stateIndex_impression = {}
        for i, val in enumerate(self.states_impression):
            self.stateIndex_impression[val] = i
        #zscore = scipy.stats.zscore(list(self.stateIndex_impression.values()))
        zscore = preprocessing.minmax_scale(list(self.stateIndex_impression.values()))
        for k, i in enumerate(self.stateIndex_impression.keys()):
            self.stateIndex_impression[i] = zscore[k]


        self.stateIndex_states2_ID = {}

        for i, val in enumerate(self.states2_ID):
            self.stateIndex_states2_ID[val] = i

        #self.stateIndex_states2_ID['ct_0'] = i+1

        self.stateIndex_noun_presence = {}
        for i , val in enumerate(self.states_noun_presence):
            self.stateIndex_noun_presence[val] = i

        self.stateIndex_negative = {}
        for i , val in enumerate(self.states_negative):
            self.stateIndex_negative[val] = i


        self.thres_low_UI = 3
        self.thres_high_UI = 5
        self.persist_UI_exchgs = 3
        self.reward_da_df = pd.read_csv(self.get('path_reward_da'), index_col=0)
        self.weight_specific_theme = 0.6
        saveQ = 'sample210118_776_Q'
        with open(saveQ, 'rb') as f:
            Q = pickle.load(f)
            Q_save = fillQ2(Q,24,8)

            self.dialogue_Q = Q_save

    # 初期化のような感じ
    def reset(self):
        super().__init__()
        self.history_sysutte = []
        self.history_sysutte_class = []
        return self.stateIndex['io_Nx_n_ne'] #エラー吐くから消した

    # 対話行為を簡単な分類に変換(4種類(ct/io/re/qs))
    def getSimpleDAFromSysUtterance(self, sys_utterance):
        df = pd.read_csv(self.get('path_simple_da'))
        da = df[df['agent_utterance'] == sys_utterance]['da_simple'].values
        if '***' in sys_utterance:
            simple_da = 'ct'
        else:
            simple_da = da[0]
        return simple_da

    # 入力：交換番号，出力：前半か後半
    def getDialoguePosition(self, exchg_progress):
        if exchg_progress < 0.5:
            position = 'fh'
        else:
            position = 'lh'
        return position

    # 心象を離散化
    def getImpressionLevel(self, impression):
        if impression <= self.thres_low_UI:
            impression_level = 'l'
        elif self.thres_high_UI <= impression:
            impression_level = 'h'
        else:
            impression_level = 'n'
        return impression_level

    # 文から固有名詞/一般名詞の存在有無を判断
    def getSpecificNoun(self, sentence):
        #mt = MeCab.Tagger(r'-owakati -d /var/lib/mecab/dic/ipadic-utf8')
        mt = MeCab.Tagger()
        node = mt.parseToNode(sentence)
        properNouns = []
        while node:
            fields = node.feature.split(",")
            if (fields[0] == '名詞') and (fields[1] in ['固有名詞', '一般']):
                properNouns.append(node.surface)
            node = node.next
        if len(properNouns) > 0:
            return 'No'
        else:
            return 'Nx'

    def getgroup(self,sys_utterance):
        da = self.ID_df[self.ID_df['agent_utterance']==sys_utterance]['group'].values
        return da

    def getcls_ID_state(self,sys_utterance):
        da = self.ID_df[self.ID_df['agent_utterance']==sys_utterance]['cls_ID_state'].values[0]
        return da
    # 次のstateを決定
    def get_next_state(self, impression, sys_utterance, user_utterance,negative):
        
        da_simple = self.getSimpleDAFromSysUtterance(sys_utterance)
        impression_level = self.getImpressionLevel(impression)
        noun_presence = self.getSpecificNoun(user_utterance)
        n_state = self.stateIndex['{}_{}_{}_{}'.format(da_simple, noun_presence, impression_level,negative)]

        return n_state

    def get_next_state2(self, impression, sys_utterance, user_utterance,negative):
        da_simple = self.getcls_ID_state(sys_utterance)
        impression_level = self.getImpressionLevel(impression)
        noun_presence = self.getSpecificNoun(user_utterance)
        n_state = '{}_{}_{}_{}'.format(da_simple, noun_presence, impression_level,negative)
        return n_state
        
    def get_state_name(self, chg_theme, df, impression, sys_utterance, user_utterance, state2_history,negative):#
        if chg_theme:
            n_state2 = 'ct_0'
        else:

            a_name = df[df['agent_utterance']==sys_utterance]['cls_ID_action'].values[0]
            a_name_dialogue = a_name.split('_')
            del a_name_dialogue[-1]
            a_name_dialogue ='_'.join(a_name_dialogue)
           
            hozon = 0
            if a_name_dialogue == 'qs_o_d':
                for i in state2_history:
                    if len(i.split('_'))<9 and ('qs_o_s' in i or 'qs_x_s' in i):
                        hozon = i.split('_')[0:4]
                        hozon = '_'.join(hozon)
            
            if hozon != 0:
                n_state2 = hozon + '_' + self.get_next_state2(impression, sys_utterance, user_utterance,negative)
            else:
                n_state2 = self.get_next_state2(impression, sys_utterance, user_utterance,negative)
        return n_state2

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



    # 特定話題の選択に重み（weight）をつける
    def weightSpecificTheme(self, df):
        themes = df['theme'].values
        themes = [1-self.weight_specific_theme if t == 'default' else self.weight_specific_theme for t in themes]
        themes = [x/np.sum(themes) for x in themes]
        df = df.reset_index(drop=True)
        select_index = np.random.choice(df.index.values, p=themes)
        return df.loc[select_index]

    def utterance_selection(self, action, theme):#行動(action)から発話を選択(学習用)
        # 話題変更のタイミングでは専用の発話を使用する
        action_dialogue = action.split('_')
        del action_dialogue[-1]
        action_dialogue = '_'.join(action_dialogue)
        if action == 'change_theme':
            next_sysutte = ' *** これから{}の話をしましょう***'.format(theme)
            self.add_sysutte(next_sysutte, action)
        else:
            # 選択する
            df = self.ID_df
            #CANDIDATEdf = df[df['ID117'] == action]
            CANDIDATEdf = df[(df['cls_ID_action'] == action) & ((df['theme'] == theme) | (df['theme'] == 'default'))]


            CANDIDATEdf = CANDIDATEdf.reset_index(drop=True)
            CANDIDATEdf = CANDIDATEdf[['agent_utterance', 'theme', 'cls']]

            if len(CANDIDATEdf) == 0:
                df[df['cls'] == action_dialogue]
                CANDIDATEdf = CANDIDATEdf.reset_index(drop=True)
                CANDIDATEdf = CANDIDATEdf[['agent_utterance', 'theme', 'cls']]

            # 使えないものを削除
            
            for i in range(len(CANDIDATEdf)):
                if CANDIDATEdf.loc[i, :]['agent_utterance'] in self.history_sysutte:
                    CANDIDATEdf = CANDIDATEdf.drop(index=[i])

            if len(CANDIDATEdf) == 0:
                CANDIDATEdf = df[(df['cls'] == action_dialogue) & ((df['theme'] == theme) | (df['theme'] == 'default'))]
                CANDIDATEdf = CANDIDATEdf.reset_index(drop=True)
                CANDIDATEdf = CANDIDATEdf[['agent_utterance', 'theme', 'cls']]            


            # 候補が残っていないなら，action気にせず候補を決定
            
            if len(CANDIDATEdf) == 0:
                CANDIDATEdf = df
                CANDIDATEdf = CANDIDATEdf.reset_index(drop=True)
                CANDIDATEdf = CANDIDATEdf[['agent_utterance', 'theme', 'cls']]
                # 使えないものを削除
                for i in range(len(CANDIDATEdf)):
                    if CANDIDATEdf.loc[i, :]['agent_utterance'] in self.history_sysutte:
                        CANDIDATEdf = CANDIDATEdf.drop(index=[i])

            # 選択して終了
            SELECTdf = self.weightSpecificTheme(CANDIDATEdf)
            next_sysutte, next_theme, next_action = SELECTdf.values
            self.add_sysutte(next_sysutte, next_action)

        return next_sysutte

    # actionに基づいた発話選択（ランダム選択）
    # heatmapにsoftmaxをちゃんと反映させた
    def zscore(x):
        xmean = x.mean()
        xstd  = np.std(x)

        zscore = (x-xmean)/xstd
        return zscore
    def utterance_selection_softmax(self, chg_theme, theme, prob_actions, modelname, coef=1):
        actions = copy.deepcopy(self.actions2)

        prob_actions = copy.deepcopy(prob_actions)

        t = 0
        #prob_actions = scipy.stats.zscore(prob_actions,axis=0)
        # 話題変更のタイミングでは専用の発話を使用する
        if chg_theme:
            next_sysutte = ' *** これから{}の話をしましょう***'.format(theme)
            self.add_sysutte(next_sysutte, 'change_theme')
        else:
            done = False
            while not done:
                # 選択する

                actions_dic = dict(zip(actions,prob_actions))
                actions_dic_sorted = dict(sorted(actions_dic.items(), key=lambda x:x[1],reverse=True))
                actions_sorted = list(actions_dic_sorted.keys())
                prob_actions_sorted = list(actions_dic_sorted.values())

                k = 0
                actions_sorted_max5 = actions_sorted[:5]
                prob_actions_sorted_max5 = prob_actions_sorted[:5]

                while t == 0:
                    if np.all(np.array(prob_actions_sorted)==0):
                        prob_actions_sorted_max5 = [0.2,0.2,0.2,0.2,0.2]
                        #prob_actions_sorted_max5 = list(scipy.stats.zscore(prob_actions_sorted_max5,axis=0))
                    else:
                        prob_actions_sorted_max5 = list(scipy.stats.zscore(prob_actions_sorted_max5,axis=0))
                    action = np.random.choice(actions_sorted_max5, p=self.softmax(prob_actions_sorted_max5, coef=coef))
                    action_index = actions_sorted_max5.index(action)
                    #action = np.random.choice(actions_dialogue, p=self.softmax(prob_actions_dialogue, coef=coef))
                    df = pd.read_csv(self.get(modelname))
                    CANDIDATEdf = df[(df['cls_ID_action'] == action) & ((df['theme'] == theme) | (df['theme'] == 'default'))]
                    CANDIDATEdf = CANDIDATEdf.reset_index(drop=True)
                    CANDIDATEdf = CANDIDATEdf[['agent_utterance', 'theme', 'cls', 'cls_ID_action']]
                    
                    # 使えないものを削除
                    for i in range(len(CANDIDATEdf)):
                        if CANDIDATEdf.loc[i, :]['agent_utterance'] in self.history_sysutte:
                            CANDIDATEdf = CANDIDATEdf.drop(index=[i])

                    # 候補が残っていない
                    if len(CANDIDATEdf) == 0:
                        t = 0


                        del actions_sorted_max5[action_index]
                        del prob_actions_sorted_max5[action_index]
                        k += 1
                        actions_sorted_max5.append(actions_sorted[5+k])
                        prob_actions_sorted_max5.append(prob_actions_sorted[5+k])
                    else:
                        t = 1

                # 候補が残っている（選択して終了）
                else:
                    SELECTdf = self.weightSpecificTheme(CANDIDATEdf)
                    next_sysutte, next_theme, next_action, next_ID117 = SELECTdf.values
                    self.add_sysutte(next_sysutte, next_action)
                    done = True
                    #print(self.history_sysutte)

        return next_sysutte

    def add_sysutte(self, utterance, clas):
        self.history_sysutte.append(utterance)
        self.history_sysutte_class.append(clas)

    # ソフトマックス関数
    # coefは推定値の振れ幅を調整するためのもの．（デフォルトは1）
    def softmax(self, a, coef=1):
        c = np.max(a)
        exp_a = np.exp(coef * (a - c))
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y



if __name__ == '__main__':


    pass



