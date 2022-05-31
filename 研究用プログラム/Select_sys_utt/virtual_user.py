# coding:utf-8
import pandas as pd
import numpy as np
from params import params
import math


#### ユーザモデルを定義するファイル

# ユーザモデル
class UserModel(params):

    def __init__(self):
        super().__init__()
        self.voca = pd.read_csv(self.get('path_virtual_user_data'))
        self.log_UI_1theme = []
        self.log_sys_utterance_1theme = []
        self.log_user_utterance_1theme = []

    # システム発話に対して何か応答する
    # コーパスからs−uの交換の情報を用いて応答
    def getResponse(self, sys_utterance):

        # UIが範囲[1, 7]を超えないように調整
        def controlImpressionRange(impression, diff, min_UI=1.0, max_UI=7.0):
            if impression + diff < min_UI:
                new_impression = min_UI
            elif max_UI < impression + diff:
                new_impression = max_UI
            else:
                new_impression = impression + diff
            return new_impression

        # ユーザ心象の初期値を生成
        def initImpression(myu=4.0, sigma=1.0):
            ND = lambda x: (math.exp(-(x-myu)**2/(2*sigma**2))) / math.sqrt(2*math.pi)
            UI_candidate = range(1, 7+1)
            UI_prob = [ND(x) for x in UI_candidate]# 正規分布とって
            UI_prob = [x/np.sum(UI_prob) for x in UI_prob]# 確率に変更
            return np.random.choice(UI_candidate, p=UI_prob)

        # 候補が複数あるときはランダムに選択
        if '***' in sys_utterance:
            user_utterance = 'はい'
            self.UI = initImpression()
            self.log_UI_1theme = [self.UI]
            self.log_sys_utterance_1theme = [sys_utterance]
            self.log_user_utterance_1theme = [user_utterance]
            UI_diff = 0.0
        else:
            responseInfo = self.voca[self.voca['sys_utterance'] == sys_utterance].sample()
            user_utterance = responseInfo['user_utterance'].values[0]
            UI_diff = responseInfo['UI3average_diff'].values[0]

            self.UI = controlImpressionRange(self.UI, UI_diff)
            self.log_UI_1theme.append(self.UI)
            self.log_sys_utterance_1theme.append(sys_utterance)
            self.log_user_utterance_1theme.append(user_utterance)

        return user_utterance, self.UI



