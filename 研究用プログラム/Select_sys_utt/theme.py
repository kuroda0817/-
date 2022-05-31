# -*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd
import MeCab
from params import params
import pickle
from sklearn import preprocessing,svm
from sklearn.svm import SVC
# themeを管理してます
class HistoryTheme(params):
	def __init__(self, random_choice=True):
		super().__init__()
		#self.allTheme = list(pd.read_csv(self.get('path_using_theme'), header=None)[0].values)
		self.allTheme = ['スポーツ','音楽','食事','旅行']
		#self.allTheme.append('終了')
		self.random_choice = True
		self.nowTheme_ExchgNum = 0
		self.history_impression_1theme = []
		self.max_exchange_num_1theme = 10
		self.min_exchange_num_1theme = 5
		self.low_UI3_border = 3
		with open('clf_model.pickle', mode='rb') as f:
			self.load_clf = pickle.load(f)

	# 最初，話題変更する時はUIにNoneを入れること
	def decideNextTheme(self,user_utt_negative_onehot, len_user_utterance,len_chg_theme,user_utt_renzoku7_onehot,tokutei_machigai,tsukaitsukushi):
		#end = False 
		# 変更可否の決定
		#print(user_utt_negative_onehot, sys_utt_re_renzoku_onehot,len_user_utterance,len_chg_theme)#svmモデル
		if user_utt_negative_onehot == None:
			chg = True 
		elif tsukaitsukushi:
			chg=True
		else:
			data_setsumei=np.array([[user_utt_negative_onehot,len_chg_theme ,len_user_utterance,user_utt_renzoku7_onehot,tokutei_machigai],[1,27,294,1,1],[0,0,0,0,0]])
			features = preprocessing.minmax_scale(data_setsumei)
			svm=self.load_clf.predict([features[0]])
			if svm==1:
				chg = True
			else:
				chg = False
			#print(self.allTheme)

		if chg:# 変更ff
			if len(self.allTheme)==0:
				return True, "END"
			if self.random_choice:
				self.nowTheme = np.random.choice(self.allTheme)
			else:
				print('使用する話題をindexで指定してください')
				for i, val in enumerate(self.allTheme):
					print(i, val)
				index = int(input('>> '))
				self.nowTheme = self.allTheme[index]
			self.allTheme.remove(self.nowTheme)
			self.nowTheme_ExchgNum = 0
			self.history_impression_1theme = []
		else:# 変更しない
			self.nowTheme_ExchgNum += 1
			self.history_impression_1theme.append(3)

		return chg, self.nowTheme




