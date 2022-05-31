# -*-coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import re
import MeCab


def getThemeOnehot(df):
	themeInfo = df['theme'].values#含まれていること前提
	themeDict = {}
	theme = list(sorted(set(themeInfo)))
	for i, val in enumerate(theme):
		themeDict[val] = i

	themeInfo = [themeDict[x] for x in themeInfo]
	theme_onehot = np.eye(len(theme))[themeInfo]
	for i, t in enumerate(theme):
		df[t] = theme_onehot[:, i]
	return df


# dfに特徴量増やしてreturn
# 指示語，相槌，感謝についての単語
def getSpecificWord(df):
	sysUtte = df['agent_utterance'].values

	WORDdf = pd.read_csv('./refData/dependWord.txt', header=None)
	CDword = WORDdf[0].values
	CDfea = [0] * len(sysUtte)
	for i, val in enumerate(sysUtte):
		for w in CDword:
			if w in val:
				CDfea[i] = 1
				break
	df['depend_word'] = CDfea

	WORDdf = pd.read_csv('./refData/nodWord.txt', header=None)
	CDword = WORDdf[0].values
	CDfea = [0] * len(sysUtte)
	for i, val in enumerate(sysUtte):
		for w in CDword:
			if w in val:
				CDfea[i] = 1
				break
	df['nod_word'] = CDfea

	WORDdf = pd.read_csv('./refData/thankWord.txt', header=None)
	CDword = WORDdf[0].values
	CDfea = [0] * len(sysUtte)
	for i, val in enumerate(sysUtte):
		for w in CDword:
			if w in val:
				CDfea[i] = 1
				break
	df['thank_word'] = CDfea

	return df


# dfに特徴量増やしてreturn
# ユーザの発話長，直前と直後のもの
# 音声認識の不具合がおおいので，語数ではなく発話時間として近似している（2019/09/11）
def getUserUtteLength(df, userID):
	TSdf = pd.read_csv('/Users/haruto/Desktop/mainwork/MMdata_201902/timestamp/{}.csv'.format(userID))
	index = TSdf['ts'].values
	index = [userID + '_' + x.replace('ts','').zfill(3) for x in index]
	TSdf['name'] = index
	TMPdf = pd.merge(df, TSdf, on='name')

	s_end = TMPdf['s_end'].values
	u_end = TMPdf['u_end'].values

	u_utte_len_after = u_end - s_end
	u_utte_len_before = np.array([0] + u_utte_len_after[:-1].tolist())
	df['u_utte_len_after'] = u_utte_len_after / 1000 * 5
	df['u_utte_len_before'] = u_utte_len_before / 1000 * 5
	return df

# dfに特徴量増やしてreturn
# システム発話長の特徴量
def getSysUtteLength(df):
	lenS = df['lenS'].values
	lenS_diff = [0] + np.diff(lenS).tolist()
	df['lenS_diff'] = lenS_diff
	return df

# dfに特徴量増やしてreturn
# 前後の発話の名詞の数
def getNumNoun(df):
	num_noun_after = df['名詞'].values
	num_noun_before = [0] + num_noun_after[:-1].tolist()
	df['num_noun_after'] = num_noun_after
	df['num_noun_before'] = num_noun_before
	return df

# dfに特徴量増やしてreturn
# UIについての特徴量
def getUserImpression(df):
	UI3average = df['label'].values
	UI3average_diff = [0] + np.diff(UI3average).tolist()
	df['UI3average'] = UI3average
	df['UI3average_diff'] = UI3average_diff
	return df

# 表層的な特徴量（？が入っているかどうか）
def getIfQuestion(df):
	sysUtte = df['agent_utterance'].values
	# 入っている:1，入っていない:0
	sysUtte = [1 if '？' in x else 0 for x in sysUtte]
	df['questionmark'] = sysUtte
	return df


if __name__ == '__main__':
	pass