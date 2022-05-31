# coding:utf-8
import sys
import csv
from params import params
import pandas as pd
import numpy as np
import copy
import pickle
import os
import MeCab
import agent
from agent import Agent, QlearningAgent, TrainedQlearningAgent
from heatmapQ import show_q_value, make_risouQ
from optparse import OptionParser
from dialogue_env import DialogueEnv
import time
import warnings
def write_csv(file, save_dict):#辞書をcsvにする
    save_row = {}

    with open(file,'w') as f:
        writer = csv.DictWriter(f, fieldnames=save_dict.keys(),delimiter=",",quotechar='"')
        writer.writeheader()

        k1 = list(save_dict.keys())[0]
        length = len(save_dict[k1])

        for i in range(length):
            for k, vs in save_dict.items():
                save_row[k] = vs[i]

            writer.writerow(save_row)

if __name__ == "__main__":
    # option
    optparser = OptionParser()
    optparser.add_option('-A', dest='action', default=None, type='str')
    optparser.add_option('--model', dest='model', default='sample', type='str')
    optparser.add_option('--model_sports', dest='model_sports', default='sample', type='str')
    optparser.add_option('--model_music', dest='model_music', default='sample', type='str')
    optparser.add_option('--model_eating', dest='model_eating', default='sample', type='str')
    optparser.add_option('--model_travel', dest='model_travel', default='sample', type='str')
    optparser.add_option('--ep', dest='n_episode', default=50000, type='int')
    optparser.add_option('--seed', dest='seed', default=776, type='int')
    optparser.add_option('--alpha', dest='alpha', default=0.1, type='float')
    optparser.add_option('--interval', dest='interval', default=10, type='int')
    optparser.add_option('--epsilon', dest='epsilon', default=0.1, type='float')
    optparser.add_option('--coef_epsilon', dest='coef_epsilon', default=0.99, type='float')#0.99995 179000
    optparser.add_option('--m_node', dest='m_node', default=1000, type='int')
    optparser.add_option('--seiki', dest='seiki', default="True", type='str')
    optparser.add_option('--bigramhosyu', dest='bigramhosyu', default=-10, type='int')
    optparser.add_option('--learning_theme', dest='learning_theme', default='sports', type='str')
    (options, args) = optparser.parse_args()
    if options.action is None:
        sys.exit('System will exit')
    else:
        print('############\n{}\n############'.format(str(options)))

    # seed 学習時は固定,対話時は外す
    #np.random.seed(options.seed)

    ######## python q_learning.py -A [ACT] --model [MODEL] ##############
    Qtable_name = '{}/{}_Q'.format(options.model, options.model)
    Qtable_name_list = ['{}/{}_Q'.format(options.model_sports, options.model_sports), '{}/{}_Q'.format(options.model_music, options.model_music), '{}/{}_Q'.format(options.model_eating, options.model_eating), '{}/{}_Q'.format(options.model_travel, options.model_travel)]
    Qfreq_name = '{}/{}_Qfreq'.format(options.model, options.model)
    hm_name = '{}/{}_hm.png'.format(options.model, options.model)
    reward_name = '{}/{}_reward.png'.format(options.model, options.model)
    reward_list_name = '{}/{}_reward.npy'.format(options.model, options.model)
    log_name = '{}/{}_log.csv'.format(options.model_sports, options.model_sports)
    #各話題のモデルを指定
    env_sports_DQN = 'agent_{}'.format(options.model_sports, options.model_sports)
    env_music_DQN = 'agent_{}'.format(options.model_music, options.model_music)
    env_eating_DQN = 'agent_{}'.format(options.model_eating, options.model_eating)
    env_travel_DQN = 'agent_{}'.format(options.model_travel, options.model_travel)

    # params
    params = params()



    # Qを学習
    if options.action == 'train':
        t1 = time.time()
        # dir作成
        if not os.path.exists(options.model):
            os.mkdir(options.model)
        else:
            print('model "{}" already exists.'.format(options.model))
            if_del = input('### overwrite if you push enter. ###')

        env = DialogueEnv('path_utterance_by_class_named_{}'.format(options.learning_theme))
        agent = QlearningAgent(epsilon=options.epsilon)
        agent.learn(env,
            episode_count=options.n_episode,
            learning_rate=options.alpha,
            coef_epsilon=options.coef_epsilon, name=options.model, reward_seiki=options.seiki, m_node=options.m_node, bigramhosyu=options.bigramhosyu, learning_theme=options.learning_theme)
        agent.saveQ(agent.Q2, Qtable_name)
        agent.show_reward_log(interval=options.interval, filename=reward_name)
        agent.saveR(reward_list_name)
        show_q_value(agent.Q2, env.states, env.actions, hm_name)
        # アクセスした回数を保存
        #write_csv(os.path.join(r, Qtable_name.split("/")[0]),agent.Q2)
        t2 = time.time()
        #print(t2-t1)
        t = t2-t1
        print(t)

    # 学習済みQを用いて対話
    warnings.simplefilter('ignore')
    if options.action == 'dialogue':
        env_sports = DialogueEnv('path_utterance_by_class_named_sports')
        env_music = DialogueEnv('path_utterance_by_class_named_music')
        env_eating = DialogueEnv('path_utterance_by_class_named_eating')
        env_travel = DialogueEnv('path_utterance_by_class_named_travel')
        
        agent = TrainedQlearningAgent(Qtable_name_list)
        agent.fillQ(env_sports, env_music, env_eating, env_travel)
        #agent.conversation(env_sports_DQN, env_music_DQN, env_eating_DQN, env_travel_DQN)
        try:
            agent.conversation(env_sports, env_music, env_eating, env_travel)
        except:
            
            agent.write_dialogue_log(log_name)
        agent.write_dialogue_log(log_name)
    # 理想のQを作成
    if options.action == 'make_risouQ':
        modelname = params.get('path_risouQ')
        # dir作成
        if not os.path.exists(modelname):
            os.mkdir(modelname)
        else:
            print('model "{}" already exists.'.format(modelname))
            if_del = input('### overwrite if you push enter. ###')

        env = DialogueEnv()
        Q_name = '{}/{}'.format(modelname, modelname.split('/')[-1])
        make_risouQ(Q_name, env.states, env.actions)

    # 理想のQを用いて対話
    if options.action == 'dialogue_Q':
        modelname = params.get('path_risouQ')
        Q_name = '{}/{}'.format(modelname, modelname.split('/')[-1])
        log_name = '{}/{}_log.csv'.format(modelname, modelname.split('/')[-1])
        env = DialogueEnv()
        agent = TrainedQlearningAgent(Q_name)
        agent.fillQ(env)
        agent.conversation(env)
        agent.write_dialogue_log(log_name)



