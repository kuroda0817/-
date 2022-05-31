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

optparser = OptionParser()
optparser.add_option('-A', dest='action', default=None, type='str')
optparser.add_option('--model', dest='model', default='sample', type='str')
optparser.add_option('--ep', dest='n_episode', default=268500, type='int')
optparser.add_option('--seed', dest='seed', default=776, type='int')
optparser.add_option('--alpha', dest='alpha', default=0.1, type='float')
optparser.add_option('--interval', dest='interval', default=10, type='int')
optparser.add_option('--epsilon', dest='epsilon', default=0.1, type='float')
optparser.add_option('--coef_epsilon', dest='coef_epsilon', default=0.99997, type='float')#0.99995 179000
(options, args) = optparser.parse_args()
if options.action is None:
    sys.exit('System will exit')
else:
    print('############\n{}\n############'.format(str(options)))

# seed
np.random.seed(options.seed)

######## python q_learning.py -A [ACT] --model [MODEL] ##############
Qtable_name = '{}/{}_Q'.format(options.model, options.model)
Qfreq_name = '{}/{}_Qfreq'.format(options.model, options.model)
hm_name = '{}/{}_hm.png'.format(options.model, options.model)
reward_name = '{}/{}_reward.png'.format(options.model, options.model)
reward_list_name = '{}/{}_reward.npy'.format(options.model, options.model)
log_name = '{}/{}_log.csv'.format(options.model, options.model)

# params
params = params()

def dialogue(self, env, n_exchange):
