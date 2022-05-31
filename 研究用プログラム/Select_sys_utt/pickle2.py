import pickle
import csv
import sys

def fillQ(Q,state_size,action_size):
    for k in range(state_size):
        if k not in Q.keys():
            Q[k] = [0] * action_size
        else:
            pass
    return Q

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
saveQ = ""
print('name')
saveQ = saveQ + str(sys.argv[1]) + '_Q'
print('state_size')
state_size = int(sys.argv[2])
print('action_size')
action_size = int(sys.argv[3])
#saveQ = "sample201223_5_Q"
print(saveQ)
with open(saveQ, 'rb') as f:
    Q = pickle.load(f)
    Q_save = fillQ(Q,state_size,action_size)
    #print(Q_save)
    write_csv(saveQ + '.csv',Q_save)
