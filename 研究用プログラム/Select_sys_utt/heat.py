import numpy as np
from matplotlib import pyplot as plt
import sys
fig = plt.figure()
dataname = 'sample210107_'
for i in range(1):
    dataname = ''
    dataname = dataname + sys.argv[1] +'_Q.csv'
    data = np.loadtxt(dataname, delimiter=",")
    plt.imshow(data, aspect="auto", interpolation = "none")
    plt.xlabel('stateID')
    plt.ylabel('actionID')
    plt.colorbar()
    #plt.show()
    dataname = dataname.strip('.csv')
    fig.savefig(dataname.format('.eps'))
