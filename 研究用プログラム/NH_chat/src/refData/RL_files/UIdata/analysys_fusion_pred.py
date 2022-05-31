import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mu, sigma = 100, 15
#x = mu + sigma * np.random.randn(10000)
x=pd.read_csv("fusion.csv")
x=x["fusion_pred"].values
print(np.mean(x),np.var(x),np.max(x),np.min(x))
y=[(6/(5.23-3.95))*(i-3.95)+1 for i in x]
y=[(10)*(i-4.4)+3 for i in x]
#y=[(2/(4.92-3.99))*(i-3.99)+3 for i in x]
hozon=[0,0]
x_hozon=[0,0]
count=[0,0,0]
for i,value in enumerate(y):
  if value<1:
    y[i]=1
    #hozon[0]=value
    #x_hozon[0]=x[i]
  if value>7:
    #count+=1
    y[i]=7
print(len(y))
#x=x["label"].values
for i in y:
    if i >= 5:
        count[2]+=1
    elif i <= 3:
        count[0]+=1
    else:
        count[1]+=1
print(count)
#print(np.mean(x),np.var(x),np.max(x),np.min(x))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bins=60
ax.hist(y, bins)
ax.set_title('svrモデルによる予測', fontname="MS Gothic")
ax.set_xlabel('心象(0.1きざみ)', fontname="MS Gothic")
ax.set_ylabel('度数', fontname="MS Gothic")
ax.set_xlim(1,7)
ax.set_ylim(0,600)
fig.show()
fig.savefig("count_{}.png".format(bins))