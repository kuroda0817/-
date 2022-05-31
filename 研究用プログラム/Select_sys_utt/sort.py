import pandas as pd
import csv
import sys
dataname = ''+sys.argv[1]+'_Q.csv'
#dataname = 
df = pd.read_csv(dataname)
#df = df.sort_values(by=[0],axis=1)
#print(df.columns)
df.columns = [int(s) for s in df.columns]
df = df.sort_index(axis='columns')
df.to_csv(dataname, header = False, index = False)
#print(df)

