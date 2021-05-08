import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("data/mtcars.csv")
#print(df.head())

minmax_scaler = preprocessing.MinMaxScaler().fit(df[['qsec']] )
df_minmax = minmax_scaler.transform(df[["qsec"]])
print(df_minmax[df_minmax>0.5].shape[0])