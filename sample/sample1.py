import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("data/mtcars.csv")
#print(df.head())

minmax_scaler = preprocessing.MinMaxScaler().fit(df[['qsec']] )
print(type(minmax_scaler))
df_minmax = minmax_scaler.transform(df[["qsec"]])
print(type(df_minmax))
print(df_minmax[df_minmax>0.5].shape[0])

minmax_scaler2 = preprocessing.MinMaxScaler().fit_transform(df[['qsec']] )
print(type(minmax_scaler2))
print(minmax_scaler2[minmax_scaler2>0.5].shape[0])
