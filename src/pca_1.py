import pandas as pd
import numpy as np
import sklearn

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, FactorAnalysis

iris, labels = load_iris(return_X_y=True)
print(iris)
print(iris.shape)
print(labels)
print(labels.shape)

model = PCA(n_components=2, random_state=1)
model.fit(iris)
df_res = model.transform(iris)
print(df_res)
print(df_res.shape)

model = FactorAnalysis(n_components=3, random_state=1)
model.fit(iris)
df_res = model.transform(iris)
print(df_res)
print(df_res.shape)

help(sklearn.decomposition)