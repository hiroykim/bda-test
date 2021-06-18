import pandas as pd
import numpy as np
import sklearn

from sklearn import cluster
from sklearn import datasets
from sklearn.datasets import load_iris

import traceback

'''
iris = load_iris()

model = cluster.KMeans(n_clusters=3)
model.fit(iris.data)
predict = model.predict(iris.data)
print(predict)
print(iris.target)

idx = np.where(predict==0)
print(iris.target[idx])
idx = np.where(predict==1)
print(iris.target[idx])
idx = np.where(predict==2)
print(iris.target[idx])
'''

help(sklearn.cluster)
