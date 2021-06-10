import pandas as pd
import numpy as np
import sys
import sklearn

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, Binarizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

pd.set_option("display.max_rows", 99)
pd.set_option("display.max_columns", 999)

X_train= pd.read_csv("data/X_train.csv", encoding='cp949')
X_test= pd.read_csv("data/X_test.csv", encoding='cp949')

X = X_train.iloc[:10,:5]
print(X)
print("----------------")
print(X.sort_values(by="총구매액"))
print("----------------")
print(X.groupby('주구매상품')["총구매액"].sum())
print(X.groupby('주구매상품').총구매액.agg([sum, max, min, len]))

print(X.총구매액.quantile([0.5,1]))

print("----------------")
Bin = Binarizer(threshold=3000000)
print(Bin.fit_transform(X.총구매액))