import os
import sys
import traceback
import pandas as pd
import numpy as np
import sklearn

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, Binarizer
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

pd.set_option("max_rows", 20)
pd.set_option("max_columns", 20)

X_train_data  = pd.read_csv("data/X_train.csv", encoding='cp949')
X_test_data  = pd.read_csv("data/X_test.csv", encoding='cp949')
y_train_data  = pd.read_csv("data/y_train.csv", encoding='cp949')

X_all = pd.concat([X_test_data, y_train_data.gender], axis=1)
X_all = pd.concat([X_all, X_test_data], axis=0)

# missing value
X_all['환불금액'] = X_all['환불금액'].fillna(0)

# One Hot Encoding
df_pd = pd.get_dummies(X_all.주구매상품)
df_ag = pd.get_dummies(X_all.주구매지점)
X_all = pd.concat([df_pd, df_ag, X_all], axis=1)

# Scaler
X_all = X_all.drop(['주구매상품', '주구매지점', 'cust_id'], axis=1)
X_all = pd.DataFrame( StandardScaler().fit_transform(X_all), columns=X_all.columns)

print(X_all.info())
print(X_all.describe())
print(X_all.head())


