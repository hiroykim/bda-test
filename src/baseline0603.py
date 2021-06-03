import sys
import traceback
import pandas as pd
import numpy as np
import multiprocessing
import platform

import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 20)

if platform.system() == "Windows":
    X_train_all = pd.read_csv("data/X_train.csv", encoding='cp949')
    y_train_all = pd.read_csv("data/y_train.csv", encoding='cp949')
    X_test_all = pd.read_csv("data/x_test.csv", encoding='cp949')
else:
    X_train_all = pd.read_csv("data/X_train.csv")
    y_train_all = pd.read_csv("data/y_train.csv")
    X_test_all = pd.read_csv("data/x_test.csv")

#EDA
print(X_train_all.info())
print(X_test_all.info())
print(y_train_all.info())


