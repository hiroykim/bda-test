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

pd.set_option("display.max_rows", 99)
pd.set_option("display.max_columns", 999)

if platform.system() == "Windows":
    X_train_all = pd.read_csv("data/X_train.csv", encoding='cp949')
    y_train_all = pd.read_csv("data/y_train.csv", encoding='cp949')
    X_test_all = pd.read_csv("data/X_test.csv", encoding='cp949')
else:
    X_train_all = pd.read_csv("data/X_train.csv")
    y_train_all = pd.read_csv("data/y_train.csv")
    X_test_all = pd.read_csv("data/X_test.csv")

#INFO
if 0:
    print(X_train_all.info())
    print(X_test_all.info())
    print(y_train_all.info())

#DATA edit
X_train_all_size = len(X_train_all)
X_train_all['gender']=y_train_all['gender']
y_train_label=y_train_all['gender']
X_all = pd.concat([X_train_all, X_test_all], axis=0)
if 0:
    print(X_all.info())
    print(X_all.head())
    print(X_all.tail())

#EDA
if 0:
    print(X_all.groupby('gender')['주구매상품'].value_counts())
    print(X_all.groupby('gender')['주구매지점'].value_counts())

#feature selection
X_all=X_all.drop(['gender', 'cust_id'], axis=1)

#결측치
if 0:
    X_all['환불금액']= X_all['환불금액'].fillna(0)
else:
   X_all= X_all.drop(['환불금액'], axis=1)

if 0:
    print(X_all.head())

#Encoding
if 1:
    pd_tmp = pd.get_dummies(X_all['주구매상품'])
    ag_tmp = pd.get_dummies(X_all['주구매지점'])
    X_all = pd.concat([X_all, pd_tmp, ag_tmp], axis=1)
    X_all = X_all.drop(['주구매상품', '주구매지점'], axis=1)
else:
    X_all['주구매상품'] = LabelEncoder().fit_transform(X_all['주구매상품'])
    X_all['주구매지점'] = LabelEncoder().fit_transform(X_all['주구매지점'])

if 0:
    print(X_all.head())

#Scaling
x_columns= X_all.columns
if 1:
    X_all= pd.DataFrame(StandardScaler().fit_transform(X_all), columns=x_columns)
else:
    X_all = pd.DataFrame(MinMaxScaler().fit_transform(X_all), columns=x_columns)

if 1:
    print(X_all.info())
    print(X_all.head())

#model_selection
X_train_all= X_all.iloc[:X_train_all_size,:]
X_test= X_all.iloc[X_train_all_size:,:]

v_rand=123

X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_label, test_size=0.2, random_state=v_rand)
print(y_train_label)
#make_predict_eval func

d_scores = dict()
d_models = {
    "svc": SVC(probability=True,  random_state=v_rand)
    , "lg": LogisticRegression( random_state=v_rand)
    , "ab": AdaBoostClassifier(n_estimators=100, random_state=v_rand)
    , "rf": RandomForestClassifier( random_state=v_rand)
    , "kn": KNeighborsClassifier()
    , "gb": GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=v_rand)
    , "bc": BaggingClassifier(n_estimators=100, random_state=v_rand)
    , "nb": GaussianNB()
    , "tc": DecisionTreeClassifier( random_state=v_rand)
    }

def model_predict(model, X_train, y_train, X_val, y_val, d_scores, mkey):
    model.fit(X_train, y_train)
    cvs= cross_val_score(model, X_train, y_train, cv=5, n_jobs=multiprocessing.cpu_count(), scoring='accuracy')
    predicts= model.predict(X_val)
    d_score=dict()
    d_score['accuracy']= np.round(cvs.mean(), 3)
    d_score['precision']= np.round(precision_score(y_val, predicts), 3)
    d_score['recall']= np.round(recall_score(y_val, predicts), 3)
    d_score['f1']= np.round(f1_score(y_val, predicts), 3)
    d_score['ras']= np.round(roc_auc_score(y_val, predicts), 3)
    d_scores[mkey]=d_score

    return model

# ensemble -> bagging, boosting, voting
if 1:
    for mkey, model in d_models.items():
        model_predict(model, X_train, y_train, X_val, y_val, d_scores, mkey)

v_model= VotingClassifier(estimators=[('gb',d_models['gb']), ('ab',d_models['ab'])], weights=[1,1], voting='soft')
v_model= model_predict(v_model, X_train, y_train, X_val, y_val, d_scores, 'vt')

for k, v in d_scores.items():
    print(k, v)

param_grid={
    'weights':[[1,1], [2,1], [1,2]]
}

# optimize hyper_params
gscv= GridSearchCV(v_model, param_grid=param_grid, cv=5, n_jobs=multiprocessing.cpu_count(), verbose=1, refit=True, scoring='accuracy')
gscv.fit(X_train, y_train)
scores= pd.DataFrame(gscv.cv_results_)
print(scores)
print(gscv.best_params_)
print(gscv.best_score_)
good_model= gscv.best_estimator_
print(good_model)

#submit predicts
predicts= pd.DataFrame(good_model.predict_proba(X_test)[:, 1:], columns=['gender'])
df_pred = pd.concat([X_test_all['cust_id'], predicts.apply(lambda x: np.round(x,3))], axis=1)
print(df_pred)
df_pred.to_csv("20210619.csv", index=False)
