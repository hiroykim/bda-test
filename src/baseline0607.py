import sys
import pandas as pd
import numpy as np
import sklearn

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier

pd.set_option("display.max_rows", 99)
pd.set_option("display.max_columns", 999)

X_train = pd.read_csv("data/X_train.csv", encoding='cp949')
X_test = pd.read_csv("data/X_test.csv", encoding='cp949')
y_train = pd.read_csv("data/y_train.csv", encoding='cp949')

if 0:
    print(X_train)
    print(X_test)
    print(y_train)

X_all=pd.concat([X_train, y_train['gender']], axis=1)
X_all=pd.concat([X_all, X_test])
y_train = y_train['gender']
y_test= X_test['cust_id']

if 0:
    print(X_all)
    print(X_all.info())
    print(X_all.describe(include='O'))

#missing value
X_all['환불금액']= X_all['환불금액'].fillna(0)

#object encoder
prd_tmp= pd.get_dummies(X_all['주구매상품'])
agy_tmp= pd.get_dummies(X_all['주구매지점'])
X_all= pd.concat([prd_tmp, agy_tmp, X_all], axis=1)
X_all= X_all.drop(['주구매상품','주구매지점'], axis=1)


#Scaling
X_all = pd.DataFrame(StandardScaler().fit_transform(X_all), columns=X_all.columns)

if 0:
    print(X_all)
    print(X_all.info())
    print(X_all.describe())

#feature selection
X_all = X_all.drop(['cust_id'], axis=1)

#train, valid set
X_train_size = len(X_train)
X_train = X_all.iloc[:X_train_size, :-1]
X_test  = X_all.iloc[X_train_size:, :-1]

if 1:
    print(X_train)
    print(X_test)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

#model-ensemble
def make_model(model, X_train, y_train, dt_scores, model_name):
    dt_score=dict()
    model.fit(X_train, y_train)
    cv_scores= cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    predicts= model.predict(X_val)

    dt_score['cv_mean']= np.round(cv_scores.mean(),3)
    dt_score['accuracy']= np.round(accuracy_score(y_val, predicts),3)
    dt_score['precision']= np.round(precision_score(y_val, predicts),3)
    dt_score['recall']= np.round(recall_score(y_val, predicts),3)
    dt_score['f1']= np.round(f1_score(y_val, predicts),3)
    dt_score['rocauc']= np.round(roc_auc_score(y_val, predicts),3)

    dt_scores[model_name]= dt_score

    return model

dt_scores=dict()
model = make_model(RandomForestClassifier(n_estimators=100, max_depth=3, random_state=123), X_train, y_train, dt_scores, "RandomForestClassifier")
model = make_model(GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=123), X_train, y_train, dt_scores, "GradientBoostingClassifier")
for k, v in dt_scores.items():
    print("{0:30}:{1}".format(k, v))

#hyper-param
param_grid={
    "n_estimators": [50,100,150]
    ,"max_depth": [2,3,4]
}

gscv = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, refit=True, scoring="accuracy")
gscv.fit(X_train, y_train)
cv_results= pd.DataFrame(gscv.cv_results_)
print(cv_results)
cv_model=gscv.best_estimator_
cv_score=gscv.best_score_
cv_param=gscv.best_params_
print(cv_model)
print(cv_score)
print(cv_param)

predicts=pd.DataFrame(cv_model.predict_proba(X_test)[:,-1], columns=['gender'])
y_test= pd.concat([y_test, predicts['gender'].map(lambda x: "{0:.3f}".format(x))], axis=1)
print(y_test)
y_test.to_csv("114203701.csv", index=False)