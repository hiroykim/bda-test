import sys
import traceback
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import warnings
import platform

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", 999)

if platform.system() == "Windows":
    X_train_all= pd.read_csv("data/X_train.csv", encoding='cp949')
    y_train_all= pd.read_csv("data/y_train.csv", encoding='cp949')
    X_test_all= pd.read_csv("data/X_test.csv", encoding='cp949')
else:
    X_train_all= pd.read_csv("data/X_train.csv")
    y_train_all= pd.read_csv("data/y_train.csv")
    X_test_all= pd.read_csv("data/X_test.csv")

#EDA
X_train_all['gender']= y_train_all['gender']

#Data처리
X_train_all_size= len(X_train_all)
X_all= pd.concat([X_train_all, X_test_all], axis=0)
y_train_all_one= y_train_all['gender']

#결측치 처리
X_all['환불금액']= X_all.환불금액.fillna(0)

#object 처리
df_pd= pd.get_dummies(X_all['주구매상품'])
df_ag= pd.get_dummies(X_all['주구매지점'])
X_all= pd.concat([X_all, df_pd, df_ag], axis=1)
X_all= X_all.drop(['cust_id', 'gender', '주구매상품', '주구매지점'] , axis=1)

#Scaling 처리
sc= StandardScaler()
X_all_np= sc.fit_transform(X_all)
X_all= pd.DataFrame(X_all_np, columns=X_all.columns)

#data 분류
X_train_all= X_all.iloc[:X_train_all_size,:]
X_test= X_all.iloc[X_train_all_size:, :]

try:
    X_train, X_val, y_train, y_val= train_test_split(X_train_all, y_train_all_one, train_size=0.8, random_state=123)
except Exception:
    print(traceback.format_exc())

    #model, predict

def make_model(model,X_train, y_train, X_val, y_val, model_nm, model_score):

    dt_score = dict()
    model.fit(X_train, y_train)
    cvs= cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=multiprocessing.cpu_count())

    rst= model.predict(X_val)
    dt_score['acuracy']= "{0:.4f}".format(cvs.mean())
    dt_score['precision']= "{0:.4f}".format(precision_score(y_val, rst))
    dt_score['recall']= "{0:.4f}".format(recall_score(y_val, rst))
    dt_score['f1']= "{0:.4f}".format(f1_score(y_val, rst))
    dt_score['ras']= "{0:.4f}".format(roc_auc_score(y_val, rst))

    model_score[model_nm]= dt_score

    return model


model_score=dict()
if 0:
    model= make_model(SVC(probability=True, random_state=123), X_train, y_train, X_val, y_val, "svc", model_score)
    model= make_model(DecisionTreeClassifier(random_state=123), X_train, y_train, X_val, y_val, "dtc", model_score)
    model= make_model(RandomForestClassifier(random_state=123), X_train, y_train, X_val, y_val, "rfc", model_score)
    model= make_model(GaussianNB(), X_train, y_train, X_val, y_val, "gnb", model_score)
    model= make_model(KNeighborsClassifier(), X_train, y_train, X_val, y_val, "knf", model_score)
    make_model(LogisticRegression(random_state=123), X_train, y_train, X_val, y_val, "lgr", model_score)
    model= make_model(XGBClassifier(random_state=123), X_train, y_train, X_val, y_val, "xgb", model_score)
    model = make_model(BaggingClassifier(DecisionTreeClassifier(random_state=123)), X_train, y_train, X_val, y_val,"bag", model_score)

model= make_model(XGBClassifier(random_state=123), X_train, y_train, X_val, y_val, "xgb", model_score)

print(model)

for k, v in model_score.items():
    print("{0} -> {1}".format(k, v))

#하이퍼파라미터 최적화 GridSearchCV
'''
param_grid={
    'booster':['gbtree']
    ,'max_depth':[5,6,8]
    ,'min_child_weight': [1,3,5]
    , 'gamma': [0,1,2,3]
    , 'nthread': [4]
}
#{'booster': 'gbtree', 'gamma': 3, 'max_depth': 5, 'min_child_weight': 5, 'nthread': 4}

gscv= GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=multiprocessing.cpu_count(), scoring='accuracy', refit=True)
gscv.fit(X_train, y_train)
#scores = pd.DataFrame(gscv.cv_results_)
#print(scores)
print(gscv.cv_results_)
print(gscv.best_params_)
print(gscv.best_score_)
model= gscv.best_estimator_
new_score=dict()
model= make_model(model, X_train, y_train, X_val, y_val, "xgb", new_score)
for k, v in new_score.items():
    print("{0} -> {1}".format(k, v))
'''

#제출
rst= model.predict_proba(X_test)
df_rst= pd.DataFrame(rst[:,1:], columns=['gender'])
df_rst_all= pd.concat([X_test_all.cust_id, df_rst.gender.map(lambda x: float("{0:.3f}".format(x)))], axis=1)
print(df_rst_all['gender'].apply(lambda x: "{0:.3f}".format(x)))
df_rst_all.to_csv("114203701.csv", index=False)

