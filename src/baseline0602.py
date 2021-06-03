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
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
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
    X_train_all = pd.read_csv("data/X_train.csv", encoding='cp949')
    y_train_all = pd.read_csv("data/y_train.csv", encoding='cp949')
    X_test_all = pd.read_csv("data/x_test.csv", encoding='cp949')
else:
    X_train_all = pd.read_csv("data/X_train.csv")
    y_train_all = pd.read_csv("data/y_train.csv")
    X_test_all = pd.read_csv("data/x_test.csv")

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
    X_all['주구매상품']= LabelEncoder().fit_transform(X_all['주구매상품'])
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

    #model, predict

def make_model(model,X_train, y_train, X_val, y_val, model_score):

    dt_score = dict()
    model.fit(X_train, y_train)
    cvs= cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=multiprocessing.cpu_count())

    rst= model.predict(X_val)
    dt_score['acuracy']= np.round(cvs.mean(),4)
    dt_score['precision']= np.round(precision_score(y_val, rst),4)
    dt_score['recall']= np.round(recall_score(y_val, rst),4)
    dt_score['f1']= np.round(f1_score(y_val, rst),4)
    dt_score['ras']= np.round(roc_auc_score(y_val, rst),4)

    model_nm= str(type(model)).split('.')[-1][:-2]
    model_score[model_nm]= dt_score

    return model

v_rand=123
d_models = {
    "svc": SVC(probability=True, random_state=v_rand)
    , "lg": LogisticRegression( random_state=v_rand)
    , "ab": AdaBoostClassifier(n_estimators=100, random_state=v_rand)
    , "rf": RandomForestClassifier( random_state=v_rand)
    , "kn": KNeighborsClassifier()
    , "gb": GradientBoostingClassifier(n_estimators=90, max_depth=2, random_state=v_rand)
    , "bc": BaggingClassifier(n_estimators=100, random_state=v_rand)
    , "nb": GaussianNB()
    , "tc": DecisionTreeClassifier( random_state=v_rand)
            }

model_score=dict()
models_d=dict()
i=0
for kmv, model in d_models.items():
    models_d[i]= make_model(model, X_train, y_train, X_val, y_val, model_score)
    i += 1

for k, v in model_score.items():
    print("{0:>28} -> {1}".format(k, v))

sys.exit(0)
#하이퍼파라미터 최적화 GridSearchCV
#good_model= models_d[8]
good_model= vmodel_c
#print(good_model.get_params())
#print(models_d[8].get_params())
#sys.exit(0)

'''
param_grid={
    'n_estimators': [70, 80, 90]
    , 'learning_rate': [0.05, 0.1, 1.5]
    , 'max_depth': [1, 2]
}
'''
param_grid={
    'weights': [[1,1,1], [1,2,1], [1,2,2]]
}

gscv= GridSearchCV(good_model, param_grid=param_grid, cv=5, n_jobs=multiprocessing.cpu_count(), scoring='accuracy', refit=True, verbose=2)
gscv.fit(X_train, y_train)
scores = pd.DataFrame(gscv.cv_results_)
print(scores)
#print(gscv.cv_results_)
print(gscv.best_params_)
print(gscv.get_params())
print(gscv.best_score_)
good_model= gscv.best_estimator_


#제출
rst= good_model.predict_proba(X_test)
df_rst= pd.DataFrame(rst[:,1:], columns=['gender'])
df_rst_all= pd.concat([X_test_all.cust_id, df_rst.gender.map(lambda x: float("{0:.3f}".format(x)))], axis=1)
print(df_rst_all['gender'].apply(lambda x: "{0:.3f}".format(x)))
df_rst_all.to_csv("114203701.csv", index=False)
