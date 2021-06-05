import sys
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 999)

X_train= pd.read_csv("data/X_train.csv", encoding='cp949')
y_train= pd.read_csv("data/y_train.csv", encoding='cp949')
X_test= pd.read_csv("data/X_test.csv", encoding='cp949')


X_train['gender']= y_train['gender']
print(y_train.info())
print(X_train.info())
X_test_size= len(X_test)
X_all= pd.concat([X_train, X_test], axis=0)

y_cust=  X_test['cust_id']

print(X_all.info())
print(X_all.describe(include='O'))
print(X_all.head())
print("-----------------------------------------------------------------------------------")

if 0:
    X_all= X_all.drop(['환불금액','cust_id'],axis=1)
else:
    X_all['환불금액'] = X_all['환불금액'].fillna(0)
    X_all= X_all.drop(['cust_id'],axis=1)

if 0:
    X_all = X_all[X_all['총구매액'] > 0]
    X_all = X_all[X_all['최대구매액'] > 0]
    X_all['총구매액']= np.log1p(X_all['총구매액'])
    X_all['최대구매액']= np.log1p(X_all['최대구매액'])
    X_all['구매주기']= np.log1p(X_all['구매주기'])

if 1:
    pd_tmp = pd.get_dummies(X_all['주구매상품'])
    ag_tmp = pd.get_dummies(X_all['주구매지점'])
    X_all = pd.concat([pd_tmp, ag_tmp,X_all], axis=1)
    X_all = X_all.drop(['주구매상품', '주구매지점'], axis=1)
else:
    X_all['주구매상품'] = LabelEncoder().fit_transform(X_all['주구매상품'])
    X_all['주구매지점'] = LabelEncoder().fit_transform(X_all['주구매지점'])

if 1:
    columns= list(X_all.columns)
    columns.remove("gender")
    X_all[columns]= pd.DataFrame(MinMaxScaler().fit_transform(X_all[columns]), columns=columns)

if 1:
    print(X_all.info())
    print(X_all.describe())
    print(X_all.head())
    #sys.exit(0)


X_train_size = len(X_all)- len(X_test)
print("Size : ", len(X_all),len(X_test) ,X_train_size)
X_train= X_all.iloc[:X_train_size, :-1]
X_test= X_all.iloc[X_train_size:, :-1]
y_train=X_all.iloc[:X_train_size, -1]
y_train= y_train.astype('int64')

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)
print("=================================================")

sd=dict()

def predict_model(model, X_train, y_train,X_val):
    model.fit(X_train, y_train)
    cvs= cross_val_score(model, X_train, y_train, cv=5, n_jobs=multiprocessing.cpu_count(), scoring='accuracy')
    rst= model.predict(X_val)
    return model, cvs, rst


def get_eval(sd, k_name, label, pred, cvs_mean):
    new_dict = dict()
    new_dict['cvs_mean'] = "{0:.4f}".format(cvs_mean)
    new_dict['acurracy'] = "{0:.4f}".format(accuracy_score(label, pred))
    new_dict['precision'] = "{0:.4f}".format(precision_score(label, pred))
    new_dict['recall'] = "{0:.4f}".format(recall_score(label, pred))
    new_dict['f1'] = "{0:.4f}".format(f1_score(label, pred))
    new_dict['roc_auc_score'] = "{0:.4f}".format(roc_auc_score(label, pred))
    sd[k_name]= new_dict


svc_model, cvs, rst= predict_model(SVC(probability=True, random_state=123), X_train, y_train, X_val)
get_eval(sd, 'svc_model', y_val, rst, cvs.mean())

lg_model, cvs, rst= predict_model(LogisticRegression(random_state=123), X_train, y_train, X_val)
get_eval(sd, 'lg_model', y_val, rst, cvs.mean())

dt_model, cvs, rst= predict_model(DecisionTreeClassifier(random_state=123), X_train, y_train, X_val)
get_eval(sd, 'dt_model', y_val, rst, cvs.mean())

rf_model, cvs, rst= predict_model(RandomForestClassifier(random_state=123), X_train, y_train, X_val)
get_eval(sd, 'rf_model', y_val, rst, cvs.mean())

kn_model, cvs, rst= predict_model(KNeighborsClassifier(), X_train, y_train, X_val)
get_eval(sd, 'kn_model', y_val, rst, cvs.mean())

gn_model, cvs, rst= predict_model(GaussianNB(), X_train, y_train, X_val)
get_eval(sd, 'gn_model', y_val, rst, cvs.mean())

gb_model, cvs, rst= predict_model(GradientBoostingClassifier(n_estimators=100, random_state=123), X_train, y_train, X_val)
get_eval(sd, 'gb_model', y_val, rst, cvs.mean())


#GridSearchCV
param_grid={
    "n_estimators": [40, 50, 60]
    , "max_depth":[3, 4, 5]
    , "max_features": ['auto', None]
}

gscv = GridSearchCV(gb_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring="roc_auc", refit=True)
gscv.fit(X_train, y_train)
result= pd.DataFrame(gscv.cv_results_)
print(result)
print(gscv.best_score_)
print(gscv.best_params_)
good_model = gscv.best_estimator_
good_model, cvs, rst= predict_model(good_model, X_train, y_train, X_val)
get_eval(sd, 'good_model', y_val, rst, cvs.mean())

for k,v in sd.items():
    print("{0}->{1}".format(k, v))

model= good_model
print("=================================================")

rst_p= model.predict_proba(X_test)
y_test= pd.DataFrame(rst_p, columns=['del','gender'])
y_test= y_test['gender']
y_test= pd.concat([y_cust, y_test], axis=1)
y_test['gender']= y_test['gender'].map(lambda x: '%0.3f'%x)
print(y_test)

y_test.to_csv("114203701.csv", index=False)