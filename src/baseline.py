import pandas as pd
import numpy as np
import multiprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", 999)

X_train= pd.read_csv("data/X_train.csv", encoding='cp949')
y_train= pd.read_csv("data/y_train.csv", encoding='cp949')
X_test= pd.read_csv("data/X_test.csv", encoding='cp949')

X_train['gender']= y_train['gender']
X_train_size= len(X_train)
X_all= pd.concat([X_train, X_test], axis=0)
y_train= y_train['gender']
y_cust=  X_test['cust_id']

print(X_train['주구매상품'].value_counts())
print(X_train.groupby('gender')['주구매상품'].value_counts())

pd_tmp= pd.get_dummies(X_all["주구매상품"])
ag_tmp= pd.get_dummies(X_all["주구매지점"])
X_all= pd.concat([X_all, pd_tmp, ag_tmp], axis=1)
X_all["환불금액"]= X_all["환불금액"].fillna(0)
X_all= X_all.drop(['cust_id',"주구매상품","주구매지점","gender"], axis=1)

print(X_all)
sc= StandardScaler()
X_all= sc.fit_transform(X_all)
print(X_all.shape)

X_train= X_all[:X_train_size]
X_test= X_all[X_train_size:]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=11)
print("=================================================")
print(X_train.shape)

sd=dict()

def get_eval(sd, k_name, label, pred):
    new_dict = dict()
    new_dict['acurracy'] = "{0:.4f}".format(accuracy_score(label, pred))
    new_dict['precision'] = "{0:.4f}".format(precision_score(label, pred))
    new_dict['recall'] = "{0:.4f}".format(recall_score(label, pred))
    new_dict['f1'] = "{0:.4f}".format(f1_score(label, pred))
    new_dict['roc_auc_score'] = "{0:.4f}".format(roc_auc_score(label, pred))
    sd[k_name]= new_dict

svc_model= SVC(probability=True)
svc_model.fit(X_train, y_train)
cvs= cross_val_score(svc_model, X_train, y_train, scoring='accuracy' , verbose=True, cv=5, n_jobs=multiprocessing.cpu_count())
print("cross_val:", cvs)
print("cross_val:", cvs.mean())
rst= svc_model.predict(X_val)
get_eval(sd, 'svc_model', y_val, rst)

lg_model= LogisticRegression()
lg_model.fit(X_train, y_train)
cvs= cross_val_score(lg_model, X_train, y_train, scoring='accuracy' , verbose=True, cv=5, n_jobs=multiprocessing.cpu_count())
print("cross_val:", cvs)
print("cross_val:", cvs.mean())
rst= lg_model.predict(X_val)
get_eval(sd, 'lg_model', y_val, rst)

dt_model= DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
cvs= cross_val_score(dt_model, X_train, y_train, scoring='accuracy' , verbose=True, cv=5, n_jobs=multiprocessing.cpu_count())
print("cross_val:", cvs)
print("cross_val:", cvs.mean())
rst= dt_model.predict(X_val)
get_eval(sd, 'dt_model', y_val, rst)


rf_model= RandomForestClassifier()
rf_model.fit(X_train, y_train)
cvs= cross_val_score(rf_model, X_train, y_train, scoring='accuracy' , verbose=True, cv=5, n_jobs=multiprocessing.cpu_count())
print("cross_val:", cvs)
print("cross_val:", cvs.mean())
rst= rf_model.predict(X_val)
get_eval(sd, 'rf_model', y_val, rst)

kn_model= KNeighborsClassifier()
kn_model.fit(X_train, y_train)
cvs= cross_val_score(kn_model, X_train, y_train, scoring='accuracy' , verbose=True, cv=5, n_jobs=multiprocessing.cpu_count())
print("cross_val:", cvs)
print("cross_val:", cvs.mean())
rst= kn_model.predict(X_val)
get_eval(sd, 'kn_model', y_val, rst)

gn_model= GaussianNB()
gn_model.fit(X_train, y_train)
cvs= cross_val_score(gn_model, X_train, y_train, scoring='accuracy', verbose=True, cv=5, n_jobs=multiprocessing.cpu_count())
print("cross_val:", cvs)
print("cross_val:", cvs.mean())
rst= gn_model.predict(X_val)
get_eval(sd, 'gn_model', y_val, rst)

xgb_model= XGBClassifier()
xgb_model.fit(X_train, y_train)
cvs= cross_val_score(xgb_model, X_train, y_train, scoring='accuracy', verbose=True, cv=5, n_jobs=multiprocessing.cpu_count())
print("cross_val:", cvs)
print("cross_val:", cvs.mean())
rst= xgb_model.predict(X_val)
get_eval(sd, 'xgb_model', y_val, rst)

for k,v in sd.items():
    print("{0}->{1}".format(k, v))

model= xgb_model
print("=================================================")

rst_p= model.predict_proba(X_test)
y_test= pd.DataFrame(rst_p, columns=['del','gender'])
y_test= y_test['gender']
y_test= pd.concat([y_cust, y_test], axis=1)
y_test['gender']= y_test['gender'].map(lambda x: '%0.3f'%x)
print(y_test)

y_test.to_csv("114203701.csv", index=False)