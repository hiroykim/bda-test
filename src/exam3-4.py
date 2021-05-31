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
from xgboost import XGBClassifier


def print_all():
    print("X_len: ", X_len)
    print(X_all.info())
    print(X_all.describe())
    print(X_all.head())


def preprocessing():
    global X_all
    X_all= X_all['환불금액'].fillna(0)
    X_all['주구매상품','주구매지점']= X_all['주구매상품','주구매지점'].apply(LabelEncoder.fit_transform())

def main():

    preprocessing()


if __name__ == "__main__":
    pd.set_option("display.max_rows", 10)
    pd.set_option("display.max_columns", 999)

    X_train = pd.read_csv("data/X_train.csv", encoding="cp949")
    y_train = pd.read_csv("data/y_train.csv", encoding="cp949")
    X_test = pd.read_csv("data/X_test.csv", encoding="cp949")

    X_len = len(X_train)
    X_all = pd.concat([X_train, X_test], axis=0)

    main()
    print_all()
pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", 999)

X_train= pd.read_csv("data/X_train.csv", encoding='cp949')
y_train= pd.read_csv("data/y_train.csv", encoding='cp949')
X_test= pd.read_csv("data/X_test.csv", encoding='cp949')

X_train_size= len(X_train)
X_all= pd.concat([X_train, X_test], axis=0)
y_train= y_train['gender']

X_all["주구매상품"]= LabelEncoder().fit_transform(X_all["주구매상품"])
X_all["주구매지점"]= LabelEncoder().fit_transform(X_all["주구매지점"])
X_all["환불금액"]= X_all["환불금액"].fillna(0)
X_all= X_all.drop(['cust_id'], axis=1)

print(X_all)
sc= StandardScaler()
X_all= sc.fit_transform(X_all)
print(X_all.shape)

X_train= X_all[:X_train_size]
X_test= X_all[X_train_size:]
print("=================================================")
print(X_train.shape)

svc_model= SVC()
svc_model.fit(X_train, y_train)
cvs= cross_val_score(svc_model, X_train, y_train, cv=5)
print(cvs)
print(cvs.mean())
rst=svc_model.predict(X_train)
ras=roc_auc_score(y_train, rst)
print(ras)


