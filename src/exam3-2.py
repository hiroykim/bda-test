import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def read_csv():
    global X_train, y_train, X_test
    X_train = pd.read_csv("data/X_train.csv", encoding="cp949")
    y_train = pd.read_csv("data/y_train.csv", encoding="cp949")
    X_test = pd.read_csv("data/X_test.csv", encoding="cp949")

    return X_train, y_train, X_test


def print_data():
    print(X_train.info())
    print(X_train.describe())
    print(y_train.info())
    print(y_train.describe())


def test_model(model):
    print("info : ", model.__class__)
    print("info : ", model.__str__)
    print("info : ", model.__module__)

def test():
    print(X_train.loc[:, '환불금액'])
    print(X_train.iloc[:, 3])
    print(pd.get_dummies(X_train['주구매상품']))
    X_train['환불금액'] = X_train['환불금액'].fillna(0)
    X_test['환불금액'] = X_test['환불금액'].fillna(0)
    print(X_train)

def preprocessing_all_2(X_all, X_test):
    X_all['환불금액']= X_all['환불금액'].fillna(0)
    X_test['환불금액'] = X_test['환불금액'].fillna(0)

    df_dum_pd= pd.get_dummies(X_all['주구매상품'])
    df_dum_loc= pd.get_dummies(X_all['주구매지점'])
    X_all= pd.concat([X_all, df_dum_pd, df_dum_loc], axis=1)
    X_all= X_all.drop(['cust_id','주구매상품','주구매지점'], axis=1)

    df_dum_pd2= pd.get_dummies(X_test['주구매상품'])
    df_dum_loc2= pd.get_dummies(X_test['주구매지점'])
    X_test= pd.concat([X_test, df_dum_pd2, df_dum_loc2], axis=1)
    X_test= X_test.drop(['cust_id','주구매상품','주구매지점'], axis=1)

    sc= StandardScaler()
    sc.fit(X_all)
    X_all= sc.transform(X_all)
    sc.fit(X_test)
    X_test= sc.transform(X_test)

    return X_all, X_test


def preprocessing_all(X_all):
    X_all['환불금액']= X_all['환불금액'].fillna(0)

    df_dum_pd= pd.get_dummies(X_all['주구매상품'])
    df_dum_loc= pd.get_dummies(X_all['주구매지점'])
    X_all= pd.concat([X_all, df_dum_pd, df_dum_loc], axis=1)
    X_all= X_all.drop(['cust_id','주구매상품','주구매지점'], axis=1)

    sc= MinMaxScaler()
    X_all= sc.fit_transform(X_all)

    return X_all


def preprocessing(X_all):
    X_all['환불금액']= X_all['환불금액'].fillna(0)

    df_dum_pd= pd.get_dummies(X_all['주구매상품'])
    df_dum_loc= pd.get_dummies(X_all['주구매지점'])
    X_all= pd.concat([X_all, df_dum_pd, df_dum_loc], axis=1)
    X_all= X_all.drop(['cust_id','주구매상품','주구매지점','최대구매액'], axis=1)

    #sc= MinMaxScaler()
    sc = StandardScaler()
    X_all= sc.fit_transform(X_all)

    return X_all


def make_model(model):
    print("------------------name :", model.__class__)
    model.fit(X_train, y_train)
    print("model score :", model.score(X_train, y_train))
    scores= cross_val_score(model, X_train, y_train, cv=k_f)
    print("cross val score :", scores, np.mean(scores), np.std(scores))
    predict= model.predict(X_train)
    print("roc_auc_score :", roc_auc_score(y_train, predict))


if __name__ == "__main__":
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 15)

    X_train, y_train, X_test = read_csv()

    if 0:
        print_data()
    if 0:
        test()

    #test_model(LogisticRegression())

    if 1:
        X_train= preprocessing_all(X_train)
        X_test = preprocessing_all(X_test)
    else:
        X_train, X_test = preprocessing_all_2(X_train, X_test)

    y_train= y_train.loc[:, 'gender']
    k_f = KFold(n_splits=5, shuffle=True, random_state=123)

    '''
    make_model(LogisticRegression())
    make_model(SVC())
    make_model(RandomForestClassifier())
    make_model(DecisionTreeClassifier())
    make_model(GaussianNB())
    make_model(KNeighborsClassifier())
    

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    
    tuned_parameters= [{'C': [1.2, 1.1, 1.0, 0.9, 0.8]}]

    gscv = GridSearchCV(estimator=SVC(), param_grid=tuned_parameters, n_jobs=multiprocessing.cpu_count(), cv=k_f)
    gscv.fit(X_train, y_train)
    print(gscv.best_params_)
    print(gscv.best_estimator_)
    '''

    make_model(SVC())
    #make_model(SVC(C=0.8))
    #make_model(SVC(C=0.8, kernel='linear'))


