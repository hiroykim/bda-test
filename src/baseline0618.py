import os
import traceback
import pandas as pd
import numpy as np
import sklearn

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier,\
    GradientBoostingRegressor, GradientBoostingClassifier, VotingClassifier, VotingRegressor, StackingRegressor, StackingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA, SparsePCA

pd.set_option("display.max_rows", 99)
pd.set_option("display.max_columns", 99)

#load data
X_train_data = pd.read_csv("data/X_train.csv", encoding='cp949')
y_train_data = pd.read_csv("data/y_train.csv", encoding='cp949')
X_test_data = pd.read_csv("data/X_test.csv", encoding='cp949')

# print(X_train_data)
# print(y_train_data)
# print(X_test_data)

X_all = pd.concat([X_train_data, y_train_data['gender']], axis=1)
X_all = pd.concat([X_all, X_test_data], axis=0)
#X_all = X_all.reset_index(drop=True)

#missing value check, 환불금액
print(X_all)
print(X_all.info())
print(X_all.describe())
print(X_all.describe(include='O'))

X_all['환불금액'] = X_all.환불금액.fillna(0)

#one-hot-encoding 주구매상품, 주구매지점
pd_hot = pd.get_dummies(X_all['주구매상품'])
ag_hot = pd.get_dummies(X_all['주구매지점'])
X_all = pd.concat([X_all, pd_hot, ag_hot], axis=1)
X_all = X_all.drop(['주구매상품', '주구매지점'], axis=1)


#feature selection
X_all = X_all.drop(['cust_id', 'gender'], axis=1)


#scaling
X_all = pd.DataFrame(StandardScaler().fit_transform(X_all), columns=X_all.columns)
print(X_all)
print(X_all.info())

#train_valid_test_data setting
X_train_all = X_all.iloc[:len(X_train_data), :]
X_test = X_all.iloc[len(X_train_data):, :]
y_train_all = y_train_data['gender']
print(X_train_all)
print(y_train_all)
print(X_test)

X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=1)

dt_scores = dict()

'''
def make_model(model, X_train, y_train, X_val, y_val, dt_scores, model_nm):
    dt_score= dict()
    model.fit(X_train, y_train)
    cvs = cross_val_score(model, X_train, y_train, cv=10, n_jobs=-1, scoring="accuracy")
    predicts = model.predict(X_val)
    dt_score["accuary"] = "{0:.3f}".format(cvs.mean())
    dt_score["precision_score"] = "{0:.3f}".format(precision_score(y_val, predicts))
    dt_score["recall_score"] = "{0:.3f}".format(recall_score(y_val, predicts))
    dt_score["f1_score"] = "{0:.3f}".format(f1_score(y_val, predicts))
    dt_score["roc_auc_score"] = "{0:.3f}".format(roc_auc_score(y_val, predicts))
    dt_scores[model_nm]= dt_score

    return model

model = make_model(LogisticRegression(), X_train, y_train, X_val, y_val, dt_scores, 'lrr')
model = make_model(SVC(), X_train, y_train, X_val, y_val, dt_scores, 'svc')
model = make_model(RandomForestClassifier(), X_train, y_train, X_val, y_val, dt_scores, 'rfc')
model = make_model(GaussianNB(), X_train, y_train, X_val, y_val, dt_scores, 'gnb')
model = make_model(DecisionTreeClassifier(), X_train, y_train, X_val, y_val, dt_scores, 'dtc')
model = make_model(AdaBoostClassifier(), X_train, y_train, X_val, y_val, dt_scores, 'abc')
model = make_model(GradientBoostingClassifier(), X_train, y_train, X_val, y_val, dt_scores, 'gbc')
for k, v in dt_scores.items():
    print(k, v)
'''

def make_model_r(model, X_train, y_train, X_val, y_val, dt_scores, model_nm):
    dt_score= dict()
    model.fit(X_train, y_train)
    predicts = model.predict(X_val)
    dt_score["r2"] = "{0:.3f}".format(r2_score(y_val, predicts))
    dt_score["mse"] = "{0:.3f}".format(mean_squared_error(y_val, predicts))
    dt_scores[model_nm] = dt_score

    return model

model = make_model_r(LogisticRegression(), X_train, y_train, X_val, y_val, dt_scores, 'lrr')
model = make_model_r(SVR(), X_train, y_train, X_val, y_val, dt_scores, 'svr')
model = make_model_r(RandomForestRegressor(), X_train, y_train, X_val, y_val, dt_scores, 'rfr')
model = make_model_r(GaussianNB(), X_train, y_train, X_val, y_val, dt_scores, 'gnb')
model = make_model_r(DecisionTreeRegressor(), X_train, y_train, X_val, y_val, dt_scores, 'dtr')
model = make_model_r(GradientBoostingRegressor(), X_train, y_train, X_val, y_val, dt_scores, 'gbr')
model = make_model_r(StackingRegressor(estimators=[("1", LinearRegression()), ("2", SVR())], final_estimator=AdaBoostRegressor()), X_train, y_train, X_val, y_val, dt_scores, 'str')
model = make_model_r(AdaBoostRegressor(), X_train, y_train, X_val, y_val, dt_scores, 'abr')
for k, v in dt_scores.items():
    print(k, v)

# hyper-params optimization
print(model.get_params().keys())
print(sklearn.metrics.SCORERS.keys())
param_grid={
    "n_estimators" : [50,100,150]
    , "learning_rate" : [0.1,0.2]
    }
gscv = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, refit=True, scoring="neg_mean_squared_error")
gscv.fit(X_train, y_train)
df_cv_result = pd.DataFrame(gscv.cv_results_)
print(df_cv_result)
print(gscv.best_score_)
print(gscv.best_params_)
print(gscv.best_estimator_)


# submit result
good_model = model
df_predicts = pd.DataFrame(good_model.predict(X_test), columns=['gender'])
df_predicts['cust_id'] = X_test_data['cust_id']
df_predicts.gender = df_predicts.gender.map(lambda x : "{0:.3f}".format(x))
df_predicts = df_predicts[['cust_id','gender']]
print(df_predicts)
df_predicts.to_csv("114203701.csv", index=False)