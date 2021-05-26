import pandas as pd

pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", 999)

X_train = pd.read_csv("data/X_train.csv", encoding='cp949')
y_train = pd.read_csv("data/y_train.csv", encoding='cp949')
X_test = pd.read_csv("data/X_test.csv", encoding='cp949')

print(X_train.info())
print(X_train.describe())
print(X_train.describe(include='O'))
print(X_train.head())

tmp_prd = pd.get_dummies(X_train['주구매상품'])
tmp_agy = pd.get_dummies(X_train['주구매지점'])
print(tmp_prd)
print(tmp_agy)

X_train= pd.concat([X_train, tmp_prd, tmp_agy], axis=1)
print(X_train)

X_train= X_train.drop(["cust_id", "주구매상품", "주구매지점"], axis=1)
print(X_train)

X_train['환불금액']= X_train['환불금액'].fillna(0)
print(X_train['환불금액'].isnull().any())

'''
print(X_train['내점일수'])
print(X_train['내점일수'].map(lambda x : x*2))
print(X_train[['내점일수','구매주기']].apply(lambda x : x*4))
'''

from sklearn.preprocessing import StandardScaler, MinMaxScaler

sc= StandardScaler()
X_train= sc.fit_transform(X_train)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(X_train.shape)

y_train= y_train['gender']
print(y_train.shape)

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=123)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

model= XGBClassifier()
model.fit(X_train, y_train)

mscore= model.score(X_val, y_val)
k_f= KFold(n_splits=5, shuffle=True, random_state=123)
scores= cross_val_score(model, X_train, y_train, cv=k_f)
print(mscore)
print(scores)

from sklearn.metrics import roc_auc_score

predict = model.predict(X_val)
rascore = roc_auc_score(y_val, predict)
print(rascore)