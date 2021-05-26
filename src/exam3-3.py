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