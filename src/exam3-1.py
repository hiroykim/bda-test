import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression


def print_data(cnt):
    print("<<%d cha>>---info--"%cnt)
    print(X_train.info())
    print(X_train.head())


def init():
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 15)
    #np.set_printoptions(threshold=sys.maxsize)


def read_files():
    X_tr = pd.read_csv("data/X_train.csv", encoding='cp949')
    y_tr = pd.read_csv("data/y_train.csv", encoding='cp949')
    return X_tr, y_tr


def add_columns():
    print("---ADD 환불여부--")
    for idx in range(0, X_total):
        if int(X_train.loc[idx,['환불금액']]>0) :
            X_train.loc[idx, ['환불여부']] = 1
        else:
            X_train.loc[idx, ['환불여부']] = 0
    print(X_train.loc[:,['환불금액','환불여부']])

    print("---ADD Gender--")
    X_train['gender'] = y_train['gender']
    print(X_train['gender'])

    return X_train.astype({'환불여부':'int'})

def pre_processing():
    global X_train

    # 1. pd.get_dummies
    print("---Get Dummies--")
    df_tmp = pd.get_dummies(X_train['주구매상품'])
    df_tmp['cust_id'] = np.arange(0,3500,1)
    print(df_tmp)
    X_train = pd.merge(X_train, df_tmp, on='cust_id')

    #2. drop columns
    X_train = X_train.drop(['환불금액', '주구매상품', '주구매지점'], axis=1)

    # 2. Scaler
    mms = MinMaxScaler()
    X_train = pd.DataFrame(mms.fit_transform(X_train), columns=X_train.columns)


def make_model():
    global X_train, y_train

    X_train = X_train.to_numpy()
    y_train = y_train.gender
    y_train = y_train.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    lg_model = LogisticRegression()
    lg_model.fit(X_train, y_train)

    print("훈련 점수 : {}", lg_model.score(X_train, y_train))
    print("평가 점수 : {}", lg_model.score(X_test, y_test))

    scores = cross_val_score(lg_model, X_train, y_train, cv=10)

    print("교차 검증 정확도 : {}".format(scores))
    print("교차 검증 정확도 : {} +- {}".format(np.mean(scores), np.std(scores)))

    scores = cross_val_score(lg_model, X_test, y_test, cv=10)

    print("교차 검증 정확도 : {}".format(scores))
    print("교차 검증 정확도 : {} +- {}".format(np.mean(scores), np.std(scores)))


    return 0

def val_model():
    return 0



if __name__=="__main__":
    init()
    X_train, y_train = read_files()
    X_total = X_train.shape[0]
    print_data(1)
    X_train = add_columns()
    print_data(2)
    pre_processing()
    print_data(3)
    make_model()
    '''
    print_data(4)
    val_model()
    print_data(5)
    '''