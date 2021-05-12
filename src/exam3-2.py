import pandas as pd
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
    X= pd.read_csv("data/X_train.csv", encoding="CP949")
    y= pd.read_csv("data/y_train.csv", encoding="CP949")

    return X, y


def print_data():
    print(X_train.info())
    print(X_train.describe())
    print(y_train.info())
    print(y_train.describe())


if __name__=="__main__":
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 15)

    X_train, y_train= read_csv()

    print_data()



