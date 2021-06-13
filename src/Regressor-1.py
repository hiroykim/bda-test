import sklearn
from sklearn.datasets import load_boston, load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

boston = load_boston()
diabetes = load_diabetes()

basemodel = make_pipeline( StandardScaler(), KNeighborsRegressor() )
cross_val = cross_validate(estimator= basemodel, X=boston.data, y=boston.target, cv=5)

print("avg fit time : {} (+- {})".format(cross_val['fit_time'].mean(), cross_val['fit_time'].std()))
print("avg score time : {} (+- {})".format(cross_val['score_time'].mean(), cross_val['score_time'].std()))
print("avg test score : {} (+- {})".format(cross_val['test_score'].mean(), cross_val['test_score'].std()))

baggingmodel = BaggingRegressor(basemodel, n_estimators=10, max_samples=0.5, max_features=0.5)

cross_val = cross_validate(estimator= baggingmodel, X=boston.data, y=boston.target, cv=5)

print("avg fit time : {} (+- {})".format(cross_val['fit_time'].mean(), cross_val['fit_time'].std()))
print("avg score time : {} (+- {})".format(cross_val['score_time'].mean(), cross_val['score_time'].std()))
print("avg test score : {} (+- {})".format(cross_val['test_score'].mean(), cross_val['test_score'].std()))