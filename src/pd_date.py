import pandas as pd

data={
    'col1': [1,2,3,4,5]
    , 'col2': ['a', 'b','c','d','e']
    , 'col3': ['2020-10-01', '2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01']
}

pd_1= pd.DataFrame(data)
print(pd_1)

pd_1['date']= pd.to_datetime(pd_1['col3'])

print(pd_1.info())
pd_1['year'] = pd_1.date.dt.year
pd_1['mm'] = pd_1.date.dt.month
pd_1['dd'] = pd_1.date.dt.day
print(pd_1.info())
print(pd_1)