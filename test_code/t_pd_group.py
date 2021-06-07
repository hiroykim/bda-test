import pandas as pd

if 0:
    data = [
        ['A', 10]
        , ['B', 100]
        , ['A', 20]
        , ['B', 200]
        , ['A', 30]
        , ['B', 300]
    ]

    colname = ['Col1', 'Col2']
    df = pd.DataFrame(data, columns=colname)
else:
    data = {
        'Col1': ['A', 'B', 'A', 'B', 'A', 'B'],
        'Col2': [10, 100, 20, 400, 30, 800, ],
    }

    df = pd.DataFrame(data)

print(df)

df['Col3'] = df.groupby('Col1')['Col2'].transform('mean')
print(df)

df_tr = df.transpose()
print(df_tr)

#https://yganalyst.github.io/data_handling/Pd_14/
'''
pdf2 = pd.pivot_table(df,                # 피벗할 데이터프레임
                     index = 'class',    # 행 위치에 들어갈 열
                     columns = 'sex',    # 열 위치에 들어갈 열
                     values = 'survived',     # 데이터로 사용할 열
                     aggfunc = ['mean', 'sum'])   # 데이터 집계함수
pdf2
'''
df_pivot = pd.pivot_table(df, index=['Col1'], columns=['Col2'], values=['Col3'], aggfunc='sum')
print(df_pivot)