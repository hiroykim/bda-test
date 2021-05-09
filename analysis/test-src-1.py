from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

items = ['TV', '냉장고', '전자렌지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

# 숫자 값으로 변환하기 위해 LabelEncoder로 먼저 변환한다.
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print(labels)

# Test
test = encoder.inverse_transform(labels)
print("test : ",test)

# 2차원 데이터로 변환하기
labels = labels.reshape(-1,1)
print("reshape:",labels)

# 원-핫 인코딩 적용
oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print(oh_labels.toarray())