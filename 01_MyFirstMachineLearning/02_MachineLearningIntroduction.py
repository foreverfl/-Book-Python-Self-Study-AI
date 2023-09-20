import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

current_script_path = os.path.abspath(__file__)  # 현재 스크립트 파일의 절대 경로
current_directory = os.path.dirname(current_script_path)
fish_csv_path = os.path.join(current_directory, 'fish.csv')

data = pd.read_csv(fish_csv_path)
print(data)

bream_length = data[data['Species'] == 'Bream']['Length1'].tolist()
bream_weight = data[data['Species'] == 'Bream']['Weight'].tolist()
smelt_length = data[data['Species'] == 'Smelt']['Length1'].tolist()
smelt_weight = data[data['Species'] == 'Smelt']['Weight'].tolist()

# Bream과 Smelt의 length와 weight를 병합
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 각 요소는 [length, weight] 형태로 구성
# zip: iterable 객체를 인자로 받아, 각 원소들을 튜플의 형태로 접근할 수 있게 해줌
fish_data = [[l, w] for l, w in zip(length, weight)]
# Bream은 1, Smelt는 0으로 라벨링
fish_target = [1] * len(bream_length) + [0] * len(smelt_length)

# K-Nearest Neighbors 분류기를 생성하고 훈련
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)

# 훈련 데이터에 대한 정확도
print("정확도:", kn.score(fish_data, fish_target))

# K-Nearest
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^')  # 점의 모양을 삼각형으로 지정
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# K-Nearest Neighbors 예측
kn.predict([[30, 600]])
print(kn._fit_X)  # 피팅된 특성 데이터(X)를 출력
print(kn._y)  # 피팅된 라벨 데이터(y)를 출력

# 49개의 이웃을 고려하는 K-Nearest Neighbors 분류기를 생성
kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data, fish_target)
print("정확도:", kn49.score(fish_data, fish_target))
