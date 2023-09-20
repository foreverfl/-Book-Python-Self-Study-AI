"""
- SGDClassifier
* Scikit-Learn 라이브러리에서 제공하는 분류(classification) 알고리즘 중 하나로, "Stochastic Gradient Descent" (SGD, 확률적 경사 하강법)을 사용.
* 확률적 경사 하강법은 대규모 데이터셋에 효율적인 최적화 알고리즘이며, 각 반복에서 랜덤하게 선택한 하나의 샘플을 사용하여 가중치를 업데이트.
* 이 알고리즘은 선형 모델(예: 선형 회귀, 로지스틱 회귀 등)을 최적화하는 데 사용되며, 빠르게 수렴하는 특성이 있음. 따라서 큰 데이터셋에 적합.
"""

# 표준 라이브러리 모듈
import numpy as np

# 서드파티 라이브러리 모듈
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
from tensorflow import keras
import matplotlib.pyplot as plt

(train_input, train_target), (test_input,
                              test_target) = keras.datasets.fashion_mnist.load_data()

# 데이터 크기 확인하기
print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)

# 데이터 샘플 확인하기
# fig, axs = plt.subplots(1, 10, figsize=(10, 10))
# for i in range(10):
#     axs[i].imshow(train_input[i], cmap='gray_r')
#     axs[i].axis('off')

# plt.show()

# 레이블 확인하기
print(np.unique(train_target, return_counts=True))

# SGDClassifier를 위한 1차원 배열로 변환
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28 * 28)
print(train_scaled.shape)

# SGDClassifier
sc = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=1)
print(np.mean(scores['test_score']))
