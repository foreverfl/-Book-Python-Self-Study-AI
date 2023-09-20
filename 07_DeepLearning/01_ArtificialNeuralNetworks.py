from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input,
                              test_target) = keras.datasets.fashion_mnist.load_data()

# 데이터 크기 확인하기
print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)

# 1차원 배열로 변환
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28 * 28)

# 훈련 세트와 검증 세트를 나눔
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 모델 생성
# keras.layers.Dense(뉴런 개수, 뉴런의 출력에 적용할 함수, 입력의 크기)
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784, ))
model = keras.Sequential(dense)
# label이 one-hot encoding 되어있지 않으므로 sparse_categorical_srossentropy를 사용
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

# 모델 훈련
model.fit(train_scaled, train_target, epochs=5)

# 모델 평가
history = model.evaluate(val_scaled, val_target)
print(history)
