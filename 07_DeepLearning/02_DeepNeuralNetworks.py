from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input,
                              test_target) = keras.datasets.fashion_mnist.load_data()

# 데이터 크기 확인하기
print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)

train_scaled = train_input / 255.0

# 훈련 세트와 검증 세트를 나눔
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 모델 생성
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

# 모델 훈련
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)

# 모델 평가
history = model.evaluate(val_scaled, val_target)
print(history)
