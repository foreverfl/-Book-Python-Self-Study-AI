import os
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))

(train_input, train_target), (test_input,
                              test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

# 훈련 세트와 검증 세트를 나눔
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)


def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model


# 모델 생성
model = model_fn(keras.layers.Dropout(0.3))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    os.path.join(current_path, 'best-model.h5'), save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=2, restore_best_weights=True)

# 모델 평가
history = model.fit(train_scaled, train_target, epochs=10,
                    verbose=1, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
print(history.history.keys())

# 모델 현재 경로에 저장하기

model.save_weights(os.path.join(current_path, 'model-weights.h5'))
model.save(os.path.join(current_path, 'model-whole.h5'))

# 결과 시각화
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()


# plt.plot(history.history['accuracy'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
