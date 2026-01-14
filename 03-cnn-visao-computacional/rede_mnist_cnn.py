import tensorflow as tf
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

model = Sequential()
model.add(Input(shape=(28, 28, 1)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--- Treinando CNN (Isso vai ser um pouco mais lento) ---")
model.fit(X_train, y_train, epochs=5)

val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAcurácia da CNN: {val_acc*100:.2f}%")

nome_arquivo = "mnist_cnn_model.keras"
path = "models/"
model.save(path + nome_arquivo)
print(f"\nModelo salvo com sucesso em: {path + nome_arquivo}")