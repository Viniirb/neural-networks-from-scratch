import tensorflow as tf
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"Formato da imagem: {X_train[0].shape}")

model = Sequential()
model.add(Input(shape=(28, 28)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--- Treinando Leitor de Dígitos (Isso pode levar alguns segundos) ---")
model.fit(X_train, y_train, epochs=5)

val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAcurácia em imagens nunca vistas: {val_acc*100:.2f}%")

indice = 100
imagem = X_test[indice]
label_real = y_test[indice]

previsao = model.predict(np.expand_dims(imagem, axis=0))
numero_previsto = np.argmax(previsao)

print(f"\nTeste da Imagem #{indice}:")
print(f"A Rede diz que é o número: {numero_previsto}")
print(f"O número real é: {label_real}")