import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = to_categorical(y, num_classes=3)

X_trieno, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Input(shape=(4,)))

model.add(Dense(10, activation='relu'))

model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("--- Treinando nas Flores ---")
model.fit(X_trieno, y_treino, epochs=100, verbose=0)

loss, accuracy = model.evaluate(X_teste, y_teste, verbose=0)
print(f"Acurácia no teste: {accuracy*100:.2f}%")

print("\n--- Teste Real ---")
amostra = X_teste[0:1]
previsao = model.predict(amostra)
real = y_teste[0]

print(f"Probabilidades: {previsao[0]}")
print(f"Classe Prevista: {np.argmax(previsao)} (Maior probabilidade)")
print(f"Classe Real:     {np.argmax(real)}")