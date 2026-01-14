import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([
    [0], 
    [1], 
    [1], 
    [0]
], dtype=np.float32)

model = Sequential()

model.add(Input(shape=(2,)))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("--- Iniciando Treinamento ---")

model.fit(X, y, epochs=2000, verbose=0)
print("--- Treinamento Concluído ---")

print("\n--- Resultados Finais ---")
previsoes = model.predict(X)

for i in range(4):
    entrada = X[i]
    saida_real = y[i]
    previsao_rede = previsoes[i]

    print(f"Entrada: {entrada} | Esperado: {saida_real} | Rede previu: {previsao_rede[0]:.4f} -> {round(previsao_rede[0])}")