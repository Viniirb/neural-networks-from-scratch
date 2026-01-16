import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

VOCABULARIO = 10000
TAMANHO_MAX = 200

print("--- Carregando dados do IMDB ---")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCABULARIO)

print(f"Treino: {len(X_train)} críticas")
print(f"Teste: {len(X_test)} críticas")

print(f"\nExemplo de crítica (em números): {X_train[0][:10]} ...")

print("\n--- Padronizando tamanho das críticas (Padding) ---")
X_train = sequence.pad_sequences(X_train, maxlen=TAMANHO_MAX)
X_test = sequence.pad_sequences(X_test, maxlen=TAMANHO_MAX)

model = Sequential()

model.add(Embedding(input_dim=VOCABULARIO, output_dim=128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print("\n--- Iniciando Treinamento (Pode demorar um pouco) ---")
model.fit(X_train, y_train, batch_size=32, epochs=3, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAcurácia no Teste: {accuracy*100:.2f}%")


word_index = imdb.get_word_index()

def prever_frase(texto):
    palavras = texto.lower().split()

    vetor = [word_index.get(word, 0) + 3 for word in palavras]

    vetor = [id if id < VOCABULARIO else 2 for id in vetor]

    vetor_pad = sequence.pad_sequences([vetor], maxlen=TAMANHO_MAX)

    predicao = model.predict(vetor_pad, verbose=0)[0][0]
    sentimento = 'POSITIVA' if predicao > 0.5 else 'NEGATIVA'
    return sentimento, predicao

print("\n--- Teste Prático ---")
frase1 = "This movie was fantastic really great acting"
frase2 = "Total waste of time terrible plot and boring"

print(f"Frase: '{frase1}' -> {prever_frase(frase1)}")
print(f"Frase: '{frase2}' -> {prever_frase(frase2)}")