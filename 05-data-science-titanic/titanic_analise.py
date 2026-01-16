import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("--- Carregando Dataset Titanic ---")

df = sns.load_dataset('titanic')
print(df.head())

print("\n--- Analisando Valores Faltantes (Buracos) ---")
print(df.isnull().sum())

df = df.drop(['deck', 'embark_town', 'alive', 'who', 'adult_male', 'class', 'embarked'], axis=1)

media_idade = df['age'].mean()
df['age'] = df['age'].fillna(media_idade)

print("\n--- Convertendo Textos para Números ---")
df['sex'] = df['sex'].map({'male': 0 , 'female': 1})
df['alone'] = df['alone'].astype(int)

df = df.dropna()

print("\n--- Dados Prontos para a IA ---")
print(df.head())

X = df.drop('survived', axis=1)
y = df['survived']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTreinaremos com {len(X_train)} passageiros e testaremos com {len(X_test)}.")

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- Treinando ---")
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Acurácia Final: {accuracy*100:.2f}%")