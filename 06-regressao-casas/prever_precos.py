import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("--- Carregando Dados Imobiliários da Califórnia ---")
housing = fetch_california_housing()
X = housing.data
y = housing.target

df = pd.DataFrame(X, columns=housing.feature_names)
df['PRECO'] = y
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

print("\n--- Treinando Corretor Artificial ---")
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)

loss, mae = model.evaluate(X_test, y_test, verbose=0)

print(f"\nErro Médio Absoluto (MAE): {mae:.4f}")
print("Isso significa que, em média, a IA erra o preço por:")
print(f"${mae * 100000:.2f}")

print("\n--- Teste de uma Casa Específica ---")
nova_casa = X_test[0:1]
preco_real = y_test[0]
preco_previsto = model.predict(nova_casa)[0][0]

print(f"Preço Real: ${preco_real * 100000:.2f}")
print(f"Preço Previsto pela IA: ${preco_previsto * 100000:.2f}")
print(f"Diferença: ${abs(preco_real - preco_previsto) * 100000:.2f}")