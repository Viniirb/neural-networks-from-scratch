import tensorflow as tf
import numpy as np
import cv2 as cv

print("Carregando o modelo treinado...")

model = tf.keras.models.load_model('../models/mnist_cnn_model.keras')

print("Modelo carregado! Pronto para previsões.")

(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

indice = 777 
imagem_input = X_test[indice:indice+1]

previsao = model.predict(imagem_input, verbose=0)
numero_detectado = np.argmax(previsao)
certeza = np.max(previsao) * 100

print(f"\n--- Resultado da IA ---")
print(f"Número Real: {y_test[indice]}")
print(f"IA Detectou: {numero_detectado}")
print(f"Certeza: {certeza:.2f}%")