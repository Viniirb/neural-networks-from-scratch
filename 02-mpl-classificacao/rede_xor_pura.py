import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)

pesos_camada1 = np.random.uniform(size=(2,4))
bias_camada1 = np.random.uniform(size=(1,4))

pesos_camada2 = np.random.uniform(size=(4,1))
bias_camada2 = np.random.uniform(size=(1,1))

taxa_aprendizado =0.1
epocas = 10000

print("Treinando...")

for i in range(epocas):
    entrada_camada1 = np.dot(X, pesos_camada1) + bias_camada1
    saida_camada1 = sigmoid(entrada_camada1)

    entrada_camada2 = np.dot(saida_camada1, pesos_camada2) + bias_camada2
    saida_final = sigmoid(entrada_camada2)

    erro = y - saida_final

    delta_saida = erro * sigmoid_derivative(saida_final)

    erro_camada_oculta = delta_saida.dot(pesos_camada2.T)
    delta_camada_oculta = erro_camada_oculta * sigmoid_derivative(saida_camada1)

    pesos_camada2 += saida_camada1.T.dot(delta_saida) * taxa_aprendizado
    bias_camada2 += np.sum(delta_saida, axis=0, keepdims=True) * taxa_aprendizado

    pesos_camada1 += X.T.dot(delta_camada_oculta) * taxa_aprendizado
    bias_camada1 += np.sum(delta_camada_oculta, axis=0, keepdims=True) * taxa_aprendizado

print("Treino finalizado!")
for i in range(len(X)):
    camada1 = sigmoid(np.dot(X[i], pesos_camada1) + bias_camada1)
    resultado = sigmoid(np.dot(camada1, pesos_camada2) + bias_camada2)
    print(f"Entrada: {X[i]} | Esperado: {y[i]} | Rede: {resultado[0][0]:.4f}")