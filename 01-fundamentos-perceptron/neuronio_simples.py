import numpy as np

class Neuronio:
    def __init__(self, pesos, bias):
        self.pesos = np.array(pesos)
        self.bias = bias

    def ativacao_degrau(self, x):
        return 1 if x >= 0 else 0
    
    def processar(self, entradas):
        soma_ponderada = np.dot(entradas, self.pesos) + self.bias

        saida = self.ativacao_degrau(soma_ponderada)
        return saida
    

pesos_iniciais = [1.0, 1.0]
bias_inicial = -1.5

meu_neuronio = Neuronio(pesos_iniciais, bias_inicial)

entradas_teste = [
    np.array([0, 0]), 
    np.array([0, 1]), 
    np.array([1, 0]), 
    np.array([1, 1])
]

print("Testando Neurônio Lógico AND:")

for ent in entradas_teste:
    resultado = meu_neuronio.processar(ent)
    print(f"Entradas: {ent} -> Saída: {resultado}")