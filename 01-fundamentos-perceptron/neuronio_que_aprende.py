import numpy as np

class NeuronioQueAprende:
    def __init__(self, taxa_aprendizado=0.1 ):
        self.pesos = np.array([0.0, 0.0])
        self.bias = 0.0
        self.taxa_aprendizado = taxa_aprendizado

    def ativacao_degrau(self, x):
        return 1 if x >= 0 else 0
    
    def prever(self, entradas):
        soma_ponderada = np.dot(entradas, self.pesos) + self.bias
        return self.ativacao_degrau(soma_ponderada)
    
    def treinar(self, dados_treino, gabaritos, epocas=10):
        print(f"--- Iniciando Treinamento (Taxa: {self.taxa_aprendizado}) ---")

        for epoca in range(epocas):
            erro_total = 0
            
            for i in range(len(dados_treino)):
                entrada_atual = dados_treino[i]
                gabarito_atual = gabaritos[i]

                previsao = self.prever(entrada_atual)
                erro = gabarito_atual - previsao

                if erro != 0:
                    self.pesos += self.taxa_aprendizado * erro * entrada_atual
                    self.bias += self.taxa_aprendizado * erro

                    erro_total += 1
            print(f"Época {epoca + 1} - Erros Cometidos: {erro_total}")

            if erro_total == 0:
                print(">>> Aprendizado concluído com sucesso!")
                break


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

brain = NeuronioQueAprende(taxa_aprendizado=0.1)

print(f"Pesos INICIAIS: {brain.pesos}, Bias: {brain.bias}")

brain.treinar(X, y)

print(f"\nPesos FINAIS aprendidos: {brain.pesos}, Bias: {brain.bias}")

print("\n--- Teste Final ---")
for entrada in X:
    print(f"Entrada {entrada} -> Previsão do Neurônio: {brain.prever(entrada)}")