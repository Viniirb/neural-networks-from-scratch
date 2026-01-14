# 🧠 Jornada Deep Learning: Do Perceptron à CNN

Este repositório documenta minha evolução no estudo de Redes Neurais Artificiais, partindo da matemática básica de um único neurônio até o desenvolvimento de Redes Convolucionais (CNNs) para Visão Computacional com TensorFlow/Keras.

## 📂 Estrutura do Projeto

O projeto está dividido em módulos de complexidade crescente:

- **01 - Fundamentos:** Implementação de um Perceptron do zero (Python puro) entendendo pesos, bias e função de ativação.
- **02 - MLP (Multilayer Perceptron):** Resolução de problemas não-lineares (XOR) e classificação multiclasse (Iris Dataset). Inclui implementação de Backpropagation "na mão".
- **03 - CNN (Convolutional Neural Networks):** Classificação de dígitos manuscritos (MNIST) atingindo **99% de acurácia** usando camadas de Convolução e Pooling.
- **04 - Inferência:** Script simulando um ambiente de produção que carrega o modelo treinado para realizar predições.

## 🚀 Tecnologias

- Python 3.13
- TensorFlow & Keras
- NumPy (Álgebra Linear)
- Scikit-Learn (Pré-processamento)

## 📊 Resultados Obtidos

| Modelo | Arquitetura | Dataset | Acurácia |
| :--- | :--- | :--- | :--- |
| Perceptron Simples | 1 Neurônio | Porta Lógica AND | 100% |
| MLP (Dense) | 2 Camadas Ocultas | MNIST (Dígitos) | ~97.5% |
| **CNN (Conv2D)** | **2 Blocos Convolucionais** | **MNIST (Dígitos)** | **99.08%** |

## 💻 Como Rodar

1. Clone o repositório:
```bash
git clone [https://github.com/Viniirb/neural-networks-from-scratch.git](https://github.com/Viniirb/neural-networks-from-scratch.git)
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute a inferência(Teste o modelo final):
```bash
python 04-app-inferencia/usar_ia.py
```

---

Desenvolvido por Vinicius Rolim Barbosa - Estudante de Ciência da Computação & Dev Full-Stack

---

