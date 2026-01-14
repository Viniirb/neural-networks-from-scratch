<div align="center">

# 🧠 Jornada Deep Learning: do Perceptron à CNN

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&center=true&vCenter=true&width=900&lines=Do+Perceptron+%C3%A0+CNN+no+MNIST;Implementa%C3%A7%C3%B5es+em+Python+puro+e+Keras;MLP%2C+Backprop%2C+Vis%C3%A3o+Computacional" alt="Typing SVG" />

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)

</div>

Este repositório documenta minha evolução no estudo de Redes Neurais Artificiais — saindo da matemática de um único neurônio e indo até uma CNN para Visão Computacional com TensorFlow/Keras.

## 📌 Conteúdo

- [Estrutura](#-estrutura-do-projeto)
- [Como rodar](#-como-rodar)
- [Scripts por módulo](#-scripts-por-módulo)
- [Resultados](#-resultados-obtidos)
- [Modelo treinado](#-modelo-treinado)

## 📂 Estrutura do Projeto

O projeto está dividido em módulos de complexidade crescente:

- **01 - Fundamentos:** neurônio/perceptron em Python puro (pesos, bias e ativação).
- **02 - MLP:** problemas não-lineares (XOR) e classificação (Iris). Inclui parte "na mão".
- **03 - CNN:** MNIST com camadas Conv2D/Pooling.
- **04 - Inferência:** script que carrega o modelo treinado e realiza predições.

## 💻 Como Rodar

### 1) Clonar o repositório

```bash
git clone https://github.com/Viniirb/neural-networks-from-scratch.git
cd neural-networks-from-scratch
```

### 2) Criar ambiente e instalar dependências

Recomendado usar `venv`:

```bash
python -m venv .venv
```

Ativar no Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Ativar no Windows (CMD):

```bat
.\.venv\Scripts\activate
```

Instalar:

```bash
pip install -r requirements.txt
```

> Nota: o TensorFlow costuma ter suporte oficial para versões específicas do Python (frequentemente 3.10–3.12). Se você estiver no 3.13 e der erro ao instalar/importar `tensorflow`, troque para uma versão suportada.

## ▶️ Scripts por módulo

Fundamentos:

```bash
python 01-fundamentos-perceptron/neuronio_simples.py
python 01-fundamentos-perceptron/neuronio_que_aprende.py
```

MLP / Classificação:

```bash
python 02-mpl-classificacao/rede_xor_pura.py
python 02-mpl-classificacao/rede_xor_keras.py
python 02-mpl-classificacao/rede_iris.py
```

MNIST (MLP e CNN):

```bash
python 03-cnn-visao-computacional/rede_mnist_mpl.py
python 03-cnn-visao-computacional/rede_mnist_cnn.py
```

Inferência (carrega o modelo salvo):

```bash
python 04-app-inferencia/usar_ia.py
```

## 📊 Resultados Obtidos

| Modelo | Arquitetura | Dataset | Acurácia (referência) |
| :--- | :--- | :--- | :--- |
| Perceptron simples | 1 neurônio | Porta lógica (ex.: AND) | 100% |
| MLP (Dense) | camadas densas | MNIST | ~97% |
| CNN (Conv2D) | blocos convolucionais | MNIST | ~99% |

> Os valores podem variar por seed/hiperparâmetros/ambiente.

## 🧩 Modelo treinado

O modelo final já está versionado em:

- `models/mnist_cnn_model.keras`

---

Feito por **Vinicius Rolim Barbosa**

- GitHub: https://github.com/Viniirb
- Sugestões/bugs: abra uma issue no repositório


