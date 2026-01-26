import pandas as pd
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.tensorflow
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data():
    """Carrega dataset UCI Parkinson e pré-processa os dados."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

    print(f"Baixando dataset de: {url}")
    df = pd.read_csv(url)

    X= df.drop(['name', 'status'], axis=1)
    y= df['status'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def build_model(input_dim, num_hidden_layers, neurons=64):
    """Realiza Construção do modelo MLP."""
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))

    for i in range(num_hidden_layers):
        model.add(layers.Dense(neurons, activation='relu'))

        model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def main():
    mlflow.set_experiment("Parkinson_Deep_MLP_Experiments")

    X_train, X_test, y_train, y_test = load_data()

    input_dim = X_train.shape[1]

    depths = [1, 3, 5, 10]

    print("\n Iniciando Ciclo de Experimentos...")

    for depth in depths:
        run_name = f"Deep_MLP_{depth}_Layers"
        print(f"\nTreinando: {run_name}")

        with mlflow.start_run(run_name=run_name):

            mlflow.log_param("num_hidden_layers", depth)
            mlflow.log_param("neurons_per_layer", 64)
            mlflow.log_param("optimizer", "adam")

            model = build_model(input_dim, num_hidden_layers=depth)

            history = model.fit(
                X_train, y_train,
                epochs=60,
                batch_size=16,
                validation_split=0.2,
                verbose=0
            )

            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

            mlflow.log_metric("test_loss", loss)
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.tensorflow.log_model(model, "model")

            print(f"   ✅ Acurácia: {accuracy:.4f} | Loss: {loss:.4f}")

if __name__ == "__main__":
    main()