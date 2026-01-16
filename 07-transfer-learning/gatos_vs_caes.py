import tensorflow as tf
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
import os
import numpy as np

PATH = os.path.join(os.getcwd(), 'dados', 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

if not os.path.exists(train_dir):
    print("="*50)
    print(f"ERRO: O Python não achou a pasta 'train'.")
    print(f"Ele procurou aqui: {train_dir}")
    print(" Verifique se você extraiu o ZIP dentro da pasta 'dados' corretamente.")
    print("="*50)
    exit()

print(f"--- Dados encontrados em: {PATH} ---")

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=32, image_size=(160, 160))

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, batch_size=32, image_size=(160, 160))

print("\n--- Carregando a VGG16 (O Cérebro) ---")
base_model = VGG16(input_shape=(160, 160, 3), include_top=False, weights='imagenet')

base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- Treinando apenas a parte final ---")
history = model.fit(train_dataset, epochs=5, validation_data=validation_dataset)

print("\n--- Avaliação ---")
loss, accuracy = model.evaluate(validation_dataset)
print(f"Acurácia Final: {accuracy*100:.2f}%")