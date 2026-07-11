"""
train_model.py

Entrena un clasificador de desechos usando transfer learning sobre MobileNetV2.
Lee las imágenes desde la carpeta dataset/ (una subcarpeta por categoría) y
genera dos archivos:
    - waste_classifier.h5   -> modelo entrenado
    - class_names.json      -> lista de categorías, en el orden que usa el modelo
"""

import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15          # Baja este número si solo quieres una prueba rápida
LEARNING_RATE = 1e-4

# ---------------------------------------------------------------------------
# 1. Cargar y preparar los datos
# ---------------------------------------------------------------------------
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
)

num_classes = train_gen.num_classes
print(f"Categorías detectadas: {train_gen.class_indices}")

# ---------------------------------------------------------------------------
# 2. Construir el modelo (MobileNetV2 preentrenado + capas propias)
# ---------------------------------------------------------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # Congelamos el modelo base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------------------------
# 3. Entrenar
# ---------------------------------------------------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
)

# ---------------------------------------------------------------------------
# 4. Guardar el modelo y las categorías
# ---------------------------------------------------------------------------
model.save("waste_classifier.h5")

# Guardamos las categorías en el orden correcto (index -> nombre)
class_names = {v: k for k, v in train_gen.class_indices.items()}
with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print("\n✅ Entrenamiento terminado.")
print("Modelo guardado como: waste_classifier.h5")
print("Categorías guardadas como: class_names.json")
