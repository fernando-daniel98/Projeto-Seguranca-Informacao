import tensorflow as tf
from keras import layers, models, callbacks
from keras.applications.vgg16 import VGG16, preprocess_input
import os
import numpy as np
from pathlib import Path

# Diretório base do Método 2
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Caminhos dos diretórios de dados (relativo à raiz do projeto)
# TREINO E TESTE: dataset2 (CASIA-FASD)
PROJECT_ROOT = BASE_DIR.parent
train_dir = PROJECT_ROOT / "dataset2" / "train"
val_dir = PROJECT_ROOT / "dataset2" / "val"
test_dir = PROJECT_ROOT / "dataset2" / "test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2 

# --- CARREGAMENTO DOS DADOS ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    str(train_dir),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(val_dir),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False 
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    str(test_dir),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Otimização de I/O
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# --- CALCULAR CLASS WEIGHTS PARA LIDAR COM DESBALANCEAMENTO ---
print("\nCalculando class weights...")
train_labels = []
for _, labels in train_ds.unbatch().as_numpy_iterator():
    train_labels.append(np.argmax(labels))

train_labels = np.array(train_labels)
class_counts = np.bincount(train_labels)
total_samples = len(train_labels)

# Weight inversamente proporcional à frequência
class_weight = {i: total_samples / (len(class_counts) * count) 
                for i, count in enumerate(class_counts)}

print(f"Distribuição de classes no treino:")
print(f"  Fake (0): {class_counts[0]} imagens (weight: {class_weight[0]:.4f})")
print(f"  Real (1): {class_counts[1]} imagens (weight: {class_weight[1]:.4f})")

# Re-criar train_ds pois foi consumido
train_ds = tf.keras.utils.image_dataset_from_directory(
    str(train_dir),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 1. DATA AUGMENTATION (DENTRO DO MODELO) ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2), 
    layers.RandomBrightness(0.2),
], name="data_augmentation")

# --- MODELO ---
base_model = VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,),
    pooling="avg" 
)

# Congelar backbone (inicialmente)
base_model.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))

# A ordem importa: Augmentation -> Preprocess -> Backbone
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)

# --- 2. CABEÇALHO MAIS ROBUSTO (COM DROPOUT) ---
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

# Compilação
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- 3. CALLBACKS (CHECKPOINT E EARLY STOPPING) ---
model_path = MODELS_DIR / "metodo2_vgg16_best.keras"

checkpoint_cb = callbacks.ModelCheckpoint(
    str(model_path), 
    save_best_only=True, 
    monitor="val_loss",
    mode="min"
)

early_stopping_cb = callbacks.EarlyStopping(
    patience=7,  # Aumentado para permitir mais recuperação
    monitor="val_loss",
    restore_best_weights=True
)

# Treino com class_weight
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weight,  # Aplicar pesos de classe
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# Salvar também o modelo final (última época com best weights restaurados)
final_model_path = MODELS_DIR / "metodo2_vgg16_final.keras"
model.save(str(final_model_path))
print(f"\nModelo final salvo em: {final_model_path}")
print(f"Melhor modelo (checkpoint) salvo em: {model_path}")

# --- AVALIAÇÃO RÁPIDA (CASIA Test Set) ---
print("\n--- Avaliando no Test Set (CASIA-FASD) ---")
best_model = models.load_model(str(model_path))
test_loss, test_acc = best_model.evaluate(test_ds)
print(f"Acurácia Final no Teste (CASIA): {test_acc:.4f}")