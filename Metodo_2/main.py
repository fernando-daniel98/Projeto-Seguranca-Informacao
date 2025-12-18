import tensorflow as tf
from keras import layers, models, callbacks
from keras.applications.vgg16 import VGG16, preprocess_input
import os
from pathlib import Path

# Diretório base do Método 2
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Caminhos dos diretórios de dados (relativo à raiz do projeto)
PROJECT_ROOT = BASE_DIR.parent
train_dir = PROJECT_ROOT / "dataset" / "train"
val_dir = PROJECT_ROOT / "dataset" / "val"
test_dir = PROJECT_ROOT / "dataset" / "test"

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
    patience=5,
    monitor="val_loss",
    restore_best_weights=True
)

# Treino
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# Salvar também o modelo final (última época com best weights restaurados)
final_model_path = MODELS_DIR / "metodo2_vgg16_final.keras"
model.save(str(final_model_path))
print(f"\nModelo final salvo em: {final_model_path}")
print(f"Melhor modelo (checkpoint) salvo em: {model_path}")

# --- AVALIAÇÃO RÁPIDA ---
print("\n--- Avaliando no Test Set (Subject Hold-out) ---")
best_model = models.load_model(str(model_path))
test_loss, test_acc = best_model.evaluate(test_ds)
print(f"Acurácia Final no Teste: {test_acc:.4f}")