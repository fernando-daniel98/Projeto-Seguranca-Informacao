import tensorflow as tf
from keras import layers, models, callbacks
from keras.applications.vgg16 import VGG16, preprocess_input
import os

# Caminhos dos diretórios
train_dir = "dataset/train"  # Ajustei para o caminho relativo padrão do seu script db
val_dir = "dataset/val"
test_dir = "dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2 

# --- CARREGAMENTO DOS DADOS ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False 
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
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
# Isso acontece na GPU e é crucial para evitar overfitting
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1), # Rotação leve (10%)
    layers.RandomZoom(0.1),     # Zoom leve
    # Brilho e Contraste são essenciais para diferenciar 'tela' de 'pele'
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
x = preprocess_input(x)  # A VGG precisa disso para normalizar as cores
x = base_model(x, training=False) # training=False é importante para BatchNormalization (se houvesse)

# --- 2. CABEÇALHO MAIS ROBUSTO (COM DROPOUT) ---
x = layers.Dense(256, activation="relu")(x) # Camada intermediária para aprender combinações
x = layers.Dropout(0.5)(x)                  # <--- O SEGREDO DO ANTI-OVERFITTING (apaga 50% dos neurônios no treino)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

# Compilação com Learning Rate explícito (ajuda a controlar a descida)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- 3. CALLBACKS (CHECKPOINT E EARLY STOPPING) ---
checkpoint_cb = callbacks.ModelCheckpoint(
    "melhor_modelo_vgg16.keras", 
    save_best_only=True, 
    monitor="val_loss",
    mode="min"
)

early_stopping_cb = callbacks.EarlyStopping(
    patience=5,             # Espera 5 épocas sem melhorar antes de parar
    monitor="val_loss",
    restore_best_weights=True
)

# Treino
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,  # Aumente as épocas, o EarlyStopping vai parar na hora certa
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# --- AVALIAÇÃO ---
print("\n--- Avaliando no Test Set (Subject Hold-out) ---")
# Carrega o melhor modelo salvo (não necessariamente o da última época)
best_model = models.load_model("melhor_modelo_vgg16.keras")
test_loss, test_acc = best_model.evaluate(test_ds)
print(f"Acurácia Final no Teste: {test_acc:.4f}")