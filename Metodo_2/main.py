import tensorflow as tf
from keras import layers, models
from keras.applications.vgg16 import VGG16, preprocess_input  # <-- alterado

# Caminhos dos diretórios
train_dir = "../dataset/train"
val_dir = "../dataset/val"
test_dir = "../dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2  # fake, real

# Carrega datasets a partir dos diretórios (labels vêm dos nomes das pastas)
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Opcional: melhora desempenho com prefetch
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Modelo base VGG16 (pré-treinado no ImageNet)
base_model = VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,),
    pooling="avg"   # global average pooling
)

# Para primeiros testes, mantemos o backbone congelado
base_model.trainable = False

# Cabeçalho simples para 2 classes (fake / real)
inputs = layers.Input(shape=IMG_SIZE + (3,))
x = preprocess_input(inputs)              # preprocess da VGG16
x = base_model(x, training=False)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Treino inicial
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# Avaliação no conjunto de teste
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")
