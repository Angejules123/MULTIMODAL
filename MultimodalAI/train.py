import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers

from MultimodalAI.config import (
    DATA_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, SEED, MODEL_PATH
)
from MultimodalAI.utils import (
    set_seed, setup_callbacks, plot_training_history,
    evaluate_model, save_model, count_parameters
)


# ================================================================
# ⚙️ CONFIGURATION INITIALE
# ================================================================

set_seed(SEED)

print("📁 Dossier de données :", DATA_DIR)
print("🧠 Entraînement du modèle Alzheimer Detector")
print("=" * 60)


# ================================================================
# 🧹 CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
# ================================================================

def get_data_generators():

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED
    )

    return train_gen, val_gen


# ================================================================
# 🧠 CRÉATION DU MODÈLE CNN
# ================================================================

def build_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Construit un petit modèle CNN pour la détection d'Alzheimer.
    (Facilement remplaçable par EfficientNet, MobileNetV2, etc.)
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    count_parameters(model)
    return model


# ================================================================
# 🚀 ENTRAÎNEMENT DU MODÈLE
# ================================================================

def train_model():
    """
    Entraîne le modèle Alzheimer et sauvegarde les meilleurs poids automatiquement.
    """
    train_gen, val_gen = get_data_generators()

    model = build_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=train_gen.num_classes
    )

    callbacks = setup_callbacks(model_path=MODEL_PATH, patience=5)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    plot_training_history(history)
    save_model(model, MODEL_PATH)

    class_names = list(train_gen.class_indices.keys())
    evaluate_model(model, val_gen, class_names)


# ================================================================
# 🏁 POINT D’ENTRÉE
# ================================================================

if __name__ == "__main__":
    train_model()
