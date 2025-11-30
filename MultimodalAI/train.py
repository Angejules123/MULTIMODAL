<<<<<<< HEAD
"""Logique d'entraînement"""
import torch
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger):
    """
    Entraîne le modèle pour une époque.
    
    Args:
        model: modèle PyTorch
        train_loader: DataLoader d'entraînement
        criterion: fonction de loss
        optimizer: optimiseur
        device: device (CPU/GPU)
        epoch: numéro d'époque
        logger: logger
    
    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Train {epoch+1}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Métriques
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    if logger:
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device, epoch, logger):
    """
    Valide le modèle.
    
    Args:
        model: modèle PyTorch
        val_loader: DataLoader de validation
        criterion: fonction de loss
        device: device (CPU/GPU)
        epoch: numéro d'époque
        logger: logger
    
    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val {epoch+1}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Métriques
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    if logger:
        logger.info(f"Epoch {epoch+1} - Val Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy
=======
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
>>>>>>> 3994636c730aad2d87d13b99a6031df3de3d57db
