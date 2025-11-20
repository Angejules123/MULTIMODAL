"""
utils.py
---------
Fonctions utilitaires pour le projet de détection d'Alzheimer à partir d'images IRM.

Inclut :
- Gestion des chemins, logs, seeds
- Visualisation des images et des métriques
- Évaluation du modèle
- Sauvegarde/chargement du modèle
- Intégration complète avec TensorBoard

Auteur : [Ton nom]
Date : [Date du jour]
"""

import os
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model


# ================================================================
# 📁 CONFIGURATION ET UTILITAIRES DE BASE
# ================================================================

def create_dir(path: str):
    """Crée un dossier s'il n'existe pas déjà."""
    os.makedirs(path, exist_ok=True)
    return path


def set_seed(seed: int = 42):
    """Fixe les graines aléatoires pour la reproductibilité."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def get_timestamp():
    """Retourne un horodatage au format lisible."""
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# ================================================================
# 🧠 TENSORBOARD — ENREGISTREMENT DES LOGS
# ================================================================

def setup_tensorboard(log_dir_base="logs/fit"):
    """
    Configure le callback TensorBoard avec un dossier horodaté.
    Retourne le callback prêt à être ajouté à model.fit().
    """
    log_dir = os.path.join(log_dir_base, get_timestamp())
    create_dir(log_dir)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print(f"📊 TensorBoard est configuré : {log_dir}")
    print("👉 Lance-le avec : tensorboard --logdir logs/fit")
    return tensorboard_cb


def setup_callbacks(model_path="models/best_model.h5", patience=5):
    """
    Crée une liste de callbacks standard :
      - TensorBoard
      - EarlyStopping
      - ModelCheckpoint
    """
    create_dir(os.path.dirname(model_path))
    tensorboard_cb = setup_tensorboard()
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        model_path, save_best_only=True, monitor="val_accuracy", mode="max"
    )
    return [tensorboard_cb, earlystop_cb, checkpoint_cb]


# ================================================================
# 🖼️ VISUALISATION D’IMAGES
# ================================================================

def show_random_images(data_gen, n=9, title="Aperçu des images d'entraînement"):
    """
    Affiche un échantillon aléatoire d’images à partir d’un ImageDataGenerator.
    """
    images, labels = next(data_gen)
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.suptitle(title, fontsize=14)
    plt.show()


# ================================================================
# 📈 VISUALISATION DE L'ENTRAÎNEMENT
# ================================================================

def plot_training_history(history):
    """
    Affiche les courbes de perte et d’accuracy pendant l’entraînement.
    """
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Évolution de l'accuracy")
    plt.xlabel("Époque")
    plt.ylabel("Accuracy")
    plt.legend
