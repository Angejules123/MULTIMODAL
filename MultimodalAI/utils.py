<<<<<<< HEAD
"""Utilitaires"""
import torch
import logging
from pathlib import Path

def get_device(device_str='auto'):
    """
    Récupère le device approprié (CPU, CUDA, MPS).
    
    Args:
        device_str: 'auto', 'cpu', 'cuda', ou 'mps'
    
    Returns:
        torch.device
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    elif device_str == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_str == 'mps':
        return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        return torch.device('cpu')

def count_parameters(model):
    """Compte le nombre de paramètres d'un modèle"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(checkpoint_dict, is_best, checkpoint_path, best_model_dir):
    """
    Sauvegarde un checkpoint du modèle.
    
    Args:
        checkpoint_dict: dictionnaire contenant les infos du checkpoint
        is_best: bool, si c'est le meilleur modèle
        checkpoint_path: chemin du checkpoint
        best_model_dir: répertoire du meilleur modèle
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_dict, checkpoint_path)
    
    if is_best:
        best_model_path = Path(best_model_dir) / 'best_model.pth'
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_dict, best_model_path)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Charge un checkpoint du modèle.
    
    Args:
        checkpoint_path: chemin du checkpoint
        model: modèle PyTorch
        optimizer: optimiseur (optionnel)
    
    Returns:
        dictionnaire du checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

def setup_logging(log_file):
    """
    Configure le logging vers un fichier et console.
    
    Args:
        log_file: chemin du fichier de log
    
    Returns:
        logger: logger configuré
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Handler fichier
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Handler console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
=======
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
>>>>>>> 3994636c730aad2d87d13b99a6031df3de3d57db
