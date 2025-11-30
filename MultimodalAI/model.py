<<<<<<< HEAD
"""Architectures des modèles"""
import torch
import torch.nn as nn
import torchvision.models as models

def create_model(config):
    """
    Crée un modèle selon la configuration.
    
    Args:
        config: dictionnaire de configuration contenant:
            - model.architecture: nom de l'architecture (resnet50, etc.)
            - model.pretrained: booléen pour utiliser des poids pré-entraînés
            - model.num_classes: nombre de classes
            - model.dropout: taux de dropout
    
    Returns:
        model: modèle PyTorch configuré
    """
    
    architecture = config['model'].get('architecture', 'resnet50')
    pretrained = config['model'].get('pretrained', True)
    num_classes = config['model'].get('num_classes', 4)
    dropout = config['model'].get('dropout', 0.5)
    
    if architecture == 'resnet50':
        # Charger ResNet50 pré-entraîné
        model = models.resnet50(pretrained=pretrained)
        
        # Remplacer la dernière couche pour le nombre de classes
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Architecture non supportée: {architecture}")
    
    return model

def freeze_backbone(model, freeze=True):
    """
    Gèle ou dégèle les couches du backbone ResNet.
    
    Args:
        model: modèle ResNet
        freeze: bool, si True gèle les couches
    """
    for param in model.parameters():
        param.requires_grad = not freeze
    
    # Dégeler la couche finale
    for param in model.fc.parameters():
        param.requires_grad = True
=======
"""
model.py
--------
Définit les architectures de modèles CNN, EfficientNet et ResNet50.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import (
    EfficientNetB0, ResNet50
)

from nom_du_projet.config import IMG_SIZE, LEARNING_RATE, MODEL_TYPE


def build_simple_cnn(input_shape, num_classes):
    """Petit CNN personnalisé pour la détection d'Alzheimer."""
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
    return model


def build_efficientnet(input_shape, num_classes):
    """Modèle EfficientNetB0 pré-entraîné."""
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg"
    )

    base_model.trainable = False  # pour le fine-tuning plus tard

    x = layers.Dense(128, activation="relu")(base_model.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model


def build_resnet50(input_shape, num_classes):
    """Modèle ResNet50 pré-entraîné."""
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg"
    )

    base_model.trainable = False

    x = layers.Dense(128, activation="relu")(base_model.output)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model


def get_model(model_type, input_shape, num_classes):
    """Retourne le modèle choisi selon le type défini dans config.py."""
    if model_type == "simple_cnn":
        model = build_simple_cnn(input_shape, num_classes)
    elif model_type == "efficientnet":
        model = build_efficientnet(input_shape, num_classes)
    elif model_type == "resnet50":
        model = build_resnet50(input_shape, num_classes)
    else:
        raise ValueError(f"❌ Type de modèle inconnu : {model_type}")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
>>>>>>> 3994636c730aad2d87d13b99a6031df3de3d57db
