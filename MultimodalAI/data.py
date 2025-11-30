<<<<<<< HEAD
"""Gestion des données"""
# MultimodalAI/data.py - CODE PRÊT POUR ENTRAÎNEMENT

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlzheimerDataset(Dataset):
    """Dataset pour la détection Alzheimer - PRÊT POUR ENTRAÎNEMENT"""
    
    def __init__(self, split='train', transform=None, data_root="data/processed"):
        self.split = split
        self.transform = transform
        self.data_root = Path(data_root)
        
        # Charger les images
        self.images_dir = self.data_root / split / "images"
        self.image_paths = list(self.images_dir.glob("*.jpg"))
        
        # Charger les labels
        self.labels = self._load_labels()
        
        print(f"✅ {split}: {len(self.image_paths)} images, {len(self.labels)} labels chargés")
    
    def _load_labels(self):
        """À ADAPTER SELON VOTRE FORMAT DE LABELS"""
        labels = {}
        labels_dir = self.data_root / self.split / "labels"
        
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*"))
            if label_files:
                label_file = label_files[0]
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                
                                # ⚠️ À ADAPTER SELON VOTRE FORMAT:
                                if len(parts) == 2:
                                    # FORMAT: "image.jpg class_index"
                                    filename, label = parts
                                    labels[filename] = int(label)
                                elif len(parts) == 1:
                                    # FORMAT: "image.jpg" seulement
                                    filename = parts[0]
                                    labels[filename] = 0  # Défaut
                                elif len(parts) == 5:
                                    # FORMAT YOLO: "class x y w h"
                                    filename = "???"  # À adapter
                                    labels[filename] = [float(x) for x in parts]
                except Exception as e:
                    print(f"❌ Erreur lecture labels: {e}")
        
        return labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Récupérer le label
        filename = image_path.name
        label = self.labels.get(filename, 0)  # 0 par défaut
        
        # Transformation
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# Transformations pour images médicales
def get_transforms():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

def get_data_loaders(batch_size=32):
    """Crée les DataLoaders pour l'entraînement"""
    train_transform, val_transform = get_transforms()
    
    train_dataset = AlzheimerDataset('train', transform=train_transform)
    val_dataset = AlzheimerDataset('val', transform=val_transform)
    test_dataset = AlzheimerDataset('test', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# Wrappers de compatibilité pour scripts/run_training.py
def create_datasets(config, debug=False):
    """
    Wrapper pour compatibilité avec scripts/run_training.py.
    Crée les datasets train et val selon les configurations.
    """
    train_transform, val_transform = get_transforms()
    
    train_dataset = AlzheimerDataset('train', transform=train_transform)
    val_dataset = AlzheimerDataset('val', transform=val_transform)
    
    # Mode debug: utiliser un petit subset
    if debug:
        train_dataset.image_paths = train_dataset.image_paths[:16]
        val_dataset.image_paths = val_dataset.image_paths[:8]
        print(f"🔧 Mode DEBUG: {len(train_dataset)} images train, {len(val_dataset)} images val")
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config, debug=False):
    """
    Wrapper pour compatibilité avec scripts/run_training.py.
    Crée les DataLoaders à partir des datasets.
    """
    batch_size = config['training']['batch_size']
    num_workers = config['hardware'].get('num_workers', 4)
    pin_memory = config['hardware'].get('pin_memory', True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
=======
# data.py — gestion des données et augmentation
import pandas as pd
import numpy as np
from PIL import Image
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import IMG_SIZE, SEED, BATCH_SIZE, DATA_DIR

def find_image_paths(data_dir=DATA_DIR):
    """Explore récursivement les dossiers et retourne un DataFrame (path, label)."""
    image_ext = ('*.png','*.jpg','*.jpeg')
    rows = []
    for ext in image_ext:
        for img in pathlib.Path(data_dir).rglob(ext):
            rows.append((str(img), img.parent.name))
    return pd.DataFrame(rows, columns=['path','label']).sample(frac=1, random_state=SEED)

def prepare_df_and_split(df, test_size=0.2):
    """Encode les labels et split train/val stratifié."""
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df["label_id"], random_state=SEED)
    return train_df, val_df, le

def get_data_generators(train_df, val_df):
    """Crée les DataGenerators pour Keras."""
    train_gen = ImageDataGenerator(
        rescale=1./255, rotation_range=15, zoom_range=0.1, horizontal_flip=True
    )
    val_gen = ImageDataGenerator(rescale=1./255)
    return (
        train_gen.flow_from_dataframe(train_df, x_col='path', y_col='label', target_size=IMG_SIZE, batch_size=BATCH_SIZE),
        val_gen.flow_from_dataframe(val_df, x_col='path', y_col='label', target_size=IMG_SIZE, batch_size=BATCH_SIZE)
    )
>>>>>>> 3994636c730aad2d87d13b99a6031df3de3d57db
