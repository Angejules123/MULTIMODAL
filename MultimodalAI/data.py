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
