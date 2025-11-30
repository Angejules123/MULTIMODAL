<<<<<<< HEAD
"""Configuration du projet"""
=======
import pathlib
# ==============================================================================
# 0. PARAMÈTRES ET CHEMINS
# ==============================================================================

# Dossier Racine contenant les sous-dossiers de classes (MildDemented, NonDemented, etc.)
DATA_ROOT_DIR = 'MultimodalAI/Alzheimer/data/processed/train'


# Paramètres de l'image et du modèle
IMAGE_SIZE = (256, 256)  # Redimensionnement des images (Section 12: Resizing Images)
BATCH_SIZE = 32
EPOCHS = 2
RANDOM_STATE = 42

CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
NUM_CLASSES = len(CLASS_NAMES)

# Définir l'ordre des classes si nécessaire pour l'interprétation des résultats
# Keras chargera les classes par ordre alphanumérique, donc vérifiez l'ordre réel.

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "data/raw"
PROCESSED_DIR = DATA_DIR / "data/processed"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.h5"
>>>>>>> 3994636c730aad2d87d13b99a6031df3de3d57db
