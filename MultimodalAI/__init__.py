<<<<<<< HEAD
"""Package MultimodalAI"""
=======
"""
MultimodalAI Package
====================

Architecture multimodale explicable pour le diagnostic précoce
des troubles cognitifs et psychologiques (Alzheimer, TDAH, etc.)

Intègre :
- IRM structurelle (Vision Transformer)
- Signaux EEG (Conv1D + BiLSTM)
- Données comportementales ADL (MLP)
- Fusion multimodale avec cross-attention
"""

__version__ = "0.1.0"
__author__ = "Votre Nom"
__email__ = "votre.email@example.com"

# Imports principaux
from .model import (
    MultimodalAlzheimerNet,
    EEGEncoder,
    BehaviorEncoder,
    MultimodalFusionBlock
)

from .config import (
    Config,
    ModelConfig,
    TrainingConfig,
    DataConfig
)

from .data import (
    MultimodalDataset,
    MultimodalDataLoader,
    get_transforms
)

from .train import (
    Trainer,
    train_epoch,
    validate_epoch
)

from .utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    get_device,
    count_parameters
)

from .visualize import (
    plot_attention_maps,
    plot_training_curves,
    visualize_embeddings,
    generate_gradcam
)

__all__ = [
    # Model
    'MultimodalAlzheimerNet',
    'EEGEncoder',
    'BehaviorEncoder',
    'MultimodalFusionBlock',
    
    # Config
    'Config',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    
    # Data
    'MultimodalDataset',
    'MultimodalDataLoader',
    'get_transforms',
    
    # Training
    'Trainer',
    'train_epoch',
    'validate_epoch',
    
    # Utils
    'set_seed',
    'save_checkpoint',
    'load_checkpoint',
    'get_device',
    'count_parameters',
    
    # Visualize
    'plot_attention_maps',
    'plot_training_curves',
    'visualize_embeddings',
    'generate_gradcam',
]

# Informations sur le package
def get_info():
    """Retourne les informations sur le package"""
    info = {
        'name': 'MultimodalAI',
        'version': __version__,
        'author': __author__,
        'description': 'Architecture multimodale explicable pour diagnostic Alzheimer',
        'modalities': ['IRM', 'EEG', 'Comportement'],
        'features': [
            'Vision Transformer pour IRM',
            'Conv1D + BiLSTM pour EEG',
            'Cross-attention multimodale',
            'Explicabilité (XAI)',
            'Métriques cliniques'
        ]
    }
    return info

def print_info():
    """Affiche les informations sur le package"""
    info = get_info()
    print(f"\n{'='*70}")
    print(f"📦 {info['name']} v{info['version']}")
    print(f"{'='*70}")
    print(f"\n👤 Auteur: {info['author']}")
    print(f"📝 Description: {info['description']}")
    print(f"\n🧠 Modalités supportées:")
    for mod in info['modalities']:
        print(f"   • {mod}")
    print(f"\n✨ Fonctionnalités:")
    for feat in info['features']:
        print(f"   • {feat}")
    print(f"\n{'='*70}\n")
>>>>>>> 3994636c730aad2d87d13b99a6031df3de3d57db
