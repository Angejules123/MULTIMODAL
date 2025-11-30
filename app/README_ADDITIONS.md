

## 🏗️ Architecture du Projet

```
projet/
├── MultimodalAI/          # Package principal
├── data/                  # Données
│   ├── raw/              # Données brutes
│   ├── processed/        # train/val/test
│   └── cleaned/          # Intermédiaires
├── models/               # Modèles sauvegardés
├── logs/                 # Logs d'entraînement
├── configs/              # Fichiers de configuration
├── scripts/              # Scripts utilitaires
├── notebooks/            # Analyses et expérimentations
└── tests/                # Tests unitaires
```

## 🚀 Quick Start

### Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Configuration

1. Copiez `.env.template` en `.env` et configurez vos valeurs
2. Modifiez `configs/config.yaml` selon vos besoins
3. Organisez vos données avec `python 03_organize_data.py`

### Entraînement

```bash
# Avec configuration par défaut
python scripts/run_training.py

# Avec configuration personnalisée
python scripts/run_training.py --config configs/experiments/advanced.yaml

# Avec arguments en ligne de commande
python scripts/run_training.py --batch-size 64 --epochs 100
```

### Évaluation

```bash
python scripts/evaluate_model.py --model models/best/model.pth
```

## 📊 Monitoring

- **TensorBoard**: `tensorboard --logdir logs/tensorboard`
- **Logs**: Consultez `logs/training/` pour les logs détaillés

## 🔧 Scripts Utilitaires

- `01_create_project_structure.py`: Crée la structure du projet
- `02_create_gitignore.py`: Configure Git
- `03_organize_data.py`: Organise et split les données
- `04_create_config.py`: Génère les configurations
