
=======
# Multimodal-AI

"""# Diagnostic Précoce des Troubles Cognitifs
## Intelligence Artificielle Multimodale Explicable

**Auteurs:** ANGE JULES & MAHAMADOU SAADOU ADANANE

---

## 📋 Description

Système d'IA multimodale explicable (XAI) pour le diagnostic précoce de:
- Alzheimer
- TDAH
- Dépression
- Autisme
- Stress chronique

### Modalités Intégrées
- 🧠 **EEG**: Signaux électroencéphalographiques
- 🔬 **IRM**: Imagerie par résonance magnétique
- 🎯 **ADL**: Activités de la vie quotidienne

---

## 🚀 Installation

```bash
# Clone du repository
git clone <repo_url>
cd projet_federal

# Installation des dépendances
pip install -r requirements.txt

# OU avec Conda
conda env create -f environment.yml
conda activate projet-federal
```

---

## 📂 Structure

Voir `docs/PROJECT_STRUCTURE.md` pour la structure complète.

---

## 💻 Utilisation

### 1. Prétraitement

```python
from src.preprocessing.eeg_preprocessor import EEGPreprocessor

preprocessor = EEGPreprocessor('data/raw/eeg', 'data/processed/eeg')
result = preprocessor.preprocess_participant('4917218', 'P01')
```

### 2. Entraînement

```bash
python scripts/train_model.py --data data/processed/features/ml_ready
```

### 3. Évaluation

```bash
python scripts/evaluate_model.py --model models/trained/best_model.pth
```

---

## 📊 Objectifs

Conforme au cahier des charges:
- ✅ **Précision**: ≥ 85%
- ✅ **Sensibilité**: ≥ 80%
- ✅ **Spécificité**: ≥ 85%

---

## 📚 Documentation

- [Guide de démarrage](docs/tutorials/getting_started.md)
- [Guide prétraitement](docs/tutorials/preprocessing_guide.md)
- [API Documentation](docs/api/)

---

## 🧪 Tests

```bash
pytest tests/
```

---

## 📄 Licence

MIT License - Voir `LICENSE`

---

## 👥 Contact

- ANGE JULES
- MAHAMADOU SAADOU ADANANE
""""# MULTIMODAL" 
