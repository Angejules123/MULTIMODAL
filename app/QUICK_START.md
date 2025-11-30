# 🚀 Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies
```powershell
cd "e:\Master data science\MPDS3_2025\projet federal\projet"
python -m pip install -r requirements.txt
```

### 2. Train a Model (Debug Mode)
```powershell
python scripts/run_training.py --config configs/config.yaml --debug --epochs 1 --batch-size 8 --device cpu
```
⏱️ Takes ~2 minutes on CPU (1 epoch, small dataset)

### 3. Launch Streamlit App
```powershell
streamlit run app_streamlit.py
```
🌐 Opens at `http://localhost:8501`

---

## What You Can Do Now

### 🔮 Inference
1. Go to **Inférence** tab
2. Select the trained model
3. Upload an image (JPG/PNG)
4. Click **Prédiction**
5. See prediction + confidence

### 📊 Evaluation
1. Go to **Évaluation** tab
2. Select model & dataset split
3. Click **Évaluer le modèle**
4. View accuracy metrics

### 📈 Analytics
1. Go to **Statistiques** tab
2. Select a training run
3. View loss/accuracy curves
4. Check config used

---

## Project Structure

```
├── 🧠 MultimodalAI/          # Core code
├── 🏃 scripts/               # Training & evaluation
├── ⚙️ configs/               # YAML configuration
├── 📊 data/                  # Data splits
├── 🤖 models/                # Trained models
├── 📝 logs/                  # Training history
│
├── 🌐 app_streamlit.py       # Web interface
├── 📦 requirements.txt       # Dependencies
│
└── 📚 DOCUMENTATION
    ├── .github/copilot-instructions.md
    ├── STREAMLIT_GUIDE.md
    ├── PROJECT_STRUCTURE.md
    └── COMPLETION_CHECKLIST.md
```

---

## Common Commands

### Training
```powershell
# Standard
python scripts/run_training.py --config configs/config.yaml

# Debug (fast)
python scripts/run_training.py --config configs/config.yaml --debug

# Custom params
python scripts/run_training.py --config configs/config.yaml --epochs 100 --batch-size 16 --lr 0.0001

# Resume from checkpoint
python scripts/run_training.py --config configs/config.yaml --resume models/checkpoints/checkpoint_epoch_10.pth
```

### Web Interface
```powershell
# Launch app
streamlit run app_streamlit.py

# With options
streamlit run app_streamlit.py --logger.level=debug
```

### Evaluation
```powershell
python scripts/evaluate_model.py --model models/best/best_model.pth --test-dir data/processed/test
```

---

## First Time: Step-by-Step

### Step 1: Verify Setup ✅
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import streamlit; print('Streamlit: OK')"
```

### Step 2: Check Config
Open `configs/config.yaml` and verify:
- `data.train_dir`, `data.val_dir`, `data.test_dir` exist
- `model.num_classes` matches your data
- `training.batch_size` fits your GPU memory

### Step 3: Train Small Model
```powershell
# Start with debug mode
python scripts/run_training.py --config configs/config.yaml --debug --epochs 1
```

Expected output:
```
✅ DÉMARRAGE DE L'ENTRAÎNEMENT
📊 Chargement des données...
   ✅ Train: 16 samples
   ✅ Val: 8 samples
🏗️  Création du modèle...
   ✅ Paramètres: 23,587,714
```

### Step 4: Check Results
```powershell
# List logs
ls logs/
# Open latest log directory
ls logs/training_<TIMESTAMP>/
```

### Step 5: Launch App
```powershell
streamlit run app_streamlit.py
# Visit http://localhost:8501
```

---

## Troubleshooting

### ❌ "No module named 'torch'"
```powershell
python -m pip install torch torchvision
```

### ❌ "Config file not found"
Make sure you're in project root:
```powershell
cd "e:\Master data science\MPDS3_2025\projet federal\projet"
pwd  # Verify location
```

### ❌ "No GPU available"
Use CPU:
```powershell
python scripts/run_training.py --config configs/config.yaml --device cpu
```

### ❌ "Out of memory"
Reduce batch size:
```powershell
python scripts/run_training.py --config configs/config.yaml --batch-size 8
```

### ❌ Streamlit not starting
Install Streamlit:
```powershell
python -m pip install streamlit>=1.28.0
```

---

## Data Preparation

Your data should be organized:
```
data/processed/
├── train/
│   ├── images/          # .jpg files
│   └── labels/          # labels.txt (optional)
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Label format** (one of):
- `image.jpg 0` (class index)
- YOLO format: `0 x y w h`
- Directory structure: organize by class

See `MultimodalAI/data.py` for label loading logic.

---

## Configuration: Key Settings

Edit `configs/config.yaml`:

```yaml
# Data
data:
  processed_dir: data/processed
  num_classes: 4           # NonDemented, VeryMild, Mild, Moderate

# Training
training:
  batch_size: 32           # Reduce if OOM
  num_epochs: 50
  learning_rate: 0.001

# Hardware
hardware:
  device: auto             # auto, cpu, cuda, mps
  num_workers: 4           # Parallel data loading
```

---

## Performance Tips

### 🚀 Faster Training
- Use GPU: `--device cuda` or `device: cuda` in config
- Increase `num_workers`: `hardware.num_workers: 8`
- Reduce image size: `data.image_size: [224, 224]`

### 💾 Less Memory
- Reduce `batch_size`: `training.batch_size: 8`
- Use CPU: `--device cpu`
- Use mixed precision: `training.mixed_precision: true`

### 📊 Better Models
- Train longer: `--epochs 100`
- Lower LR: `--lr 0.0001`
- Use scheduler: `training.scheduler.type: cosine`

---

## Next Steps

1. ✅ Prepare your data in `data/processed/`
2. ✅ Update `configs/config.yaml` if needed
3. ✅ Train: `python scripts/run_training.py --debug --epochs 1`
4. ✅ Test app: `streamlit run app_streamlit.py`
5. ✅ Full training: `python scripts/run_training.py --epochs 50`
6. ✅ Deploy: Share Streamlit app link or containerize

---

## Documentation Links

- 📖 **Architecture**: `.github/copilot-instructions.md`
- 📖 **Streamlit**: `STREAMLIT_GUIDE.md`
- 📖 **Structure**: `PROJECT_STRUCTURE.md`
- 📖 **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- ✅ **Checklist**: `COMPLETION_CHECKLIST.md`

---

## Support

### Common Issues
1. Check `.github/copilot-instructions.md` for known pitfalls
2. Read `STREAMLIT_GUIDE.md` troubleshooting section
3. Review `PROJECT_STRUCTURE.md` file organization

### Getting Help
- Error in training? Check `logs/training_*/training.log`
- Model issue? Inspect `MultimodalAI/model.py`
- Data problem? Review `MultimodalAI/data.py` comments

---

**🎯 You're ready to go! Start with:**
```powershell
python scripts/run_training.py --config configs/config.yaml --debug --epochs 1
streamlit run app_streamlit.py
```

Enjoy! 🚀
