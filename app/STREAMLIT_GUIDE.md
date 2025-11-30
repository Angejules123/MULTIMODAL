# Streamlit App Usage Guide

## Installation

Ensure Streamlit is installed:

```powershell
python -m pip install streamlit>=1.28.0
# or
python scripts/setup_environment.py
```

## Running the App

From the project root:

```powershell
streamlit run app_streamlit.py
```

The app will open at `http://localhost:8501`

## Features

### 🔮 Inference Mode
- **Select a model**: Choose from trained models in `models/best/`
- **Upload image**: Upload an image (JPG, PNG) or take a photo with your camera
- **Get prediction**: View the predicted Alzheimer's stage and confidence score
- **Probability distribution**: See the confidence for each class

**Supported classes:**
- `NonDemented`: No signs of dementia
- `VeryMildDemented`: Very mild symptoms
- `MildDemented`: Mild symptoms
- `ModerateDemented`: Moderate symptoms

### 📊 Evaluation Mode
- **Select a model**: Choose a trained model
- **Choose dataset split**: Test on `val` (validation) or `test` (test) split
- **Run evaluation**: See overall accuracy and per-class metrics
- **Class-by-class results**: Detailed accuracy for each dementia stage

### 📈 Statistics Mode
- **View training history**: Select a training run from `logs/training_*/`
- **Loss curve**: Training vs validation loss over epochs
- **Accuracy curve**: Training vs validation accuracy over epochs
- **Configuration**: View the exact config used for that run

## Files & Directories

- **Models**: `models/best/*.pth` — trained model files
- **Logs**: `logs/training_*/history.json` — training metrics
- **Config**: `configs/config.yaml` — project configuration

## Troubleshooting

### "No models available"
- Train a model first: `python scripts/run_training.py --config configs/config.yaml`
- Check that models are saved in `models/best/` directory

### "Cannot import MultimodalAI"
- Ensure you're running from the project root: `streamlit run app_streamlit.py`
- Check that `MultimodalAI/` package exists and contains the modules

### "No training history found"
- Run at least one training before viewing statistics
- Check that logs are saved in `logs/training_*/` directory

## Tips

1. **Quick test**: Use `--debug --epochs 1` to train a small model for testing
2. **Batch inference**: Upload multiple images in sequence to compare predictions
3. **Export results**: Streamlit can download metrics data in CSV format
4. **Camera input**: Some browsers may not support camera access—use file upload as fallback

## Project Architecture

```
MultimodalAI/
├── data.py           # Data loading & augmentation
├── model.py          # Model architecture (ResNet50)
├── train.py          # Training loops
├── utils.py          # Utilities (device, logging, checkpoints)
└── visualize.py      # Visualization helpers

scripts/
├── run_training.py   # Main training script
├── evaluate_model.py # Evaluation script
└── setup_environment.py

app_streamlit.py       # This app!
```

## API Integration

To programmatically use the trained models:

```python
from MultimodalAI.model import create_model
from MultimodalAI.utils import get_device
import torch

# Load config
import yaml
config = yaml.safe_load(open('configs/config.yaml'))

# Load model
model = create_model(config)
device = get_device('auto')
checkpoint = torch.load('models/best/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
from MultimodalAI.data import get_transforms
_, val_transform = get_transforms()

# Your image processing code here...
```

## Performance Notes

- **First run**: May take a few seconds to load the model
- **Large datasets**: Evaluation on >1000 images may take longer
- **GPU**: If available, the app will use CUDA for faster inference
- **Memory**: Keep batch size reasonable if running on limited RAM

---

For more information, see `.github/copilot-instructions.md`
