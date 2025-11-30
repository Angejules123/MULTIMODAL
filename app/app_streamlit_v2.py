"""
Application Streamlit - MultimodalAI Pro v2.0
Interface moderne et avancée pour détection d'Alzheimer
Corrections: weights_only=False, build_resnet50, meilleure gestion d'erreurs
"""

import streamlit as st
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import pandas as pd
import json
import yaml
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# ===== CONFIGURATION STREAMLIT =====
st.set_page_config(
    page_title="🧠 MultimodalAI Pro v2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== THÈME ET STYLES =====
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .class-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .class-non { background-color: #d4edda; border-left: 5px solid #28a745; }
    .class-very { background-color: #d1ecf1; border-left: 5px solid #17a2b8; }
    .class-mild { background-color: #fff3cd; border-left: 5px solid #ffc107; }
    .class-mod { background-color: #f8d7da; border-left: 5px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# ===== CACHE =====
@st.cache_resource
def load_config():
    """Charge la configuration"""
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return None

@st.cache_resource
def get_device():
    """Retourne le device optimal"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# ===== FONCTIONS UTILITAIRES =====

def get_available_models():
    """Liste modèles disponibles"""
    models_dir = Path("models/best")
    if not models_dir.exists():
        return []
    return sorted([m for m in models_dir.glob("*.pth") if m.is_file()], 
                  key=lambda x: x.stat().st_mtime, reverse=True)

def load_model_checkpoint(model_path):
    """Charge checkpoint avec gestion d'erreurs"""
    try:
        device = get_device()
        
        # Ajouter globals sûrs
        torch.serialization.add_safe_globals([
            np.core.multiarray._reconstruct,
            np.core.multiarray._array_ufunc,
        ])
        
        # CORRECTION: weights_only=False pour anciens modèles
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        return checkpoint, device, True
    except Exception as e:
        return None, None, False

def build_resnet50_model(num_classes=4):
    """Construit ResNet50 - CORRECTION de l'import"""
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        return model
    except Exception as e:
        st.error(f"❌ Erreur modèle: {str(e)[:100]}")
        return None

def load_or_build_model(checkpoint, num_classes=4):
    """Charge ou construit modèle"""
    try:
        model = build_resnet50_model(num_classes)
        if not model:
            return None
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
        
        return model
    except Exception as e:
        st.warning(f"⚠️ Chargement partiel: {str(e)[:100]}")
        return model

def get_class_names():
    """Noms des classes"""
    return ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

def preprocess_image(image, device):
    """Prétraite image"""
    try:
        image = image.resize((224, 224))
        img_array = np.array(image).astype(np.float32)
        image_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image_tensor = (image_tensor / 255.0 - mean) / std
        
        return image_tensor.to(device)
    except Exception as e:
        st.error(f"❌ Erreur prétraitement: {e}")
        return None

def run_inference(model, image_tensor, device, class_names):
    """Inférence"""
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1)
        
        pred_class = class_names[pred.item()]
        confidence = probs[0, pred].item() * 100
        
        return pred_class, confidence, probs.cpu().numpy()[0]
    except Exception as e:
        st.error(f"❌ Erreur inférence: {e}")
        return None, None, None

# ===== UI PRINCIPALE =====

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="main-header">🧠 MultimodalAI Pro v2.0</div>', unsafe_allow_html=True)
        st.markdown("**Détection d'Alzheimer par IA - Système Avancé**")
    with col2:
        device = get_device()
        st.metric("Device", str(device).upper())
    
    config = load_config()
    class_names = get_class_names()
    
    # Sidebar
    with st.sidebar:
        st.header("📱 Navigation")
        st.divider()
        app_mode = st.radio(
            "Mode:",
            [
                "🔮 Inférence",
                "🔄 Comparaison",
                "📈 Résultats",
                "⚙️ Config",
                "❓ Aide",
                "ℹ️ À Propos"
            ]
        )
        st.divider()
        st.info("💡 Démarrez par: **Inférence** pour tester des images")
    
    # ===== MODE INFÉRENCE =====
    if app_mode == "🔮 Inférence":
        st.header("🔮 Analyse d'image MRI")
        st.markdown("Chargez une image MRI pour prédire le stade de démence")
        
        available_models = get_available_models()
        if not available_models:
            st.error("❌ Aucun modèle trouvé dans `models/best/`")
            st.info("Attendu: `models/best/*.pth`")
            return
        
        device = get_device()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("⚙️ Modèle")
            model_names = [m.name for m in available_models]
            model_name = st.selectbox("Sélectionner:", model_names)
            model_path = Path("models/best") / model_name
            st.metric("Taille", f"{model_path.stat().st_size / 1e6:.2f} MB")
        
        with col2:
            st.subheader("📤 Image")
            uploaded_file = st.file_uploader("JPG, PNG", type=["jpg", "jpeg", "png"])
        
        with col3:
            st.subheader("🎯 Paramètres")
            confidence_threshold = st.slider("Seuil confiance", 0, 100, 50) / 100
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_column_width=True)
            with col2:
                img_224 = image.resize((224, 224))
                st.image(img_224, caption="224x224", use_column_width=True)
            
            if st.button("🚀 Prédire", use_container_width=True, type="primary"):
                with st.spinner("⏳ Traitement..."):
                    checkpoint, dev, _ = load_model_checkpoint(model_path)
                    
                    if checkpoint:
                        model = load_or_build_model(checkpoint, 4)
                        if model:
                            model.to(dev)
                            img_tensor = preprocess_image(image, dev)
                            
                            if img_tensor is not None:
                                pred_class, conf, probs = run_inference(model, img_tensor, dev, class_names)
                                
                                if pred_class:
                                    st.success("✅ Succès!")
                                    
                                    # Résultat principal
                                    st.markdown("---")
                                    col1, col2, col3 = st.columns([1, 1, 1])
                                    
                                    with col1:
                                        color = "🟢" if conf > 80 else "🟡" if conf > 60 else "🔴"
                                        st.metric("Prédiction", pred_class)
                                    
                                    with col2:
                                        st.metric("Confiance", f"{conf:.1f}%", f"{color}")
                                    
                                    with col3:
                                        seuil_ok = "✅ OK" if conf >= confidence_threshold * 100 else "⚠️ Faible"
                                        st.metric("Seuil", seuil_ok)
                                    
                                    st.markdown("---")
                                    
                                    # Tous les scores
                                    st.subheader("📊 Tous les résultats")
                                    prob_df = pd.DataFrame({
                                        'Classe': class_names,
                                        'Score': probs * 100
                                    }).sort_values('Score', ascending=False)
                                    
                                    fig = px.bar(prob_df, x='Score', y='Classe', orientation='h',
                                                color='Score', color_continuous_scale='RdYlGn')
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Détails
                                    st.subheader("📋 Détails")
                                    for i, (cls, score) in enumerate(zip(class_names, probs)):
                                        bar = "█" * int(score * 50)
                                        st.write(f"**{cls}**: {bar} {score*100:.1f}%")
    
    # ===== MODE COMPARAISON =====
    elif app_mode == "🔄 Comparaison":
        st.header("🔄 Comparer les modèles")
        
        available_models = get_available_models()
        if len(available_models) < 2:
            st.warning("⚠️ Besoin d'au moins 2 modèles")
            return
        
        uploaded_file = st.file_uploader("Image pour tester:", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            device = get_device()
            
            results = {}
            
            for model_path in available_models:
                with st.spinner(f"Analyse avec {model_path.name}..."):
                    checkpoint, dev, _ = load_model_checkpoint(model_path)
                    if checkpoint:
                        model = load_or_build_model(checkpoint, 4)
                        if model:
                            model.to(dev)
                            img_tensor = preprocess_image(image, dev)
                            if img_tensor is not None:
                                pred, conf, probs = run_inference(model, img_tensor, dev, class_names)
                                if pred:
                                    results[model_path.name] = {'pred': pred, 'conf': conf}
            
            if results:
                st.subheader("📊 Résultats")
                comp_df = pd.DataFrame([
                    {'Modèle': m, 'Prédiction': d['pred'], 'Confiance': d['conf']}
                    for m, d in results.items()
                ])
                st.dataframe(comp_df, use_container_width=True)
    
    # ===== MODE RÉSULTATS =====
    elif app_mode == "📈 Résultats":
        st.header("📈 Historique d'entraînement")
        
        logs_dir = Path("logs")
        if logs_dir.exists():
            training_logs = sorted([d for d in logs_dir.glob("training_*") if d.is_dir()],
                                  key=lambda x: x.stat().st_mtime, reverse=True)
        else:
            training_logs = []
        
        if not training_logs:
            st.info("ℹ️ Pas d'historique. Lancez:")
            st.code("python scripts/run_training.py --config configs/config.yaml")
        else:
            selected = st.selectbox("Run:", [d.name for d in training_logs[:10]])
            log_dir = logs_dir / selected
            history_file = log_dir / "history.json"
            
            if history_file.exists():
                with open(history_file) as f:
                    history = json.load(f)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Loss")
                    loss_df = pd.DataFrame({
                        'Epoch': range(len(history['train_loss'])),
                        'Train': history['train_loss'],
                        'Val': history['val_loss']
                    })
                    st.line_chart(loss_df.set_index('Epoch'))
                
                with col2:
                    st.subheader("Accuracy")
                    acc_df = pd.DataFrame({
                        'Epoch': range(len(history['train_acc'])),
                        'Train': history['train_acc'],
                        'Val': history['val_acc']
                    })
                    st.line_chart(acc_df.set_index('Epoch'))
    
    # ===== MODE CONFIG =====
    elif app_mode == "⚙️ Config":
        st.header("⚙️ Configuration du projet")
        
        if config:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Model")
                st.json(config.get('model', {}))
            
            with col2:
                st.subheader("🎯 Training")
                st.json(config.get('training', {}))
            
            st.subheader("📂 Paths")
            st.json(config.get('paths', {}))
    
    # ===== MODE AIDE =====
    elif app_mode == "❓ Aide":
        st.header("❓ Aide et FAQ")
        
        st.markdown("""
        ### 🤔 Questions fréquentes
        
        **Q: L'erreur "Aucun modèle trouvé"?**
        
        R: Les modèles doivent être dans `models/best/`
        ```powershell
        Copy-Item models/alzheimer_model_final.pth -Destination models/best/
        ```
        
        **Q: Quelle est la meilleure confiance?**
        
        R: >80% est très bon, >60% est acceptable
        
        **Q: Puis-je comparer 2 modèles?**
        
        R: Oui! Utilisez le mode "Comparaison" avec une même image
        
        **Q: Comment entraîner un nouveau modèle?**
        
        R: 
        ```powershell
        python scripts/run_training.py --config configs/config.yaml
        ```
        """)
    
    # ===== MODE À PROPOS =====
    else:
        st.header("ℹ️ À Propos")
        
        st.markdown("""
        ### 🧠 MultimodalAI v2.0
        
        **Système d'IA pour détection des démences de type Alzheimer**
        
        ✅ **Objectifs:**
        - ML en pratique avec PyTorch
        - Collaboration équipe
        - Solutions réelles
        
        📊 **Architecture:**
        - Data Layer: Images 224x224
        - Model Layer: ResNet50
        - Training Layer: PyTorch
        - Deployment: Streamlit
        
        🔧 **Outils:**
        - PyTorch 2.0+
        - Streamlit 1.28+
        - TensorBoard
        
        📁 **Structure:**
        ```
        ├── MultimodalAI/      Package
        ├── scripts/           Scripts
        ├── configs/           Config YAML
        ├── data/              Images
        ├── models/best/       Modèles
        └── logs/              Historique
        ```
        
        **Version**: 2.0.0 | **Status**: ✅ Production Ready
        """)

if __name__ == "__main__":
    main()
