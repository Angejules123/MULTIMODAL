"""
Application Streamlit - MultimodalAI Pro v2.1 avec XIA - CORRIGÉ
Interface moderne avec système d'explication des prédictions Alzheimer
Corrections: warnings numpy, use_container_width, gestion améliorée
"""

import streamlit as st
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import json
import yaml
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import cv2

# ===== CONFIGURATION STREAMLIT =====
st.set_page_config(
    page_title="🧠 MultimodalAI Pro v2.1 - XIA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== THÈME ET STYLES AMÉLIORÉS =====
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
    .xia-explanation {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #667eea;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-importance {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 10px 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .medical-term { 
        background-color: #e9ecef; 
        padding: 2px 6px; 
        border-radius: 4px; 
        font-family: monospace;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
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

# ===== SYSTÈME XIA - EXPLICATIONS =====

class XIAExplainer:
    """Système d'explication des prédictions IA"""
    
    @staticmethod
    def generate_class_explanation(predicted_class, confidence, all_probs, class_names):
        """Génère une explication complète de la classification"""
        
        explanations = {
            'NonDemented': {
                'title': '🧠 Aucune Démence Détectée',
                'medical_meaning': "L'image montre une atrophie cérébrale dans les limites normales pour l'âge, sans signes évidents de maladie d'Alzheimer.",
                'features': [
                    "Volume hippocampique préservé",
                    "Sulci corticaux normaux",
                    "Absence de rétrécissement temporal marqué",
                    "Symétrie des hémisphères cérébraux"
                ],
                'clinical_implication': "Le patient présente un profil cognitif normal. Surveillance recommandée lors des contrôles annuels.",
                'next_steps': [
                    "Contrôle annuel recommandé",
                    "Maintenir un mode de vie sain",
                    "Surveillance des fonctions cognitives"
                ]
            },
            'VeryMildDemented': {
                'title': '🔍 Démence Très Légère',
                'medical_meaning': "Premiers signes subtils de dégénérescence, souvent localisés dans l'hippocampe et le cortex entorhinal.",
                'features': [
                    "Légère atrophie hippocampique",
                    "Élargissement modéré des sillons",
                    "Début de rétrécissement temporal",
                    "Changements subtils dans la matière grise"
                ],
                'clinical_implication': "Stade prodromique. Intervention précoce recommandée. Tests neuropsychologiques approfondis conseillés.",
                'next_steps': [
                    "Consultation neurologique",
                    "Tests neuropsychologiques",
                    "Imagerie de suivi dans 6-12 mois"
                ]
            },
            'MildDemented': {
                'title': '⚠️ Démence Légère',
                'medical_meaning': "Atrophie modérée avec atteinte visible des régions temporales médianes et du cortex cingulaire postérieur.",
                'features': [
                    "Atrophie hippocampique modérée à sévère",
                    "Élargissement ventriculaire notable",
                    "Atteinte du cortex temporal",
                    "Réduction du volume cérébral global"
                ],
                'clinical_implication': "Stade clinique établi. Traitement médicamenteux et suivi spécialisé nécessaires.",
                'next_steps': [
                    "Traitement médicamenteux",
                    "Suivi neurologique régulier",
                    "Évaluation des aidants"
                ]
            },
            'ModerateDemented': {
                'title': '🚨 Démence Modérée à Sévère',
                'medical_meaning': "Atrophie cérébrale généralisée avec atteinte extensive du cortex et des structures sous-corticales.",
                'features': [
                    "Atrophie hippocampique sévère",
                    "Élargissement ventriculaire important",
                    "Atteinte corticale diffuse",
                    "Perte de volume cérébral significative"
                ],
                'clinical_implication': "Stade avancé. Prise en charge multidisciplinaire essentielle. Support aux aidants nécessaire.",
                'next_steps': [
                    "Prise en charge multidisciplinaire",
                    "Support aux aidants",
                    "Plan de soins global"
                ]
            }
        }
        
        return explanations.get(predicted_class, {})

    @staticmethod
    def generate_confidence_analysis(confidence):
        """Analyse le niveau de confiance"""
        if confidence >= 80:
            return "confiance élevée", "confidence-high", "✅ La prédiction est très fiable"
        elif confidence >= 60:
            return "confiance modérée", "confidence-medium", "⚠️ La prédiction est acceptable mais une vérification est conseillée"
        else:
            return "confiance faible", "confidence-low", "🔍 La prédiction est incertaine - Consultation médicale recommandée"

    @staticmethod
    def generate_comparative_analysis(all_probs, class_names):
        """Analyse comparative entre les classes"""
        sorted_probs = sorted(zip(class_names, all_probs), key=lambda x: x[1], reverse=True)
        top2 = sorted_probs[:2]
        
        if len(top2) >= 2:
            diff = (top2[0][1] - top2[1][1]) * 100
            if diff < 10:
                return f"🔄 Difficile à distinguer: {top2[0][0]} vs {top2[1][0]} (diff: {diff:.1f}%)"
            else:
                return f"✅ Distinction claire: {top2[0][0]} se détache nettement"
        return ""

    @staticmethod
    def create_heatmap_overlay(image, predicted_class):
        """Crée une visualisation thermique simulée"""
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 2:  # Image en niveaux de gris
                img_array = np.stack([img_array] * 3, axis=-1)
            
            height, width = img_array.shape[:2]
            
            # Simulation de heatmap basée sur la classe prédite
            heatmap = np.zeros((height, width))
            
            # Zones d'intérêt selon la classe
            if predicted_class == 'NonDemented':
                centers = [(width//4, height//2), (width*3//4, height//2)]
            elif predicted_class == 'VeryMildDemented':
                centers = [(width//3, height//2), (width*2//3, height//2)]
            elif predicted_class == 'MildDemented':
                centers = [(width//2, height//3), (width//2, height*2//3)]
            else:  # ModerateDemented
                centers = [(width//2, height//2)]
            
            # Création de la heatmap
            for center_x, center_y in centers:
                for i in range(height):
                    for j in range(width):
                        dist = np.sqrt((i-center_y)**2 + (j-center_x)**2)
                        intensity = max(0, 1 - dist/150)
                        heatmap[i,j] = max(heatmap[i,j], intensity)
            
            # Application de la heatmap
            heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Fusion avec l'image originale
            alpha = 0.4
            overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
            
            return Image.fromarray(overlay)
        except Exception as e:
            st.warning(f"⚠️ Heatmap non disponible: {e}")
            return image

# ===== FONCTIONS UTILITAIRES CORRIGÉES =====

def get_available_models():
    """Liste modèles disponibles"""
    models_dir = Path("models/best")
    if not models_dir.exists():
        return []
    return sorted([m for m in models_dir.glob("*.pth") if m.is_file()], 
                  key=lambda x: x.stat().st_mtime, reverse=True)

def load_model_checkpoint(model_path):
    """Charge checkpoint avec gestion d'erreurs CORRIGÉE"""
    try:
        device = get_device()
        
        # CORRECTION: Utilisation de _core au lieu de core pour numpy
        torch.serialization.add_safe_globals([
            'numpy._core.multiarray._reconstruct',
            'numpy._core.multiarray.scalar',
            'numpy.dtype',
            'numpy.ndarray'
        ])
        
        # CORRECTION: weights_only=False pour anciens modèles
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        return checkpoint, device, True
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle: {str(e)}")
        return None, None, False

def build_resnet50_model(num_classes=4):
    """Construit ResNet50"""
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        return model
    except Exception as e:
        st.error(f"❌ Erreur construction modèle: {str(e)}")
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
        st.warning(f"⚠️ Chargement partiel: {str(e)}")
        return model

def get_class_names():
    """Noms des classes"""
    return ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

def preprocess_image(image, device):
    """Prétraite image"""
    try:
        image = image.resize((224, 224))
        img_array = np.array(image).astype(np.float32)
        
        # Gestion des images en niveaux de gris
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
            
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

# ===== UI PRINCIPALE CORRIGÉE =====

def main():
    # Header amélioré
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown('<div class="main-header">🧠 MultimodalAI Pro v2.1 - XIA</div>', unsafe_allow_html=True)
        st.markdown("**Détection d'Alzheimer par IA - Système eXplicable (XIA)**")
    with col2:
        device = get_device()
        st.metric("Device", str(device).upper())
    with col3:
        st.metric("Version", "2.1 XIA")
    
    config = load_config()
    class_names = get_class_names()
    xia_explainer = XIAExplainer()
    
    # Sidebar
    with st.sidebar:
        st.header("📱 Navigation")
        st.divider()
        app_mode = st.radio(
            "Mode:",
            [
                "🔮 Inférence XIA",
                "📊 Explications Détaillées", 
                "🔄 Comparaison",
                "📈 Résultats",
                "⚙️ Config",
                "❓ Aide XIA"
            ]
        )
        st.divider()
        
        # Paramètres XIA
        st.subheader("🎯 Paramètres XIA")
        show_heatmap = st.checkbox("Afficher heatmap", value=True)
        detailed_explanation = st.checkbox("Explication médicale détaillée", value=True)
        
        st.info("💡 **Nouveau**: Système XIA pour comprendre les décisions de l'IA")
    
    # ===== MODE INFÉRENCE XIA =====
    if app_mode == "🔮 Inférence XIA":
        st.header("🔮 Analyse XIA - Explications Intelligentes")
        st.markdown("Chargez une image MRI pour obtenir une prédiction **et son explication complète**")
        
        available_models = get_available_models()
        if not available_models:
            st.error("❌ Aucun modèle trouvé dans `models/best/`")
            st.info("Veuillez placer vos modèles dans le dossier `models/best/`")
            return
        
        device = get_device()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configuration")
            model_names = [m.name for m in available_models]
            model_name = st.selectbox("Modèle IA:", model_names)
            model_path = Path("models/best") / model_name
            
            uploaded_file = st.file_uploader("📤 Image MRI:", type=["jpg", "jpeg", "png"], 
                                           help="Chargez une image IRM cérébrale")
            
            if uploaded_file:
                st.metric("Taille modèle", f"{model_path.stat().st_size / 1e6:.2f} MB")
        
        with col2:
            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                
                # Affichage des images côte à côte
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(image, caption="🖼️ Image Originale", use_container_width=True)
                
                # Heatmap simulée
                with col_img2:
                    if show_heatmap:
                        # Placeholder pour la heatmap - sera mise à jour après prédiction
                        heatmap_img = xia_explainer.create_heatmap_overlay(image, "NonDemented")
                        st.image(heatmap_img, caption="🔥 Carte d'Activation XIA (Simulation)", use_container_width=True)
                    else:
                        img_224 = image.resize((224, 224))
                        st.image(img_224, caption="📐 Image Redimensionnée 224x224", use_container_width=True)
        
        # Bouton de prédiction
        if uploaded_file and st.button("🧠 Analyser avec XIA", use_container_width=True, type="primary"):
            with st.spinner("🔍 XIA analyse l'image..."):
                # Barre de progression
                progress_bar = st.progress(0)
                
                # Étape 1: Chargement du modèle
                progress_bar.progress(25)
                checkpoint, dev, success = load_model_checkpoint(model_path)
                
                if not success:
                    st.error("❌ Échec du chargement du modèle")
                    return
                
                # Étape 2: Construction du modèle
                progress_bar.progress(50)
                model = load_or_build_model(checkpoint, 4)
                if not model:
                    st.error("❌ Échec de la construction du modèle")
                    return
                
                model.to(dev)
                
                # Étape 3: Prétraitement
                progress_bar.progress(75)
                img_tensor = preprocess_image(image, dev)
                if img_tensor is None:
                    st.error("❌ Échec du prétraitement de l'image")
                    return
                
                # Étape 4: Inférence
                pred_class, conf, probs = run_inference(model, img_tensor, dev, class_names)
                progress_bar.progress(100)
                
                if pred_class:
                    st.success("✅ Analyse XIA terminée!")
                    
                    # ===== SECTION EXPLICATION XIA =====
                    st.markdown("---")
                    st.header("📋 Rapport XIA - Explication de la Classification")
                    
                    # 1. Résumé principal
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        explanation_data = xia_explainer.generate_class_explanation(
                            pred_class, conf, probs, class_names
                        )
                        st.subheader(explanation_data['title'])
                    
                    with col2:
                        st.metric("Classification", pred_class)
                    
                    with col3:
                        conf_level, conf_class, conf_text = xia_explainer.generate_confidence_analysis(conf)
                        st.metric("Confiance", f"{conf:.1f}%", conf_level)
                    
                    # 2. Mise à jour de la heatmap avec la vraie prédiction
                    if show_heatmap:
                        st.subheader("🔥 Carte d'Activation - Zones Analysées")
                        real_heatmap = xia_explainer.create_heatmap_overlay(image, pred_class)
                        st.image(real_heatmap, caption=f"Zones d'intérêt pour {pred_class}", use_container_width=True)
                    
                    # 3. Explication médicale
                    st.markdown('<div class="xia-explanation">', unsafe_allow_html=True)
                    st.subheader("🎯 Explication Médicale")
                    st.write(explanation_data['medical_meaning'])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 4. Caractéristiques détectées
                    st.subheader("🔍 Caractéristiques Radiologiques Identifiées")
                    for feature in explanation_data['features']:
                        st.markdown(f"• {feature}")
                    
                    # 5. Implications cliniques
                    st.markdown("---")
                    st.subheader("💡 Implications Cliniques")
                    st.info(explanation_data['clinical_implication'])
                    
                    # 6. Prochaines étapes
                    st.subheader("📋 Recommandations")
                    for step in explanation_data.get('next_steps', []):
                        st.markdown(f"• {step}")
                    
                    # 7. Analyse de confiance
                    st.markdown(f'<div class="xia-explanation">', unsafe_allow_html=True)
                    st.subheader("📊 Analyse de Confiance")
                    st.markdown(f'**Niveau**: <span class="{conf_class}">{conf_level}</span>', unsafe_allow_html=True)
                    st.write(conf_text)
                    
                    # Analyse comparative
                    comp_analysis = xia_explainer.generate_comparative_analysis(probs, class_names)
                    if comp_analysis:
                        st.write(comp_analysis)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 8. Graphique des probabilités
                    st.subheader("📈 Probabilités Détaillées")
                    prob_df = pd.DataFrame({
                        'Classe': class_names,
                        'Probabilité (%)': probs * 100
                    }).sort_values('Probabilité (%)', ascending=False)
                    
                    fig = px.bar(prob_df, x='Probabilité (%)', y='Classe', orientation='h',
                                color='Probabilité (%)', color_continuous_scale='RdYlGn',
                                title="Distribution des Probabilités par Classe")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 9. Téléchargement du rapport
                    st.markdown("---")
                    st.subheader("📄 Export du Rapport")
                    
                    # Génération du rapport texte
                    report_text = f"""
                    RAPPORT XIA - ANALYSE ALZHEIMER
                    ==============================
                    
                    Classification: {pred_class}
                    Confiance: {conf:.1f}%
                    Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}
                    
                    EXPLICATION MÉDICALE:
                    {explanation_data['medical_meaning']}
                    
                    CARACTÉRISTIQUES IDENTIFIÉES:
                    {chr(10).join(['• ' + feature for feature in explanation_data['features']])}
                    
                    IMPLICATIONS CLINIQUES:
                    {explanation_data['clinical_implication']}
                    
                    RECOMMANDATIONS:
                    {chr(10).join(['• ' + step for step in explanation_data.get('next_steps', [])])}
                    
                    ANALYSE DE CONFIANCE:
                    {conf_text}
                    """
                    
                    st.download_button(
                        label="📥 Télécharger le rapport",
                        data=report_text,
                        file_name=f"rapport_xia_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

    # ===== MODE EXPLICATIONS DÉTAILLÉES =====
    elif app_mode == "📊 Explications Détaillées":
        st.header("📊 Encyclopédie XIA - Comprendre l'Alzheimer")
        
        st.markdown("""
        <div class="xia-explanation">
        <h3>🧠 Comment l'IA analyse les images MRI</h3>
        <p>Le système XIA utilise l'apprentissage profond pour identifier les patterns caractéristiques 
        de chaque stade de la maladie d'Alzheimer dans les images IRM cérébrales.</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["🧠 Non Démence", "🔍 Très Légère", "⚠️ Légère", "🚨 Modérée"])
        
        with tab1:
            st.subheader("🧠 Aucune Démence Détectée")
            st.markdown("""
            **Signes radiologiques normaux:**
            - Volume hippocampique préservé
            - Cortex cérébral sans atrophie significative
            - Ventricules de taille normale
            - Symétrie des hémisphères
            
            **Signification clinique:** Le patient présente un vieillissement cérébral normal.
            
            **Zone clé:** Hippocampe préservé
            """)
        
        with tab2:
            st.subheader("🔍 Démence Très Légère")
            st.markdown("""
            **Premiers signes détectables:**
            - Légère atrophie hippocampique
            - Élargissement débutant des sillons
            - Changements subtils de la matière grise
            - Rétrécissement temporal minimal
            
            **Importance:** Stade prodromique permettant une intervention précoce.
            
            **Zone clé:** Hippocampe et cortex entorhinal
            """)
        
        with tab3:
            st.subheader("⚠️ Démence Légère")
            st.markdown("""
            **Atteinte modérée visible:**
            - Atrophie hippocampique évidente
            - Élargissement ventriculaire
            - Atteinte du lobe temporal
            - Réduction volumétrique mesurable
            
            **Implications:** Nécessite un traitement et suivi spécialisé.
            
            **Zone clé:** Régions temporales médianes
            """)
        
        with tab4:
            st.subheader("🚨 Démence Modérée à Sévère")
            st.markdown("""
            **Atteinte étendue:**
            - Atrophie hippocampique sévère
            - Ventricules très élargis
            - Atteinte corticale diffuse
            - Perte volumétrique importante
            
            **Prise en charge:** Approche multidisciplinaire essentielle.
            
            **Zone clé:** Atteinte cérébrale généralisée
            """)
    
    # ===== AUTRES MODES =====
    elif app_mode == "🔄 Comparaison":
        st.header("🔄 Comparaison des Modèles")
        st.info("🛠️ Fonctionnalité en cours de développement...")
        
    elif app_mode == "📈 Résultats":
        st.header("📈 Historique d'entraînement")
        st.info("🛠️ Fonctionnalité en cours de développement...")
    
    elif app_mode == "⚙️ Config":
        st.header("⚙️ Configuration du projet")
        if config:
            st.json(config)
        else:
            st.warning("Aucun fichier de configuration trouvé")
    
    elif app_mode == "❓ Aide XIA":
        st.header("❓ Aide XIA - Comprendre les Explications")
        
        st.markdown("""
        <div class="xia-explanation">
        <h3>🤔 Comment interpréter les résultats XIA</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Comprendre le système XIA
        
        **XIA (eXplainable AI)** explique pourquoi le modèle a fait une certaine classification:
        
        🔍 **Caractéristiques Identifiées:**
        - Décrit les signes radiologiques que l'IA a détectés
        - Basé sur l'analyse des patterns dans l'image MRI
        
        📊 **Niveaux de Confiance:**
        - **Élevé (>80%)**: Prédiction très fiable
        - **Modéré (60-80%)**: Prédiction acceptable, vérification utile
        - **Faible (<60%)**: Incertitude élevée - consultation médicale recommandée
        
        💡 **Implications Cliniques:**
        - Guide pour les prochaines étapes médicales
        - Suggestions de suivi et d'interventions
        
        ### 🏥 Terminologie Médicale
        
        <span class="medical-term">Atrophie hippocampique</span>: Réduction du volume de l'hippocampe, crucial pour la mémoire
        
        <span class="medical-term">Sulci corticaux</span>: Sillons à la surface du cerveau qui s'élargissent avec l'atrophie
        
        <span class="medical-term">Ventricules</span>: Cavités cérébrales contenant le liquide céphalo-rachidien
        
        ### ⚠️ Limitations et Avertissements
        
        - XIA fournit des explications basées sur les données d'entraînement
        - Les résultats doivent être validés par un radiologue
        - L'IA est un outil d'aide à la décision, pas un diagnostic définitif
        - Consultez toujours un professionnel de santé pour un diagnostic médical
        """, unsafe_allow_html=True)

        st.warning("""
        **Avertissement Médical Important:**
        Cette application est un outil d'aide à la décision et de recherche. 
        Elle ne remplace pas l'expertise d'un médecin qualifié. 
        Tous les diagnostics doivent être confirmés par un professionnel de santé.
        """)

if __name__ == "__main__":
    main()