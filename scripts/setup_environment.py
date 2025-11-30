"""
Script d'installation de l'environnement
"""
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Installe les packages requis"""
    
    # Chercher le requirements.txt dans le dossier parent (racine du projet)
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    print(f"🔍 Recherche de {requirements_file}")
    
    if not requirements_file.exists():
        print("❌ Fichier requirements.txt introuvable")
        print("💡 Créez d'abord le fichier requirements.txt dans la racine du projet")
        return False
    
    print("🚀 Installation des dépendances...")
    
    try:
        # Installation avec pip
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True, check=True)
        
        print("✅ Dépendances installées avec succès !")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation: {e}")
        print(f"Sortie d'erreur: {e.stderr}")
        return False

def verify_installation():
    """Vérifie que les packages principaux sont installés"""
    
    packages_to_check = [
        "torch", "torchvision", "numpy", "pandas",
        "sklearn", "matplotlib", "PIL", "albumentations", "jupyter"
    ]
    
    print("\n🔍 Vérification de l'installation...")
    
    for package in packages_to_check:
        try:
            if package == "PIL":
                import PIL
                version = PIL.__version__
            elif package == "sklearn":
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'Version non disponible')
            
            print(f"✅ {package:15} | {version}")
            
        except ImportError as e:
            print(f"❌ {package:15} | NON INSTALLÉ - {e}")

if __name__ == "__main__":
    print("🔧 SETUP ENVIRONNEMENT MULTIMODAL AI")
    print("=" * 50)
    
    if install_requirements():
        verify_installation()
        
        print("\n🎉 Configuration terminée !")
        print("\n📋 Prochaines étapes:")
        print("   1. Lancez: python scripts/create_missing_files.py")
        print("   2. Commencez l'exploration: jupyter notebook")
        print("   3. Ouvrez: notebooks/01-exploration.ipynb")
    else:
        print("\n❌ Échec de l'installation")
        print("\n💡 Solutions:")
        print("   - Vérifiez que requirements.txt existe dans la racine du projet")
        print("   - Essayez: pip install -r requirements.txt manuellement")