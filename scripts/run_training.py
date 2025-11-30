"""
Script principal d'entraînement pour MultimodalAI
Usage: python scripts/run_training.py [--config path/to/config.yaml]
"""

import sys
import os
from pathlib import Path

# Ajouter le dossier parent au path pour importer MultimodalAI
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from datetime import datetime
from tqdm import tqdm
import json

# Imports du package MultimodalAI
try:
    from MultimodalAI.model import create_model
    from MultimodalAI.data import create_datasets, create_dataloaders
    from MultimodalAI.train import train_epoch, validate_epoch
    from MultimodalAI.utils import (
        save_checkpoint, 
        load_checkpoint, 
        setup_logging, 
        get_device,
        count_parameters
    )
    from MultimodalAI.visualize import plot_training_curves
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Assurez-vous que le package MultimodalAI est correctement configuré")
    sys.exit(1)


def load_config(config_path):
    """Charge la configuration depuis un fichier YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Entraînement MultimodalAI')
    
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Chemin vers le fichier de configuration')
    parser.add_argument('--resume', type=str, default=None,
                       help='Chemin vers un checkpoint à reprendre')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (override config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Nombre d\'époques (override config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (override config)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device à utiliser')
    parser.add_argument('--debug', action='store_true',
                       help='Mode debug (petit subset des données)')
    
    return parser.parse_args()


def setup_training(config, args):
    """Configure l'environnement d'entraînement"""
    
    # Override config avec les args
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Device
    device = get_device(args.device)
    print(f"🖥️  Device: {device}")
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config['paths']['logs_dir']) / f"training_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(log_dir / 'training.log')
    logger.info(f"Démarrage de l'entraînement - {timestamp}")
    logger.info(f"Configuration: {args.config}")
    
    # TensorBoard
    if config['logging']['tensorboard']:
        writer = SummaryWriter(log_dir / 'tensorboard')
        logger.info(f"TensorBoard: {log_dir / 'tensorboard'}")
    else:
        writer = None
    
    # Sauvegarder la config utilisée
    with open(log_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return device, logger, writer, log_dir


def create_optimizer(model, config):
    """Crée l'optimiseur"""
    
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimiseur non supporté: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """Crée le scheduler de learning rate"""
    
    scheduler_config = config['training']['scheduler']
    scheduler_type = scheduler_config['type']
    
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['params']['T_max'],
            eta_min=scheduler_config['params']['eta_min']
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config['params'].get('step_size', 10),
            gamma=scheduler_config['params'].get('gamma', 0.1)
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=scheduler_config['params'].get('patience', 5),
            factor=scheduler_config['params'].get('factor', 0.1)
        )
    else:
        scheduler = None
    
    return scheduler


def train(config, args):
    """Fonction principale d'entraînement"""
    
    print("="*60)
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT")
    print("="*60)
    
    # Setup
    device, logger, writer, log_dir = setup_training(config, args)
    
    # Datasets et DataLoaders
    print("\n📊 Chargement des données...")
    train_dataset, val_dataset = create_datasets(config, args.debug)
    train_loader, val_loader = create_dataloaders(
        train_dataset, 
        val_dataset, 
        config,
        args.debug
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    print(f"   ✅ Train: {len(train_dataset)} samples")
    print(f"   ✅ Val: {len(val_dataset)} samples")
    
    # Modèle
    print("\n🏗️  Création du modèle...")
    model = create_model(config)
    model = model.to(device)
    
    num_params = count_parameters(model)
    logger.info(f"Paramètres du modèle: {num_params:,}")
    print(f"   ✅ Paramètres: {num_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        print(f"\n📥 Chargement du checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        logger.info(f"Reprise depuis l'époque {start_epoch}")
    
    # Training loop
    print("\n" + "="*60)
    print("🔥 ENTRAÎNEMENT EN COURS")
    print("="*60)
    
    num_epochs = config['training']['num_epochs']
    early_stopping_patience = config['training']['early_stopping']['patience']
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n📅 Époque {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch, logger
        )
        
        # Scheduler step
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Learning_rate', current_lr, epoch)
        
        # Print epoch summary
        print(f"\n📊 Résumé Époque {epoch+1}:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   LR: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"   🎉 Nouveau meilleur modèle! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        checkpoint_path = Path(config['paths']['checkpoints_dir']) / f"checkpoint_epoch_{epoch+1}.pth"
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'config': config
        }, is_best, checkpoint_path, config['paths']['best_model_dir'])
        
        # Early stopping
        if config['training']['early_stopping']['enabled']:
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping à l'époque {epoch+1}")
                logger.info(f"Early stopping après {patience_counter} époques sans amélioration")
                break
    
    # Fin de l'entraînement
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("="*60)
    print(f"🏆 Meilleure Val Acc: {best_val_acc:.2f}%")
    
    # Sauvegarder l'historique
    history_path = log_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, log_dir / 'training_curves.png')
    
    # Fermer TensorBoard
    if writer:
        writer.close()
    
    logger.info("Entraînement terminé avec succès")
    print(f"\n📁 Logs sauvegardés dans: {log_dir}")
    print(f"📁 Meilleur modèle: {config['paths']['best_model_dir']}/best_model.pth")


if __name__ == "__main__":
    args = parse_args()
    
    # Charger la config
    config = load_config(args.config)
    
    # Lancer l'entraînement
    try:
        train(config, args)
    except KeyboardInterrupt:
        print("\n⚠️  Entraînement interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()