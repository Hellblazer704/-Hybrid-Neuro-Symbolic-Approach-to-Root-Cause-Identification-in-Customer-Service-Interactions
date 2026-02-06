"""
Training Module
Handles model training with all optimizations
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple
from config import Config
from model import ConversationStateClassifier, create_loss_function

class Trainer:
    """
    Trainer class with support for:
    - Gradient accumulation
    - Mixed precision training
    - Early stopping
    - Learning rate scheduling
    - Checkpointing
    """
    
    def __init__(self,
                 model: ConversationStateClassifier,
                 train_loader,
                 val_loader,
                 class_weights=None,
                 device=None):
        """
        Args:
            model: The neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            class_weights: Optional class weights for loss function
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or Config.DEVICE
        
        # Loss function
        self.criterion = create_loss_function(class_weights)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )
        
        # Mixed precision training
        self.use_amp = Config.USE_MIXED_PRECISION and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✓ Mixed precision training enabled")
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            (average_loss, average_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, (embeddings, labels, lengths) in enumerate(pbar):
            # Move to device
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits, _ = self.model(embeddings, lengths)
                    
                    # Reshape for loss calculation
                    batch_size, seq_len, num_classes = logits.shape
                    logits_flat = logits.reshape(-1, num_classes)
                    labels_flat = labels.reshape(-1)
                    
                    loss = self.criterion(logits_flat, labels_flat)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
            else:
                # Standard forward pass
                logits, _ = self.model(embeddings, lengths)
                
                batch_size, seq_len, num_classes = logits.shape
                logits_flat = logits.reshape(-1, num_classes)
                labels_flat = labels.reshape(-1)
                
                loss = self.criterion(logits_flat, labels_flat)
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
                
                # Backward pass
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                if self.use_amp:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        Config.GRADIENT_CLIP_VALUE
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        Config.GRADIENT_CLIP_VALUE
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Calculate accuracy
            predictions = torch.argmax(logits_flat, dim=-1)
            mask = labels_flat != -100
            correct = ((predictions == labels_flat) & mask).sum().item()
            samples = mask.sum().item()
            
            total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
            total_correct += correct
            total_samples += samples
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * Config.GRADIENT_ACCUMULATION_STEPS:.4f}",
                'acc': f"{correct / samples * 100:.2f}%"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self) -> Tuple[float, float, Dict]:
        """
        Validate the model
        
        Returns:
            (average_loss, average_accuracy, per_class_metrics)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Per-class statistics
        class_correct = np.zeros(Config.NUM_STATES)
        class_total = np.zeros(Config.NUM_STATES)
        
        with torch.no_grad():
            for embeddings, labels, lengths in self.val_loader:
                # Move to device
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                
                # Forward pass
                logits, _ = self.model(embeddings, lengths)
                
                # Reshape for loss calculation
                batch_size, seq_len, num_classes = logits.shape
                logits_flat = logits.reshape(-1, num_classes)
                labels_flat = labels.reshape(-1)
                
                loss = self.criterion(logits_flat, labels_flat)
                
                # Calculate accuracy
                predictions = torch.argmax(logits_flat, dim=-1)
                mask = labels_flat != -100
                correct = ((predictions == labels_flat) & mask).sum().item()
                samples = mask.sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += samples
                
                # Per-class accuracy
                for class_id in range(Config.NUM_STATES):
                    class_mask = (labels_flat == class_id) & mask
                    if class_mask.sum() > 0:
                        class_correct[class_id] += ((predictions == labels_flat) & class_mask).sum().item()
                        class_total[class_id] += class_mask.sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_samples
        
        # Calculate per-class accuracy
        per_class_acc = {}
        for class_id in range(Config.NUM_STATES):
            if class_total[class_id] > 0:
                per_class_acc[class_id] = class_correct[class_id] / class_total[class_id]
            else:
                per_class_acc[class_id] = 0.0
        
        return avg_loss, avg_acc, per_class_acc
    
    def train(self, num_epochs: int = None):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train (uses config default if None)
        """
        num_epochs = num_epochs or Config.NUM_EPOCHS
        
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80 + "\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, per_class_acc = self.validate()
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc * 100:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Print per-class accuracy for minority classes
            print(f"  Per-class Val Accuracy:")
            for class_id in [4, 5]:  # Friction and Escalation
                acc = per_class_acc.get(class_id, 0.0)
                print(f"    S{class_id} ({Config.STATE_NAMES[class_id]}): {acc * 100:.2f}%")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_acc
                self.patience_counter = 0
                
                if Config.SAVE_BEST_MODEL:
                    self.save_checkpoint("best_model.pth", is_best=True)
            else:
                self.patience_counter += 1
            
            # Checkpoint saving
            if Config.SAVE_CHECKPOINTS and (epoch + 1) % Config.CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
            
            # Early stopping
            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                print(f"  Best validation loss: {self.best_val_loss:.4f}")
                print(f"  Best validation accuracy: {self.best_val_accuracy * 100:.2f}%")
                break
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_accuracy * 100:.2f}%")
        print("=" * 80 + "\n")
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        filepath = Config.MODEL_SAVE_DIR / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'history': self.history
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"✓ Saved best model to {filepath}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        filepath = Config.MODEL_SAVE_DIR / filename
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.history = checkpoint['history']
        
        print(f"✓ Loaded checkpoint from {filepath}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")
        print(f"  Best Val Acc: {self.best_val_accuracy * 100:.2f}%")
