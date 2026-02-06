"""
Evaluation Module
Comprehensive metrics and visualizations for model performance
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    f1_score, precision_recall_fscore_support
)
from typing import Dict, List, Tuple
from config import Config
from model import ConversationStateClassifier

class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations
    """
    
    def __init__(self, model: ConversationStateClassifier, device=None):
        """
        Args:
            model: Trained model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device or Config.DEVICE
        self.model.to(self.device)
    
    def evaluate(self, data_loader) -> Dict:
        """
        Comprehensive evaluation on a dataset
        
        Args:
            data_loader: DataLoader for evaluation
        
        Returns:
            Dictionary with all metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        print("\nEvaluating model...")
        
        with torch.no_grad():
            for embeddings, labels, lengths in data_loader:
                # Move to device
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                
                # Forward pass
                logits, _ = self.model(embeddings, lengths)
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                
                # Flatten and filter out padding
                batch_size, seq_len = predictions.shape
                for i in range(batch_size):
                    seq_length = lengths[i].item()
                    all_predictions.extend(predictions[i, :seq_length].cpu().numpy())
                    all_labels.extend(labels[i, :seq_length].cpu().numpy())
                    all_probabilities.extend(probabilities[i, :seq_length].cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        print("✓ Evaluation complete\n")
        
        return metrics
    
    def _calculate_metrics(self, 
                          labels: np.ndarray, 
                          predictions: np.ndarray,
                          probabilities: np.ndarray) -> Dict:
        """Calculate all evaluation metrics"""
        
        # Overall accuracy
        accuracy = (predictions == labels).mean()
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Weighted and macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions, labels=range(Config.NUM_STATES))
        
        # Build metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm,
            'per_class': {
                i: {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i],
                    'support': support[i]
                }
                for i in range(Config.NUM_STATES)
            },
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print human-readable metrics"""
        print("=" * 80)
        print("EVALUATION METRICS")
        print("=" * 80)
        
        print(f"\nOverall Accuracy: {metrics['accuracy'] * 100:.2f}%")
        
        print("\nMacro-Averaged Metrics:")
        print(f"  Precision: {metrics['precision_macro'] * 100:.2f}%")
        print(f"  Recall:    {metrics['recall_macro'] * 100:.2f}%")
        print(f"  F1-Score:  {metrics['f1_macro'] * 100:.2f}%")
        
        print("\nWeighted-Averaged Metrics:")
        print(f"  Precision: {metrics['precision_weighted'] * 100:.2f}%")
        print(f"  Recall:    {metrics['recall_weighted'] * 100:.2f}%")
        print(f"  F1-Score:  {metrics['f1_weighted'] * 100:.2f}%")
        
        print("\nPer-Class Metrics:")
        print("-" * 80)
        print(f"{'State':<5} {'Name':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 80)
        
        for i in range(Config.NUM_STATES):
            state_name = Config.STATE_NAMES[i]
            pc = metrics['per_class'][i]
            print(f"S{i:<4} {state_name:<25} "
                  f"{pc['precision']*100:>10.2f}% "
                  f"{pc['recall']*100:>10.2f}% "
                  f"{pc['f1']*100:>10.2f}% "
                  f"{int(pc['support']):>10}")
        
        print("=" * 80)
        
        # Check for the "Neutral Trap"
        s0_recall = metrics['per_class'][0]['recall']
        s4_recall = metrics['per_class'][4]['recall']
        s5_recall = metrics['per_class'][5]['recall']
        
        if s0_recall > 0.95 and (s4_recall < 0.3 or s5_recall < 0.3):
            print("\n⚠️  WARNING: Possible 'Neutral Trap' detected!")
            print("   Model may be over-predicting neutral states.")
            print(f"   S0 recall: {s0_recall*100:.1f}%")
            print(f"   S4 recall: {s4_recall*100:.1f}%")
            print(f"   S5 recall: {s5_recall*100:.1f}%")
            print("=" * 80)
        
        print()
    
    def plot_confusion_matrix(self, metrics: Dict, save_path: str = None):
        """Plot confusion matrix"""
        cm = metrics['confusion_matrix']
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(
            cm_normalized,
            annot=cm,  # Show counts
            fmt='d',
            cmap='Blues',
            xticklabels=[f"S{i}" for i in range(Config.NUM_STATES)],
            yticklabels=[f"S{i}" for i in range(Config.NUM_STATES)],
            cbar_kws={'label': 'Normalized Frequency'},
            ax=ax
        )
        
        ax.set_xlabel('Predicted State', fontsize=12)
        ax.set_ylabel('True State', fontsize=12)
        ax.set_title('Confusion Matrix (Normalized by Row)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_training_history(self, history: Dict, save_path: str = None):
        """Plot training history curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, [acc * 100 for acc in history['train_acc']], 
                        'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, [acc * 100 for acc in history['val_acc']], 
                        'r-', label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss difference (overfitting indicator)
        loss_diff = [val - train for val, train in zip(history['val_loss'], history['train_loss'])]
        axes[1, 1].plot(epochs, loss_diff, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Val Loss - Train Loss')
        axes[1, 1].set_title('Overfitting Indicator')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            print(f"✓ Training history saved to {save_path}")
        
        plt.close()
    
    def plot_per_class_performance(self, metrics: Dict, save_path: str = None):
        """Plot per-class performance metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        states = list(range(Config.NUM_STATES))
        state_labels = [f"S{i}\n{Config.STATE_NAMES[i][:15]}" for i in states]
        
        precision = [metrics['per_class'][i]['precision'] * 100 for i in states]
        recall = [metrics['per_class'][i]['recall'] * 100 for i in states]
        f1 = [metrics['per_class'][i]['f1'] * 100 for i in states]
        
        # Precision
        axes[0].bar(states, precision, color='skyblue', edgecolor='black')
        axes[0].set_xticks(states)
        axes[0].set_xticklabels(state_labels, rotation=45, ha='right')
        axes[0].set_ylabel('Precision (%)')
        axes[0].set_title('Precision by State')
        axes[0].set_ylim([0, 105])
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Recall
        axes[1].bar(states, recall, color='lightcoral', edgecolor='black')
        axes[1].set_xticks(states)
        axes[1].set_xticklabels(state_labels, rotation=45, ha='right')
        axes[1].set_ylabel('Recall (%)')
        axes[1].set_title('Recall by State')
        axes[1].set_ylim([0, 105])
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # F1-Score
        axes[2].bar(states, f1, color='lightgreen', edgecolor='black')
        axes[2].set_xticks(states)
        axes[2].set_xticklabels(state_labels, rotation=45, ha='right')
        axes[2].set_ylabel('F1-Score (%)')
        axes[2].set_title('F1-Score by State')
        axes[2].set_ylim([0, 105])
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            print(f"✓ Per-class performance saved to {save_path}")
        
        plt.close()


def evaluate_model(model: ConversationStateClassifier, 
                  test_loader,
                  training_history: Dict = None) -> Dict:
    """
    Convenience function for complete model evaluation
    
    Args:
        model: Trained model
        test_loader: Test data loader
        training_history: Optional training history for plotting
    
    Returns:
        Evaluation metrics dictionary
    """
    evaluator = ModelEvaluator(model)
    
    # Evaluate
    metrics = evaluator.evaluate(test_loader)
    
    # Print metrics
    evaluator.print_metrics(metrics)
    
    # Generate plots
    evaluator.plot_confusion_matrix(
        metrics, 
        save_path=Config.RESULTS_DIR / "confusion_matrix.png"
    )
    
    evaluator.plot_per_class_performance(
        metrics,
        save_path=Config.RESULTS_DIR / "per_class_performance.png"
    )
    
    if training_history:
        evaluator.plot_training_history(
            training_history,
            save_path=Config.RESULTS_DIR / "training_history.png"
        )
    
    return metrics
