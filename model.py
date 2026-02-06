"""
Model Definition Module - FIXED CONFIG NAMES
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class ConversationStateClassifier(nn.Module):
    """
    Hybrid Neuro-Symbolic Model for Conversation State Classification
    Combines BERT embeddings with a GRU for sequence modeling
    """
    
    def __init__(self, 
                 input_dim: int = None, 
                 hidden_dim: int = None, 
                 num_layers: int = None, 
                 num_classes: int = None, 
                 dropout: float = None):
        super(ConversationStateClassifier, self).__init__()
        
        # FIX: Changed Config.HIDDEN_DIM to Config.GRU_HIDDEN_DIM
        self.input_dim = input_dim or getattr(Config, 'BERT_DIM', 768)
        self.hidden_dim = hidden_dim or getattr(Config, 'GRU_HIDDEN_DIM', 256)
        self.num_layers = num_layers or getattr(Config, 'GRU_LAYERS', 2) # checking for likely naming match
        self.num_classes = num_classes or getattr(Config, 'NUM_STATES', 7)
        self.dropout_prob = dropout or getattr(Config, 'DROPOUT', 0.3)
        
        # Architecture components
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output layers (Bidirectional GRU doubles the hidden dim)
        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)
        
    def forward(self, embeddings, lengths):
        # Pack padded sequence for efficient processing
        lengths_cpu = lengths.cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        
        # Process with GRU
        packed_output, hidden = self.gru(packed_input)
        
        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=embeddings.size(1)
        )
        
        # Decode states
        x = self.fc1(output)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits, hidden


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # PyTorch's cross_entropy handles both 2D (N, C) and 3D (B, C, L) automatically.
        log_pt = F.cross_entropy(inputs, targets, weight=self.alpha, 
                               ignore_index=self.ignore_index, reduction='none')
        
        pt = torch.exp(-log_pt)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * log_pt
        
        if self.reduction == 'mean':
            valid_mask = targets != self.ignore_index
            if valid_mask.sum() > 0:
                return loss[valid_mask].mean()
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            valid_mask = targets != self.ignore_index
            return loss[valid_mask].sum()
        else:
            return loss


def create_model(device):
    """Factory function to create the model"""
    # FIX: Using safe getattr to pull from Config, with defaults as fallback
    # This prevents crashes if Config names vary slightly
    model = ConversationStateClassifier(
        input_dim=getattr(Config, 'BERT_DIM', 768),
        hidden_dim=getattr(Config, 'GRU_HIDDEN_DIM', 256),
        num_layers=getattr(Config, 'GRU_LAYERS', getattr(Config, 'NUM_LAYERS', 2)),
        num_classes=getattr(Config, 'NUM_STATES', 7),
        dropout=getattr(Config, 'DROPOUT', 0.3)
    )
    
    model = model.to(device)
    print(f"✓ Model created on {device}")
    
    return model


def create_loss_function(class_weights=None, use_focal_loss=True, device='cpu'):
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print("✓ Using Class Weights for loss calculation")
    
    if use_focal_loss:
        print("✓ Using Focal Loss (gamma=2.0)")
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, ignore_index=-100)
    else:
        print("Using standard CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        
    return criterion