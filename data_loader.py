"""
Data Loading and Preprocessing Module
Handles loading, splitting, and batching of conversation data
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from config import Config

class ConversationDataset(Dataset):
    """
    PyTorch Dataset for conversation sequences
    Each sample is a complete conversation with its state labels
    """
    
    def __init__(self, conversations: List[List[Dict]], states: List[List[int]], 
                 embeddings: List[np.ndarray] = None):
        """
        Args:
            conversations: List of conversation turn lists
            states: List of state label sequences
            embeddings: Precomputed BERT embeddings (optional)
        """
        self.conversations = conversations
        self.states = states
        self.embeddings = embeddings
        
        assert len(conversations) == len(states), \
            "Conversations and states must have same length"
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        state_seq = self.states[idx]
        
        # Convert to tensors
        state_tensor = torch.tensor(state_seq, dtype=torch.long)
        
        if self.embeddings is not None:
            embedding_tensor = torch.tensor(self.embeddings[idx], dtype=torch.float32)
            return embedding_tensor, state_tensor
        else:
            # Return raw conversation (embeddings will be computed on-the-fly)
            return conversation, state_tensor
    
    def get_sequence_length(self, idx):
        """Get the length of a conversation sequence"""
        return len(self.conversations[idx])


def collate_fn(batch):
    """
    Custom collate function for variable-length sequences
    Pads sequences to the same length within a batch
    """
    # Separate embeddings and labels
    embeddings = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Get sequence lengths
    lengths = torch.tensor([len(seq) for seq in labels])
    
    # Pad sequences
    padded_embeddings = torch.nn.utils.rnn.pad_sequence(
        embeddings, batch_first=True, padding_value=0.0
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100  # -100 is ignored by CrossEntropyLoss
    )
    
    return padded_embeddings, padded_labels, lengths


def load_dataset(file_path: str) -> List[Dict]:
    """
    Load the conversation dataset from JSON file
    
    Args:
        file_path: Path to the JSON dataset file
    
    Returns:
        List of conversation dictionaries
    """
    print(f"Loading dataset from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ Loaded {len(data)} conversations")
        return data
    
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}\n"
            f"Please ensure the file exists in the current directory."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


def extract_conversations(data):
    """
    Extract conversation turn lists from dataset
    """
    # FIX: Handle the "transcripts" key if it exists
    if isinstance(data, dict):
        if "transcripts" in data:
            print("✓ Found 'transcripts' key in dataset")
            data = data["transcripts"]
        else:
            # Fallback: wrap single dict in list if it's not the list we expect
            data = [data]
            
    conversations = []
    
    for item in data:
        # Safety check: Ensure item is a dictionary
        if not isinstance(item, dict):
            continue

        if "conversation" in item:
            conversations.append(item["conversation"])
        else:
            # Safe access using .get() now that we know item is a dict
            print(f"⚠️  Warning: Missing 'conversation' field in item: {item.get('transcript_id', 'unknown')}")
    
    print(f"✓ Extracted {len(conversations)} valid conversations")
    
    # Filter out empty conversations
    conversations = [conv for conv in conversations if len(conv) > 0]
    print(f"✓ Filtered to {len(conversations)} non-empty conversations")
    
    return conversations


def split_dataset(conversations: List[List[Dict]], 
                  states: List[List[int]],
                  embeddings: List[np.ndarray] = None) -> Tuple:
    """
    Split dataset into train, validation, and test sets
    
    Args:
        conversations: List of conversations
        states: List of state sequences
        embeddings: Optional precomputed embeddings
    
    Returns:
        Tuple of (train_data, val_data, test_data)
        Each is a tuple of (conversations, states, embeddings)
    """
    # First split: train+val vs test
    train_val_conv, test_conv, train_val_states, test_states = train_test_split(
        conversations, states,
        test_size=Config.TEST_RATIO,
        random_state=Config.RANDOM_SEED
    )
    
    # Second split: train vs val
    val_ratio_adjusted = Config.VAL_RATIO / (1 - Config.TEST_RATIO)
    train_conv, val_conv, train_states, val_states = train_test_split(
        train_val_conv, train_val_states,
        test_size=val_ratio_adjusted,
        random_state=Config.RANDOM_SEED
    )
    
    # Handle embeddings if provided
    if embeddings is not None:
        train_val_emb, test_emb = train_test_split(
            embeddings,
            test_size=Config.TEST_RATIO,
            random_state=Config.RANDOM_SEED
        )
        train_emb, val_emb = train_test_split(
            train_val_emb,
            test_size=val_ratio_adjusted,
            random_state=Config.RANDOM_SEED
        )
    else:
        train_emb, val_emb, test_emb = None, None, None
    
    print(f"\n{'='*80}")
    print("DATA SPLIT SUMMARY")
    print(f"{'='*80}")
    print(f"Train: {len(train_conv)} conversations ({len(train_conv)/len(conversations)*100:.1f}%)")
    print(f"Val:   {len(val_conv)} conversations ({len(val_conv)/len(conversations)*100:.1f}%)")
    print(f"Test:  {len(test_conv)} conversations ({len(test_conv)/len(conversations)*100:.1f}%)")
    print(f"{'='*80}\n")
    
    return (
        (train_conv, train_states, train_emb),
        (val_conv, val_states, val_emb),
        (test_conv, test_states, test_emb)
    )


def create_dataloaders(train_data: Tuple, val_data: Tuple, test_data: Tuple) -> Tuple:
    """
    Create PyTorch DataLoaders for train, val, and test sets
    
    Args:
        train_data, val_data, test_data: Tuples of (conversations, states, embeddings)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Unpack data
    train_conv, train_states, train_emb = train_data
    val_conv, val_states, val_emb = val_data
    test_conv, test_states, test_emb = test_data
    
    # Create datasets
    train_dataset = ConversationDataset(train_conv, train_states, train_emb)
    val_dataset = ConversationDataset(val_conv, val_states, val_emb)
    test_dataset = ConversationDataset(test_conv, test_states, test_emb)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_fn
    )
    
    print(f"✓ Created DataLoaders:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches\n")
    
    return train_loader, val_loader, test_loader


def compute_class_weights(all_states: List[List[int]]) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced data
    
    Args:
        all_states: List of state sequences
    
    Returns:
        Tensor of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Flatten all states
    flat_states = np.array([state for states in all_states for state in states])
    
    # Compute weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(Config.NUM_STATES),
        y=flat_states
    )
    
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    
    print("Class Weights:")
    for i, weight in enumerate(class_weights):
        print(f"  S{i} ({Config.STATE_NAMES[i]}): {weight:.3f}")
    print()
    
    return weights_tensor
