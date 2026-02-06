"""
Utility Functions
Helper functions for various tasks
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict
from config import Config

def set_random_seed(seed: int = None):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    seed = seed or Config.RANDOM_SEED
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def save_conversation_to_json(conversation: List[Dict], filepath: str):
    """
    Save a single conversation to JSON file
    
    Args:
        conversation: List of turn dictionaries
        filepath: Output file path
    """
    data = {
        "transcript_id": "custom",
        "conversation": conversation
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump([data], f, indent=2, ensure_ascii=False)
    
    print(f"✓ Conversation saved to {filepath}")


def load_conversation_from_json(filepath: str) -> List[Dict]:
    """
    Load a single conversation from JSON
    
    Args:
        filepath: Input file path
    
    Returns:
        Conversation as list of turn dictionaries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list) and len(data) > 0:
        return data[0].get('conversation', [])
    elif isinstance(data, dict):
        return data.get('conversation', [])
    else:
        raise ValueError("Invalid conversation format")


def format_conversation_text(conversation: List[Dict]) -> str:
    """
    Format conversation as readable text
    
    Args:
        conversation: List of turn dictionaries
    
    Returns:
        Formatted text string
    """
    lines = []
    for i, turn in enumerate(conversation):
        speaker = turn.get('speaker', 'Unknown')
        text = turn.get('text', '')
        lines.append(f"[{speaker}] {text}")
    
    return '\n'.join(lines)


def create_sample_conversation() -> List[Dict]:
    """
    Create a sample conversation for testing
    
    Returns:
        Sample conversation
    """
    return [
        {"speaker": "Customer", "text": "Hello, I need help with my order"},
        {"speaker": "Agent", "text": "Hi! I'd be happy to help. What's your order number?"},
        {"speaker": "Customer", "text": "It's 12345"},
        {"speaker": "Agent", "text": "Thank you. I see your order. What's the issue?"},
        {"speaker": "Customer", "text": "It was supposed to arrive yesterday but it's still not here"},
        {"speaker": "Agent", "text": "I apologize for the delay. Let me check the shipping status"},
        {"speaker": "Customer", "text": "This is the third time this has happened!"},
        {"speaker": "Agent", "text": "I understand your frustration. I'll escalate this to our shipping team"},
        {"speaker": "Customer", "text": "I want a refund and I'm canceling my account!"},
    ]


def calculate_sequence_statistics(sequences: List[List[int]]) -> Dict:
    """
    Calculate statistics about sequences
    
    Args:
        sequences: List of sequences (e.g., state sequences)
    
    Returns:
        Dictionary with statistics
    """
    lengths = [len(seq) for seq in sequences]
    
    stats = {
        'num_sequences': len(sequences),
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'mean_length': np.mean(lengths) if lengths else 0,
        'median_length': np.median(lengths) if lengths else 0,
        'std_length': np.std(lengths) if lengths else 0,
    }
    
    return stats


def print_sequence_statistics(sequences: List[List[int]]):
    """Print sequence statistics"""
    stats = calculate_sequence_statistics(sequences)
    
    print("\nSequence Statistics:")
    print("-" * 40)
    print(f"Number of sequences: {stats['num_sequences']}")
    print(f"Length - Min: {stats['min_length']}, Max: {stats['max_length']}")
    print(f"Length - Mean: {stats['mean_length']:.1f}, Median: {stats['median_length']:.1f}")
    print(f"Length - Std Dev: {stats['std_length']:.1f}")
    print("-" * 40 + "\n")


def get_gpu_memory_info():
    """Print GPU memory information"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        print("\nGPU Information:")
        print("-" * 40)
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        print(f"Free: {(props.total_memory - torch.cuda.memory_allocated(device)) / 1024**3:.2f} GB")
        print("-" * 40 + "\n")
    else:
        print("\n⚠️  CUDA not available - using CPU\n")


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU memory cache cleared")


def save_results_summary(metrics: Dict, filepath: str):
    """
    Save evaluation metrics to JSON
    
    Args:
        metrics: Dictionary of metrics
        filepath: Output file path
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_metrics = convert_to_serializable(metrics)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"✓ Results summary saved to {filepath}")


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        Config.MODEL_SAVE_DIR,
        Config.RESULTS_DIR,
        Config.LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)
    
    print("✓ Directories created/verified")


def print_system_info():
    """Print system and environment information"""
    import platform
    import sys
    
    print("\n" + "=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA: Not Available (using CPU)")
    
    print("=" * 80 + "\n")


def validate_dataset_format(data: List[Dict]) -> bool:
    """
    Validate dataset format
    
    Args:
        data: Dataset to validate
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list")
    
    if len(data) == 0:
        raise ValueError("Dataset is empty")
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a dictionary")
        
        if 'conversation' not in item:
            raise ValueError(f"Item {i} missing 'conversation' field")
        
        conversation = item['conversation']
        if not isinstance(conversation, list):
            raise ValueError(f"Item {i}: 'conversation' must be a list")
        
        for j, turn in enumerate(conversation):
            if not isinstance(turn, dict):
                raise ValueError(f"Item {i}, turn {j}: not a dictionary")
            
            if 'speaker' not in turn or 'text' not in turn:
                raise ValueError(f"Item {i}, turn {j}: missing 'speaker' or 'text'")
    
    print(f"✓ Dataset format validated: {len(data)} conversations")
    return True


class ProgressTracker:
    """Simple progress tracker for batch operations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
    
    def update(self, n: int = 1):
        """Update progress"""
        self.current += n
        percentage = (self.current / self.total) * 100
        print(f"\r{self.description}: {self.current}/{self.total} ({percentage:.1f}%)", end='')
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def finish(self):
        """Mark as complete"""
        self.current = self.total
        self.update(0)


# Convenience function to initialize everything
def initialize_environment():
    """Initialize the complete environment"""
    print("\nInitializing environment...")
    
    # Set random seed
    set_random_seed()
    
    # Create directories
    create_directories()
    
    # Print system info
    print_system_info()
    
    # GPU memory info
    get_gpu_memory_info()
    
    print("✓ Environment initialized\n")
