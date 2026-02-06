"""
Configuration file for Causal Analysis of Conversational Defects
Contains all hyperparameters, paths, and system settings
"""

import torch
from pathlib import Path

class Config:
    """Central configuration for the entire pipeline"""
    
    # ==================== PATHS ====================
    DATA_PATH = "Conversational_Transcript_Dataset.json"
    MODEL_SAVE_DIR = Path("models")
    RESULTS_DIR = Path("results")
    LOGS_DIR = Path("logs")
    
    # Create directories if they don't exist
    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # ==================== STATE SPACE ====================
    NUM_STATES = 6
    STATE_NAMES = {
        0: "Greeting/Neutral",
        1: "Info_Exchange",
        2: "Problem_Statement",
        3: "Solution_Offer",
        4: "Friction/Pushback",
        5: "Escalation/Anger"
    }
    
    # Silver Labeling Keywords (case-insensitive)
    STATE_KEYWORDS = {
        0: ["hello", "hi", "thanks", "thank you", "okay", "ok", "bye", "goodbye", 
            "have a good day", "check", "sure", "yes", "no problem"],
        1: ["account number", "email", "verify", "address", "name", "phone", 
            "confirm", "identification", "security question", "zip code", "date of birth"],
        2: ["broken", "outage", "fail", "not working", "doesn't work", "issue", 
            "problem", "error", "charge", "charged", "billed", "billing error", 
            "overcharge", "down", "stopped working"],
        3: ["refund", "credit", "shipment", "ship", "technician", "fixed", 
            "fix", "resolve", "solution", "send", "dispatch", "schedule", 
            "appointment", "replacement", "compensate"],
        4: ["waste of time", "ridiculous", "waiting", "waited", "long time", 
            "nothing works", "frustrating", "frustrated", "unacceptable", 
            "disappointed", "still not fixed", "keep telling me"],
        5: ["manager", "supervisor", "sue", "lawyer", "attorney", "complaint", 
            "file a complaint", "better business bureau", "bbb", "cancel", 
            "canceling", "switching", "competitor", "disgust", "disgusted"]
    }
    
    # ==================== MODEL ARCHITECTURE ====================
    # BERT Encoder
    BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    BERT_DIM = 384  # Output dimension of MiniLM
    
    # GRU Parameters
    GRU_HIDDEN_DIM = 256
    GRU_NUM_LAYERS = 2
    GRU_DROPOUT = 0.3
    GRU_BIDIRECTIONAL = True  # Use bidirectional for better context
    
    # If bidirectional, actual hidden dim is 2x
    ACTUAL_HIDDEN_DIM = GRU_HIDDEN_DIM * (2 if GRU_BIDIRECTIONAL else 1)
    
    # ==================== TRAINING PARAMETERS ====================
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 7
    
    # Class imbalance handling
    USE_CLASS_WEIGHTS = True
    USE_FOCAL_LOSS = True
    FOCAL_LOSS_ALPHA = 1.0
    FOCAL_LOSS_GAMMA = 2.0
    
    # Gradient accumulation for effective larger batch size
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # Mixed precision training (RTX 4060 supports this)
    USE_MIXED_PRECISION = True
    
    # Regularization
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP_VALUE = 1.0
    
    # ==================== DATA SPLIT ====================
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    
    # ==================== MARKOV CHAIN PARAMETERS ====================
    MARKOV_ORDER = 1  # First-order Markov chain (can be increased to 2)
    SMOOTHING_ALPHA = 0.01  # Laplace smoothing to avoid zero probabilities
    
    # Hybrid risk calculation
    MARKOV_WEIGHT = 0.6  # Weight for Markov transition probability
    NEURAL_WEIGHT = 0.4  # Weight for GRU softmax prediction
    
    # ==================== HARDWARE CONFIGURATION ====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0  # Must be 0 on Windows to avoid multiprocessing issues
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # ==================== LOGGING ====================
    LOG_INTERVAL = 10  # Print training stats every N batches
    SAVE_BEST_MODEL = True
    SAVE_CHECKPOINTS = True
    CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs
    
    # ==================== VISUALIZATION ====================
    PLOT_DPI = 150
    PLOT_STYLE = "seaborn-v0_8-darkgrid"
    FIGURE_SIZE = (12, 8)
    
    @classmethod
    def print_config(cls):
        """Print all configuration settings"""
        print("=" * 80)
        print("CONFIGURATION SETTINGS")
        print("=" * 80)
        print(f"Device: {cls.DEVICE}")
        print(f"Mixed Precision: {cls.USE_MIXED_PRECISION}")
        print(f"Batch Size: {cls.BATCH_SIZE} (Effective: {cls.BATCH_SIZE * cls.GRADIENT_ACCUMULATION_STEPS})")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Num Epochs: {cls.NUM_EPOCHS}")
        print(f"GRU Hidden Dim: {cls.GRU_HIDDEN_DIM} ({'Bidirectional' if cls.GRU_BIDIRECTIONAL else 'Unidirectional'})")
        print(f"Focal Loss: {cls.USE_FOCAL_LOSS} (α={cls.FOCAL_LOSS_ALPHA}, γ={cls.FOCAL_LOSS_GAMMA})")
        print(f"Data Split: Train={cls.TRAIN_RATIO}, Val={cls.VAL_RATIO}, Test={cls.TEST_RATIO}")
        print("=" * 80)
