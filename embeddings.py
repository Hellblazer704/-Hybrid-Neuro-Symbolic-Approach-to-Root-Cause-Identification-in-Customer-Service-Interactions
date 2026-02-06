"""
Text Embedding Module
Handles BERT-based vectorization of conversation turns
"""

from pathlib import Path
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from tqdm import tqdm
from config import Config

class ConversationEmbedder:
    """
    Wrapper for BERT-based sentence embedding
    Uses sentence-transformers for efficient encoding
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.model_name = model_name or Config.BERT_MODEL
        self.device = device or str(Config.DEVICE)
        
        print(f"Loading embedding model: {self.model_name}")
        print(f"Device: {self.device}")
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        # Verify embedding dimension
        test_embedding = self.model.encode(["test"])
        self.embedding_dim = test_embedding.shape[1]
        
        print(f"✓ Model loaded successfully")
        print(f"  Embedding dimension: {self.embedding_dim}")
        
        assert self.embedding_dim == Config.BERT_DIM, \
            f"Expected dimension {Config.BERT_DIM}, got {self.embedding_dim}"
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_conversation(self, conversation: List[Dict]) -> np.ndarray:
        """
        Embed all turns in a conversation
        
        Args:
            conversation: List of {"speaker": str, "text": str} dicts
        
        Returns:
            Array of shape (num_turns, embedding_dim)
        """
        texts = [turn["text"] for turn in conversation]
        
        with torch.no_grad():
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
        
        return embeddings
    
    def embed_all_conversations(self, conversations: List[List[Dict]], 
                                 batch_size: int = 32) -> List[np.ndarray]:
        """
        Embed all conversations in the dataset
        
        Args:
            conversations: List of conversation turn lists
            batch_size: Batch size for encoding
        
        Returns:
            List of embedding arrays, each of shape (num_turns, embedding_dim)
        """
        print(f"\nEmbedding {len(conversations)} conversations...")
        
        all_embeddings = []
        
        for conversation in tqdm(conversations, desc="Embedding conversations"):
            embeddings = self.embed_conversation(conversation)
            all_embeddings.append(embeddings)
        
        print(f"✓ Embedded {len(all_embeddings)} conversations")
        
        # Calculate total number of turns
        total_turns = sum(len(emb) for emb in all_embeddings)
        print(f"  Total turns embedded: {total_turns}")
        print(f"  Average turns per conversation: {total_turns/len(all_embeddings):.1f}\n")
        
        return all_embeddings
    
    def save_embeddings(self, embeddings, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        embeddings_array = np.array(embeddings, dtype=object)
        np.save(filepath, embeddings_array, allow_pickle=True)
        print(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> List[np.ndarray]:
        """Load embeddings from disk"""
        print(f"Loading embeddings from {filepath}...")
        embeddings = np.load(filepath, allow_pickle=True)
        print(f"✓ Loaded {len(embeddings)} conversation embeddings\n")
        return embeddings.tolist()


def precompute_embeddings(conversations: List[List[Dict]], 
                          save_path: str = None) -> List[np.ndarray]:
    """
    Convenience function to precompute and optionally save embeddings
    
    Args:
        conversations: List of conversations
        save_path: Optional path to save embeddings
    
    Returns:
        List of embedding arrays
    """
    embedder = ConversationEmbedder()
    embeddings = embedder.embed_all_conversations(conversations)
    
    if save_path:
        embedder.save_embeddings(embeddings, save_path)
    
    return embeddings
