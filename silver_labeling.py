"""
Silver Labeling Module
Assigns states to conversation turns using keyword-based heuristics
"""

import numpy as np
from typing import List, Dict, Tuple
from config import Config

class SilverLabeler:
    """
    Heuristic-based state labeling for conversation turns
    Uses keyword matching with confidence scoring
    """
    
    def __init__(self):
        self.state_keywords = Config.STATE_KEYWORDS
        self.num_states = Config.NUM_STATES
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text for keyword matching"""
        return text.lower().strip()
    
    def _calculate_keyword_scores(self, text: str) -> np.ndarray:
        """
        Calculate keyword match scores for each state
        Returns array of shape (num_states,) with scores
        """
        normalized_text = self._normalize_text(text)
        scores = np.zeros(self.num_states)
        
        for state_id, keywords in self.state_keywords.items():
            for keyword in keywords:
                if keyword in normalized_text:
                    # Longer keywords get higher weight (more specific)
                    weight = len(keyword.split())
                    scores[state_id] += weight
        
        return scores
    
    def assign_state(self, text: str, speaker: str = None) -> Tuple[int, float]:
        """
        Assign a state to a conversation turn
        
        Args:
            text: The conversation turn text
            speaker: "Agent" or "Customer" (optional, for context)
        
        Returns:
            (state_id, confidence): The assigned state and confidence score
        """
        scores = self._calculate_keyword_scores(text)
        
        # If no keywords matched, assign neutral state
        if scores.sum() == 0:
            return 0, 0.5  # Low confidence neutral
        
        # Get the state with highest score
        state_id = int(np.argmax(scores))
        
        # Calculate confidence as normalized score
        total_score = scores.sum()
        confidence = scores[state_id] / total_score if total_score > 0 else 0.0
        
        # Apply speaker-specific heuristics
        if speaker == "Agent":
            # Agents are more likely to offer solutions
            if state_id == 2:  # Problem statement unusual for agent
                if scores[3] > 0:  # If solution keywords also present
                    state_id = 3
                    confidence = scores[3] / total_score
        
        elif speaker == "Customer":
            # Customers are more likely to state problems
            if state_id == 3:  # Solution offer unusual for customer
                if scores[2] > 0:  # If problem keywords also present
                    state_id = 2
                    confidence = scores[2] / total_score
        
        return state_id, confidence
    
    def label_conversation(self, conversation: List[Dict]) -> Tuple[List[int], List[float]]:
        """
        Label an entire conversation
        
        Args:
            conversation: List of {"speaker": str, "text": str} dicts
        
        Returns:
            (states, confidences): Lists of state IDs and confidence scores
        """
        states = []
        confidences = []
        
        for turn in conversation:
            state_id, confidence = self.assign_state(
                turn["text"], 
                turn.get("speaker", None)
            )
            states.append(state_id)
            confidences.append(confidence)
        
        return states, confidences
    
    def get_label_statistics(self, all_states: List[List[int]]) -> Dict:
        """
        Calculate statistics about the labeled dataset
        
        Args:
            all_states: List of state sequences
        
        Returns:
            Dictionary with label distribution statistics
        """
        # Flatten all states
        flat_states = [state for states in all_states for state in states]
        
        # Count occurrences
        state_counts = np.bincount(flat_states, minlength=self.num_states)
        total = len(flat_states)
        
        stats = {
            "total_turns": total,
            "state_counts": state_counts.tolist(),
            "state_percentages": (state_counts / total * 100).tolist(),
            "num_conversations": len(all_states)
        }
        
        return stats
    
    def print_label_statistics(self, all_states: List[List[int]]):
        """Print human-readable label statistics"""
        stats = self.get_label_statistics(all_states)
        
        print("\n" + "=" * 80)
        print("SILVER LABELING STATISTICS")
        print("=" * 80)
        print(f"Total Conversations: {stats['num_conversations']}")
        print(f"Total Turns: {stats['total_turns']}")
        print("\nState Distribution:")
        print("-" * 80)
        
        for state_id in range(self.num_states):
            state_name = Config.STATE_NAMES[state_id]
            count = stats['state_counts'][state_id]
            percentage = stats['state_percentages'][state_id]
            
            # Create visual bar
            bar_length = int(percentage / 2)  # Scale to 50 chars max
            bar = "█" * bar_length
            
            print(f"S{state_id} - {state_name:20s}: {count:6d} ({percentage:5.2f}%) {bar}")
        
        print("=" * 80)
        
        # Check for class imbalance
        min_pct = min(stats['state_percentages'])
        max_pct = max(stats['state_percentages'])
        imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
        
        if imbalance_ratio > 10:
            print(f"⚠️  WARNING: Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
            print("   → Class weighting and focal loss are recommended")
        print("=" * 80 + "\n")


def create_silver_labels(conversations: List[List[Dict]]) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Convenience function to label all conversations
    
    Args:
        conversations: List of conversations, each is a list of turn dicts
    
    Returns:
        (all_states, all_confidences): Lists of state sequences and confidence scores
    """
    labeler = SilverLabeler()
    all_states = []
    all_confidences = []
    
    for conversation in conversations:
        states, confidences = labeler.label_conversation(conversation)
        all_states.append(states)
        all_confidences.append(confidences)
    
    # Print statistics
    labeler.print_label_statistics(all_states)
    
    return all_states, all_confidences
