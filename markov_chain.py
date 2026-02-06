"""
Markov Chain and Causal Analysis Module
Builds transition matrices and identifies root causes of escalations
"""

import numpy as np
import torch
from typing import List, Tuple, Dict
from config import Config

class MarkovChain:
    """
    First-order Markov Chain for conversation state transitions
    Represents the "Normal Physics" of conversations
    """
    
    def __init__(self, num_states: int = None, smoothing_alpha: float = None):
        """
        Args:
            num_states: Number of states in the chain
            smoothing_alpha: Laplace smoothing parameter
        """
        self.num_states = num_states or Config.NUM_STATES
        self.smoothing_alpha = smoothing_alpha or Config.SMOOTHING_ALPHA
        
        # Transition matrix: M[i][j] = P(S_t+1 = j | S_t = i)
        self.transition_matrix = np.zeros((self.num_states, self.num_states))
        self.transition_counts = np.zeros((self.num_states, self.num_states))
        
        self.is_fitted = False
    
    def fit(self, state_sequences: List[List[int]]):
        """
        Build transition matrix from state sequences
        
        Args:
            state_sequences: List of state sequences from conversations
        """
        print("\nBuilding Markov Transition Matrix...")
        
        # Reset counts
        self.transition_counts = np.zeros((self.num_states, self.num_states))
        
        # Count transitions
        total_transitions = 0
        for sequence in state_sequences:
            for i in range(len(sequence) - 1):
                current_state = sequence[i]
                next_state = sequence[i + 1]
                self.transition_counts[current_state][next_state] += 1
                total_transitions += 1
        
        # Apply Laplace smoothing and normalize
        for i in range(self.num_states):
            row_sum = self.transition_counts[i].sum() + self.smoothing_alpha * self.num_states
            
            for j in range(self.num_states):
                self.transition_matrix[i][j] = \
                    (self.transition_counts[i][j] + self.smoothing_alpha) / row_sum
        
        self.is_fitted = True
        
        print(f"✓ Transition matrix built from {total_transitions} transitions")
        self._print_statistics()
    
    def _print_statistics(self):
        """Print transition matrix statistics"""
        print("\nTransition Matrix Statistics:")
        print("-" * 80)
        
        # Most common transitions
        print("Top 10 Most Common Transitions:")
        transitions = []
        for i in range(self.num_states):
            for j in range(self.num_states):
                if self.transition_counts[i][j] > 0:
                    transitions.append((i, j, self.transition_counts[i][j], 
                                      self.transition_matrix[i][j]))
        
        transitions.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, count, prob in transitions[:10]:
            print(f"  S{i} → S{j}: {int(count):6d} times (P={prob:.3f})")
        
        # Escalation probabilities
        print("\nEscalation Probabilities (→ S5 Anger):")
        for i in range(self.num_states):
            prob = self.transition_matrix[i][5]
            count = int(self.transition_counts[i][5])
            state_name = Config.STATE_NAMES[i]
            print(f"  S{i} ({state_name:20s}) → S5: P={prob:.4f} ({count} times)")
        
        print("-" * 80)
    
    def get_transition_probability(self, from_state: int, to_state: int) -> float:
        """Get probability of transition from one state to another"""
        if not self.is_fitted:
            raise ValueError("Markov chain must be fitted before querying probabilities")
        
        return self.transition_matrix[from_state][to_state]
    
    def get_escalation_risk(self, current_state: int) -> float:
        """
        Get probability of escalating to S5 (Anger) from current state
        
        Args:
            current_state: Current conversation state
        
        Returns:
            Probability of transitioning to S5
        """
        return self.get_transition_probability(current_state, 5)
    
    def predict_next_state(self, current_state: int) -> int:
        """
        Predict most likely next state (deterministic prediction)
        
        Args:
            current_state: Current state
        
        Returns:
            Most likely next state
        """
        probabilities = self.transition_matrix[current_state]
        return int(np.argmax(probabilities))
    
    def save(self, filepath: str):
        """Save transition matrix to disk"""
        np.savez(filepath, 
                 transition_matrix=self.transition_matrix,
                 transition_counts=self.transition_counts)
        print(f"✓ Markov chain saved to {filepath}")
    
    def load(self, filepath: str):
        """Load transition matrix from disk"""
        data = np.load(filepath)
        self.transition_matrix = data['transition_matrix']
        self.transition_counts = data['transition_counts']
        self.is_fitted = True
        print(f"✓ Markov chain loaded from {filepath}")


class RootCauseAnalyzer:
    """
    Identifies the root cause of conversation escalations
    Uses Markov chain baseline to detect anomalous risk jumps
    """
    
    def __init__(self, markov_chain: MarkovChain):
        """
        Args:
            markov_chain: Fitted Markov chain representing normal conversation flow
        """
        self.markov_chain = markov_chain
        self.markov_weight = Config.MARKOV_WEIGHT
        self.neural_weight = Config.NEURAL_WEIGHT
    
    def calculate_hybrid_risk(self, 
                             state_sequence: List[int],
                             neural_probabilities: np.ndarray = None) -> np.ndarray:
        """
        Calculate hybrid risk combining Markov and neural predictions
        
        Args:
            state_sequence: Predicted state sequence
            neural_probabilities: Neural network softmax outputs [seq_len, num_states]
        
        Returns:
            Risk scores for each turn [seq_len]
        """
        seq_len = len(state_sequence)
        risks = np.zeros(seq_len)
        
        for t in range(seq_len):
            current_state = state_sequence[t]
            
            # Markov-based risk
            markov_risk = self.markov_chain.get_escalation_risk(current_state)
            
            # Neural-based risk (if provided)
            if neural_probabilities is not None:
                neural_risk = neural_probabilities[t][5]  # Probability of S5
            else:
                neural_risk = markov_risk
            
            # Hybrid risk
            risks[t] = (self.markov_weight * markov_risk + 
                       self.neural_weight * neural_risk)
            
            # If actual next state is S5, set risk to 1.0
            if t < seq_len - 1 and state_sequence[t + 1] == 5:
                risks[t] = 1.0
        
        return risks
    
    def identify_root_cause(self,
                           conversation: List[Dict],
                           state_sequence: List[int],
                           neural_probabilities: np.ndarray = None) -> Dict:
        """
        Identify the root cause turn that led to escalation
        
        Args:
            conversation: Original conversation turns
            state_sequence: Predicted state sequence
            neural_probabilities: Optional neural network predictions
        
        Returns:
            Dictionary with root cause analysis results
        """
        # Calculate risks
        risks = self.calculate_hybrid_risk(state_sequence, neural_probabilities)
        
        # Calculate delta (risk jump)
        deltas = np.zeros(len(risks))
        for t in range(len(risks) - 1):
            deltas[t] = risks[t + 1] - risks[t]
        
        # Find maximum positive delta (biggest risk increase)
        root_cause_idx = int(np.argmax(deltas))
        
        # Check if conversation actually escalated
        escalated = 5 in state_sequence
        
        # Build result dictionary
        result = {
            "escalated": escalated,
            "root_cause_turn": root_cause_idx,
            "root_cause_text": conversation[root_cause_idx]["text"],
            "root_cause_speaker": conversation[root_cause_idx]["speaker"],
            "root_cause_state": Config.STATE_NAMES[state_sequence[root_cause_idx]],
            "max_delta": float(deltas[root_cause_idx]),
            "risk_before": float(risks[root_cause_idx]),
            "risk_after": float(risks[root_cause_idx + 1]) if root_cause_idx < len(risks) - 1 else 1.0,
            "full_risk_trajectory": risks.tolist(),
            "state_sequence": state_sequence,
            "delta_sequence": deltas.tolist()
        }
        
        # Find when escalation actually occurred (if it did)
        if escalated:
            escalation_points = [i for i, state in enumerate(state_sequence) if state == 5]
            result["escalation_turn"] = escalation_points[0] if escalation_points else None
        
        return result
    
    def analyze_batch(self,
                     conversations: List[List[Dict]],
                     state_sequences: List[List[int]],
                     neural_probabilities: List[np.ndarray] = None) -> List[Dict]:
        """
        Analyze a batch of conversations
        
        Args:
            conversations: List of conversations
            state_sequences: List of state sequences
            neural_probabilities: Optional list of neural predictions
        
        Returns:
            List of analysis results
        """
        results = []
        
        for i, (conv, states) in enumerate(zip(conversations, state_sequences)):
            neural_probs = neural_probabilities[i] if neural_probabilities else None
            result = self.identify_root_cause(conv, states, neural_probs)
            results.append(result)
        
        return results
    
    def print_analysis(self, result: Dict):
        """Print human-readable analysis"""
        print("\n" + "=" * 80)
        print("ROOT CAUSE ANALYSIS")
        print("=" * 80)
        
        if result["escalated"]:
            print("⚠️  ESCALATION DETECTED")
            print(f"Escalation occurred at turn: {result.get('escalation_turn', 'N/A')}")
        else:
            print("✓ No escalation detected")
        
        print("\nRoot Cause Identification:")
        print(f"  Turn: {result['root_cause_turn']}")
        print(f"  Speaker: {result['root_cause_speaker']}")
        print(f"  State: {result['root_cause_state']}")
        print(f"  Risk Jump: {result['max_delta']:.3f}")
        print(f"  Risk Before: {result['risk_before']:.3f}")
        print(f"  Risk After: {result['risk_after']:.3f}")
        
        print("\nCritical Turn:")
        print(f'  "{result["root_cause_text"]}"')
        
        print("=" * 80 + "\n")


def build_markov_chain(state_sequences: List[List[int]]) -> MarkovChain:
    """
    Convenience function to build and return a fitted Markov chain
    
    Args:
        state_sequences: List of state sequences
    
    Returns:
        Fitted MarkovChain object
    """
    markov = MarkovChain()
    markov.fit(state_sequences)
    return markov
