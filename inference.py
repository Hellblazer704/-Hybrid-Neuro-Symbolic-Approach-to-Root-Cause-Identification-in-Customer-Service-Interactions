"""
Inference Module - FIXED
Apply trained model and root cause analysis to new conversations
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict
from config import Config
from model import ConversationStateClassifier
from embeddings import ConversationEmbedder
from markov_chain import RootCauseAnalyzer

class ConversationAnalyzer:
    """
    End-to-end conversation analysis pipeline
    Combines neural prediction with causal analysis
    """
    
    def __init__(self, 
                 model: ConversationStateClassifier,
                 embedder: ConversationEmbedder,
                 analyzer: RootCauseAnalyzer,
                 device=None):
        """
        Args:
            model: Trained neural network
            embedder: BERT embedder
            analyzer: Root cause analyzer with fitted Markov chain
            device: Device to run inference on
        """
        self.model = model
        self.embedder = embedder
        self.analyzer = analyzer
        self.device = device or Config.DEVICE
        
        self.model.to(self.device)
        self.model.eval()
    
    def analyze_conversation(self, conversation: List[Dict]) -> Dict:
        """
        Complete analysis pipeline for a single conversation
        """
        # Step 1: Embed conversation
        embeddings = self.embedder.embed_conversation(conversation)
        
        # Step 2: Neural prediction
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)
        embeddings_tensor = embeddings_tensor.to(self.device)
        
        # FIX: Create lengths tensor required by the model
        seq_len = embeddings_tensor.size(1)
        lengths = torch.tensor([seq_len], dtype=torch.long).cpu()
        
        with torch.no_grad():
            # FIX: Pass lengths to the model
            logits, _ = self.model(embeddings_tensor, lengths)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_states = torch.argmax(probabilities, dim=-1)
        
        # Convert to numpy
        predicted_states = predicted_states.squeeze(0).cpu().numpy().tolist()
        probabilities = probabilities.squeeze(0).cpu().numpy()
        
        # Step 3: Root cause analysis
        root_cause_result = self.analyzer.identify_root_cause(
            conversation,
            predicted_states,
            probabilities
        )
        
        # Step 4: Build complete result
        result = {
            'conversation': conversation,
            'predicted_states': predicted_states,
            'probabilities': probabilities,
            'root_cause_analysis': root_cause_result,
            'conversation_length': len(conversation)
        }
        
        return result
    
    def analyze_batch(self, conversations: List[List[Dict]]) -> List[Dict]:
        """Analyze multiple conversations"""
        results = []
        
        print(f"\nAnalyzing {len(conversations)} conversations...")
        
        for i, conversation in enumerate(conversations):
            result = self.analyze_conversation(conversation)
            results.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(conversations)} conversations")
        
        print(f"✓ Analysis complete\n")
        
        return results
    
    def visualize_conversation(self, result: Dict, save_path: str = None):
        """Create visualization of conversation analysis"""
        conversation = result['conversation']
        states = result['predicted_states']
        risks = result['root_cause_analysis']['full_risk_trajectory']
        deltas = result['root_cause_analysis']['delta_sequence']
        root_cause_turn = result['root_cause_analysis']['root_cause_turn']
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        turns = range(len(conversation))
        
        # Plot 1: State Sequence
        ax1 = axes[0]
        colors = ['blue', 'green', 'orange', 'purple', 'yellow', 'red']
        # Safety check for colors index
        state_colors = [colors[s] if s < len(colors) else 'gray' for s in states]
        
        ax1.scatter(turns, states, c=state_colors, s=100, zorder=3)
        ax1.plot(turns, states, 'k--', alpha=0.3, zorder=1)
        
        # Highlight root cause
        ax1.axvline(root_cause_turn, color='red', linestyle='--', 
                    linewidth=2, label='Root Cause', alpha=0.7)
        
        ax1.set_ylabel('State')
        ax1.set_title('Predicted State Sequence', fontweight='bold')
        ax1.set_yticks(range(Config.NUM_STATES))
        ax1.set_yticklabels([f"S{i}\n{Config.STATE_NAMES[i][:12]}" 
                             for i in range(Config.NUM_STATES)], fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Risk Trajectory
        ax2 = axes[1]
        ax2.plot(turns, risks, 'b-', linewidth=2, label='Escalation Risk')
        ax2.fill_between(turns, risks, alpha=0.3)
        
        # Highlight root cause
        ax2.axvline(root_cause_turn, color='red', linestyle='--', 
                    linewidth=2, label='Root Cause', alpha=0.7)
        
        # Mark escalation point if exists
        if result['root_cause_analysis']['escalated']:
            esc_turn = result['root_cause_analysis'].get('escalation_turn')
            if esc_turn is not None:
                ax2.axvline(esc_turn, color='darkred', linestyle=':', 
                           linewidth=2, label='Escalation', alpha=0.7)
        
        ax2.set_ylabel('Risk Score')
        ax2.set_title('Escalation Risk Trajectory', fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Risk Delta (Jump)
        ax3 = axes[2]
        colors_delta = ['green' if d <= 0 else 'red' for d in deltas]
        ax3.bar(turns, deltas, color=colors_delta, alpha=0.7)
        
        # Highlight root cause
        ax3.axvline(root_cause_turn, color='red', linestyle='--', 
                    linewidth=2, label='Root Cause', alpha=0.7)
        
        ax3.set_xlabel('Turn Number')
        ax3.set_ylabel('Risk Delta')
        ax3.set_title('Risk Jump (ΔRisk)', fontweight='bold')
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.close()
    
    def print_detailed_analysis(self, result: Dict):
        """Print detailed human-readable analysis"""
        print("\n" + "=" * 80)
        print("DETAILED CONVERSATION ANALYSIS")
        print("=" * 80)
        
        conversation = result['conversation']
        states = result['predicted_states']
        root_cause = result['root_cause_analysis']
        
        # Print conversation with states
        print("\nConversation Transcript:")
        print("-" * 80)
        
        for i, turn in enumerate(conversation):
            state_name = Config.STATE_NAMES[states[i]]
            speaker = turn['speaker']
            text = turn['text']
            
            # Highlight root cause turn
            marker = " ⚠️ ROOT CAUSE" if i == root_cause['root_cause_turn'] else ""
            
            print(f"\nTurn {i}: [{speaker}] - State: S{states[i]} ({state_name}){marker}")
            print(f'  "{text}"')
        
        # Print root cause analysis
        print("\n" + "=" * 80)
        self.analyzer.print_analysis(root_cause)
        
        # Print recommendations
        if root_cause['escalated']:
            print("RECOMMENDATIONS:")
            print("-" * 80)
            print("1. Review agent training for handling customer frustration")
            print("2. Implement earlier intervention when risk increases")
            print("3. Analyze similar escalation patterns across conversations")
            print("4. Consider escalation prevention playbook for this scenario")
            print("=" * 80 + "\n")


def load_trained_model(model_path: str, device=None) -> ConversationStateClassifier:
    """Load a trained model from checkpoint"""
    device = device or Config.DEVICE
    
    from model import create_model
    model = create_model(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from {model_path}")
    
    return model


def create_analyzer_pipeline(model_path: str, 
                             markov_path: str = None,
                             device=None) -> ConversationAnalyzer:
    """Create complete analysis pipeline"""
    device = device or Config.DEVICE
    
    # Load model
    model = load_trained_model(model_path, device)
    
    # Create embedder
    embedder = ConversationEmbedder(device=str(device))
    
    # Load or create Markov chain
    from markov_chain import MarkovChain, RootCauseAnalyzer
    
    markov = MarkovChain()
    if markov_path:
        markov.load(markov_path)
    else:
        print("⚠️  No Markov chain provided. Analyzer will use uniform probabilities.")
    
    analyzer = RootCauseAnalyzer(markov)
    
    # Create pipeline
    pipeline = ConversationAnalyzer(model, embedder, analyzer, device)
    
    print("✓ Analysis pipeline ready\n")
    
    return pipeline