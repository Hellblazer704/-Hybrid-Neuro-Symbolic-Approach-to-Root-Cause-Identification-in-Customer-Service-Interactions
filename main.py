"""
Main Pipeline
Complete end-to-end pipeline for Causal Analysis of Conversational Defects
"""

import torch
import numpy as np
from pathlib import Path
import argparse

from config import Config
from data_loader import (
    load_dataset, extract_conversations, split_dataset,
    create_dataloaders, compute_class_weights
)
from silver_labeling import create_silver_labels
from embeddings import precompute_embeddings
from model import create_model, create_loss_function
from train import Trainer
from evaluate import evaluate_model
from markov_chain import build_markov_chain
from inference import create_analyzer_pipeline

def main():
    """Complete training and evaluation pipeline"""
    
    print("\n" + "="*80)
    print("CAUSAL ANALYSIS OF CONVERSATIONAL DEFECTS")
    print("Hybrid Neuro-Symbolic Architecture")
    print("="*80 + "\n")
    
    # Print configuration
    Config.print_config()
    
    # ========== STEP 1: DATA LOADING ==========
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80 + "\n")
    
    try:
        data = load_dataset(Config.DATA_PATH)
        conversations = extract_conversations(data)
    except FileNotFoundError:
        print("\n⚠️  ERROR: Dataset file not found!")
        print(f"Expected location: {Config.DATA_PATH}")
        print("\nCreating a DEMO dataset for testing purposes...\n")
        conversations = create_demo_dataset()
    
    # ========== STEP 2: SILVER LABELING ==========
    print("\n" + "="*80)
    print("STEP 2: SILVER LABELING")
    print("="*80 + "\n")
    
    all_states, all_confidences = create_silver_labels(conversations)
    
    # ========== STEP 3: EMBEDDING ==========
    print("\n" + "="*80)
    print("STEP 3: BERT EMBEDDING")
    print("="*80 + "\n")
    
    embeddings_path = Config.MODEL_SAVE_DIR / "embeddings.npy"
    
    if embeddings_path.exists():
        print(f"Found cached embeddings at {embeddings_path}")
        from embeddings import ConversationEmbedder
        embedder = ConversationEmbedder()
        all_embeddings = embedder.load_embeddings(embeddings_path)
    else:
        all_embeddings = precompute_embeddings(conversations, save_path=embeddings_path)
    
    # ========== STEP 4: DATA SPLITTING ==========
    print("\n" + "="*80)
    print("STEP 4: DATA SPLITTING")
    print("="*80 + "\n")
    
    train_data, val_data, test_data = split_dataset(
        conversations, all_states, all_embeddings
    )
    
    # ========== STEP 5: CREATE DATALOADERS ==========
    print("\n" + "="*80)
    print("STEP 5: CREATING DATALOADERS")
    print("="*80 + "\n")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data
    )
    
    # ========== STEP 6: COMPUTE CLASS WEIGHTS ==========
    print("\n" + "="*80)
    print("STEP 6: COMPUTING CLASS WEIGHTS")
    print("="*80 + "\n")
    
    train_states = train_data[1]
    class_weights = compute_class_weights(train_states)
    
    # ========== STEP 7: BUILD MARKOV CHAIN ==========
    print("\n" + "="*80)
    print("STEP 7: BUILDING MARKOV CHAIN")
    print("="*80 + "\n")
    
    markov_chain = build_markov_chain(train_states)
    
    # Save Markov chain
    markov_path = Config.MODEL_SAVE_DIR / "markov_chain.npz"
    markov_chain.save(markov_path)
    
    # ========== STEP 8: CREATE MODEL ==========
    print("\n" + "="*80)
    print("STEP 8: CREATING MODEL")
    print("="*80 + "\n")
    
    model = create_model(Config.DEVICE)
    
    # ========== STEP 9: TRAINING ==========
    print("\n" + "="*80)
    print("STEP 9: TRAINING")
    print("="*80 + "\n")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=Config.DEVICE
    )
    
    trainer.train(Config.NUM_EPOCHS)
    
    # ========== STEP 10: EVALUATION ==========
    print("\n" + "="*80)
    print("STEP 10: MODEL EVALUATION")
    print("="*80 + "\n")
    
    # Load best model
    best_model_path = Config.MODEL_SAVE_DIR / "best_model.pth"
    if best_model_path.exists():
        trainer.load_checkpoint("best_model.pth")
    
    test_metrics = evaluate_model(
        model=trainer.model,
        test_loader=test_loader,
        training_history=trainer.history
    )
    
    # ========== STEP 11: ROOT CAUSE ANALYSIS DEMO ==========
    print("\n" + "="*80)
    print("STEP 11: ROOT CAUSE ANALYSIS DEMO")
    print("="*80 + "\n")
    
    # Create analysis pipeline
    pipeline = create_analyzer_pipeline(
        model_path=best_model_path,
        markov_path=markov_path,
        device=Config.DEVICE
    )
    
    # Analyze a few test conversations
    test_conversations = test_data[0][:5]  # First 5 test conversations
    
    print("Analyzing sample conversations from test set...\n")
    
    for i, conversation in enumerate(test_conversations):
        print(f"\n{'='*80}")
        print(f"SAMPLE CONVERSATION {i+1}")
        print(f"{'='*80}")
        
        result = pipeline.analyze_conversation(conversation)
        
        # Print analysis
        pipeline.print_detailed_analysis(result)
        
        # Save visualization
        viz_path = Config.RESULTS_DIR / f"conversation_{i+1}_analysis.png"
        pipeline.visualize_conversation(result, save_path=viz_path)
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {Config.RESULTS_DIR}")
    print(f"Models saved to: {Config.MODEL_SAVE_DIR}")
    print("\nKey Files:")
    print(f"  - Best Model: {best_model_path}")
    print(f"  - Markov Chain: {markov_path}")
    print(f"  - Confusion Matrix: {Config.RESULTS_DIR / 'confusion_matrix.png'}")
    print(f"  - Training History: {Config.RESULTS_DIR / 'training_history.png'}")
    print(f"  - Performance Metrics: {Config.RESULTS_DIR / 'per_class_performance.png'}")
    print("\n" + "="*80 + "\n")


def create_demo_dataset():
    """Create a small demo dataset for testing"""
    print("Creating demo dataset with 100 synthetic conversations...\n")
    
    demo_conversations = []
    
    # Template conversations with different patterns
    templates = [
        # Pattern 1: Smooth resolution
        [
            {"speaker": "Customer", "text": "Hello, I need help with my account"},
            {"speaker": "Agent", "text": "Hello! I'd be happy to help. Can you provide your account number?"},
            {"speaker": "Customer", "text": "Sure, it's 12345"},
            {"speaker": "Agent", "text": "Thank you. I see your account. What seems to be the issue?"},
            {"speaker": "Customer", "text": "I was charged twice for my last order"},
            {"speaker": "Agent", "text": "I apologize for that. I'll process a refund right away"},
            {"speaker": "Customer", "text": "Thank you so much!"},
        ],
        
        # Pattern 2: Escalation
        [
            {"speaker": "Customer", "text": "My internet has been out for three days!"},
            {"speaker": "Agent", "text": "I'm sorry to hear that. Can you verify your address?"},
            {"speaker": "Customer", "text": "123 Main Street"},
            {"speaker": "Agent", "text": "I see an outage in your area. We're working on it"},
            {"speaker": "Customer", "text": "That's what you said yesterday! This is ridiculous!"},
            {"speaker": "Agent", "text": "I understand your frustration. Let me see what I can do"},
            {"speaker": "Customer", "text": "I want to speak to your manager right now!"},
        ],
        
        # Pattern 3: Multiple friction points
        [
            {"speaker": "Customer", "text": "I need to cancel my subscription"},
            {"speaker": "Agent", "text": "Can I ask why you're canceling?"},
            {"speaker": "Customer", "text": "It's too expensive and I'm not using it"},
            {"speaker": "Agent", "text": "I can offer you a discount if you stay"},
            {"speaker": "Customer", "text": "No thanks, just cancel it"},
            {"speaker": "Agent", "text": "Are you sure? Many customers find value in our service"},
            {"speaker": "Customer", "text": "I've been waiting 20 minutes. Just cancel it!"},
            {"speaker": "Agent", "text": "I'll process your cancellation now"},
        ],
    ]
    
    # Generate 100 conversations by varying the templates
    for i in range(100):
        template = templates[i % len(templates)]
        demo_conversations.append(template.copy())
    
    print(f"✓ Created {len(demo_conversations)} demo conversations\n")
    
    return demo_conversations


def run_inference_only(model_path: str, markov_path: str, conversation_file: str):
    """
    Run inference on new conversations without training
    
    Args:
        model_path: Path to trained model checkpoint
        markov_path: Path to Markov chain file
        conversation_file: Path to JSON file with new conversations
    """
    print("\n" + "="*80)
    print("INFERENCE MODE - ANALYZING NEW CONVERSATIONS")
    print("="*80 + "\n")
    
    # Load conversations
    data = load_dataset(conversation_file)
    conversations = extract_conversations(data)
    
    # Create pipeline
    pipeline = create_analyzer_pipeline(model_path, markov_path)
    
    # Analyze all conversations
    results = pipeline.analyze_batch(conversations)
    
    # Print summary
    escalated_count = sum(1 for r in results if r['root_cause_analysis']['escalated'])
    
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total Conversations: {len(results)}")
    print(f"Escalated: {escalated_count} ({escalated_count/len(results)*100:.1f}%)")
    print(f"Normal: {len(results) - escalated_count}")
    print(f"{'='*80}\n")
    
    # Detailed analysis for escalated conversations
    if escalated_count > 0:
        print("Analyzing escalated conversations in detail...\n")
        
        for i, result in enumerate(results):
            if result['root_cause_analysis']['escalated']:
                print(f"\n{'='*80}")
                print(f"ESCALATED CONVERSATION {i+1}")
                print(f"{'='*80}")
                pipeline.print_detailed_analysis(result)
                
                # Save visualization
                viz_path = Config.RESULTS_DIR / f"escalation_{i+1}_analysis.png"
                pipeline.visualize_conversation(result, save_path=viz_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Causal Analysis of Conversational Defects"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "inference"],
        help="Run mode: 'train' for complete pipeline, 'inference' for analysis only"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (for inference mode)"
    )
    
    parser.add_argument(
        "--markov",
        type=str,
        default=None,
        help="Path to Markov chain file (for inference mode)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to conversation data file (for inference mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        main()
    elif args.mode == "inference":
        if not all([args.model, args.markov, args.data]):
            print("ERROR: Inference mode requires --model, --markov, and --data arguments")
        else:
            run_inference_only(args.model, args.markov, args.data)
