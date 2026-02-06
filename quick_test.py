"""
Quick Test Script
Rapid testing of the system with minimal setup
"""

import sys
from utils import create_sample_conversation, initialize_environment
from config import Config

def quick_test_silver_labeling():
    """Test silver labeling on a sample conversation"""
    print("\n" + "="*80)
    print("QUICK TEST: Silver Labeling")
    print("="*80 + "\n")
    
    from silver_labeling import SilverLabeler
    
    # Create sample conversation
    conversation = create_sample_conversation()
    
    # Label it
    labeler = SilverLabeler()
    states, confidences = labeler.label_conversation(conversation)
    
    # Print results
    print("Sample Conversation Analysis:")
    print("-" * 80)
    
    for i, turn in enumerate(conversation):
        state_id = states[i]
        state_name = Config.STATE_NAMES[state_id]
        confidence = confidences[i]
        
        print(f"\nTurn {i}:")
        print(f"  Speaker: {turn['speaker']}")
        print(f'  Text: "{turn["text"]}"')
        print(f"  State: S{state_id} - {state_name}")
        print(f"  Confidence: {confidence:.2f}")
    
    print("\n" + "="*80 + "\n")


def quick_test_embedding():
    """Test BERT embedding on a sample"""
    print("\n" + "="*80)
    print("QUICK TEST: BERT Embedding")
    print("="*80 + "\n")
    
    from embeddings import ConversationEmbedder
    
    # Create embedder
    embedder = ConversationEmbedder()
    
    # Test single text
    sample_text = "My internet has been down for three days!"
    embedding = embedder.embed_text(sample_text)
    
    print(f"Sample Text: \"{sample_text}\"")
    print(f"Embedding Shape: {embedding.shape}")
    print(f"Embedding Dimension: {len(embedding)}")
    print(f"Sample Values: {embedding[:5]}")
    
    print("\n" + "="*80 + "\n")


def quick_test_model_creation():
    """Test model creation"""
    print("\n" + "="*80)
    print("QUICK TEST: Model Creation")
    print("="*80 + "\n")
    
    from model import create_model
    
    # Create model
    model = create_model()
    
    # Test forward pass
    import torch
    batch_size = 2
    seq_len = 5
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_len, Config.BERT_DIM).to(Config.DEVICE)
    dummy_lengths = torch.tensor([5, 3]).to(Config.DEVICE)
    
    # Forward pass
    with torch.no_grad():
        logits, hidden = model(dummy_input, dummy_lengths)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {logits.shape}")
    print(f"Expected: [batch_size={batch_size}, seq_len={seq_len}, num_states={Config.NUM_STATES}]")
    
    print("\n✓ Model forward pass successful!")
    print("="*80 + "\n")


def quick_test_markov_chain():
    """Test Markov chain building"""
    print("\n" + "="*80)
    print("QUICK TEST: Markov Chain")
    print("="*80 + "\n")
    
    from markov_chain import MarkovChain
    import numpy as np
    
    # Create sample state sequences
    sample_sequences = [
        [0, 1, 2, 3, 0],
        [0, 2, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 0],
        [0, 2, 3, 0],
    ]
    
    # Build Markov chain
    markov = MarkovChain()
    markov.fit(sample_sequences)
    
    # Test queries
    print("\nTest Queries:")
    print(f"  P(S1 → S2) = {markov.get_transition_probability(1, 2):.3f}")
    print(f"  P(S2 → S3) = {markov.get_transition_probability(2, 3):.3f}")
    print(f"  P(S4 → S5) = {markov.get_transition_probability(4, 5):.3f}")
    
    print("\n✓ Markov chain functional!")
    print("="*80 + "\n")


def quick_test_data_loader():
    """Test data loading pipeline"""
    print("\n" + "="*80)
    print("QUICK TEST: Data Loading Pipeline")
    print("="*80 + "\n")
    
    from data_loader import ConversationDataset, collate_fn
    from torch.utils.data import DataLoader
    import numpy as np
    
    # Create dummy data
    conversations = [create_sample_conversation() for _ in range(5)]
    states = [[0, 1, 2, 3, 0, 1, 2, 3, 4] for _ in range(5)]
    embeddings = [np.random.randn(9, Config.BERT_DIM) for _ in range(5)]
    
    # Create dataset
    dataset = ConversationDataset(conversations, states, embeddings)
    
    # Create loader
    loader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Test batch
    for batch_emb, batch_labels, batch_lengths in loader:
        print(f"Batch Embeddings Shape: {batch_emb.shape}")
        print(f"Batch Labels Shape: {batch_labels.shape}")
        print(f"Batch Lengths: {batch_lengths}")
        break
    
    print("\n✓ Data loader functional!")
    print("="*80 + "\n")


def run_all_quick_tests():
    """Run all quick tests"""
    print("\n" + "="*80)
    print("RUNNING ALL QUICK TESTS")
    print("="*80 + "\n")
    
    # Initialize
    initialize_environment()
    
    tests = [
        ("Silver Labeling", quick_test_silver_labeling),
        ("BERT Embedding", quick_test_embedding),
        ("Model Creation", quick_test_model_creation),
        ("Markov Chain", quick_test_markov_chain),
        ("Data Loader", quick_test_data_loader),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ Test '{test_name}' FAILED:")
            print(f"   Error: {str(e)}\n")
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("QUICK TEST SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED! System is ready to use.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
    
    print("="*80 + "\n")


def interactive_demo():
    """Interactive demo for testing custom conversations"""
    print("\n" + "="*80)
    print("INTERACTIVE DEMO")
    print("="*80 + "\n")
    
    from silver_labeling import SilverLabeler
    
    labeler = SilverLabeler()
    
    print("Enter a conversation turn-by-turn.")
    print("Format: <Speaker>: <Text>")
    print("Example: Customer: My internet is down")
    print("Type 'done' when finished, 'quit' to exit\n")
    
    conversation = []
    
    while True:
        turn_input = input(f"Turn {len(conversation) + 1}: ").strip()
        
        if turn_input.lower() == 'quit':
            print("Exiting...")
            return
        
        if turn_input.lower() == 'done':
            if len(conversation) == 0:
                print("No conversation entered. Try again.\n")
                continue
            break
        
        # Parse input
        if ':' not in turn_input:
            print("Invalid format. Use: Speaker: Text\n")
            continue
        
        speaker, text = turn_input.split(':', 1)
        speaker = speaker.strip()
        text = text.strip()
        
        if speaker not in ['Customer', 'Agent']:
            print("Speaker must be 'Customer' or 'Agent'\n")
            continue
        
        conversation.append({"speaker": speaker, "text": text})
        print(f"  Added: [{speaker}] {text}\n")
    
    # Analyze conversation
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80 + "\n")
    
    states, confidences = labeler.label_conversation(conversation)
    
    for i, turn in enumerate(conversation):
        state_id = states[i]
        state_name = Config.STATE_NAMES[state_id]
        confidence = confidences[i]
        
        print(f"Turn {i}: [{turn['speaker']}]")
        print(f'  "{turn["text"]}"')
        print(f"  → State: S{state_id} ({state_name}), Confidence: {confidence:.2f}\n")
    
    # Check for escalation
    if 5 in states:
        esc_turn = states.index(5)
        print("⚠️  ESCALATION DETECTED!")
        print(f"   First escalation at turn {esc_turn}\n")
    else:
        print("✓ No escalation detected\n")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Testing Suite")
    parser.add_argument(
        "--test",
        type=str,
        choices=['all', 'labeling', 'embedding', 'model', 'markov', 'loader', 'demo'],
        default='all',
        help="Which test to run"
    )
    
    args = parser.parse_args()
    
    if args.test == 'all':
        run_all_quick_tests()
    elif args.test == 'labeling':
        quick_test_silver_labeling()
    elif args.test == 'embedding':
        quick_test_embedding()
    elif args.test == 'model':
        quick_test_model_creation()
    elif args.test == 'markov':
        quick_test_markov_chain()
    elif args.test == 'loader':
        quick_test_data_loader()
    elif args.test == 'demo':
        interactive_demo()
