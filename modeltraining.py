import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# --- HARDWARE ACCELERATION SETUP ---
def get_device():
    if torch.cuda.is_available():
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("✅ APPLE SILICON GPU DETECTED (MPS)")
        return torch.device("mps")
    else:
        print("⚠️ NO GPU DETECTED. Running on CPU (All Cores).")
        return torch.device("cpu")

DEVICE = get_device()

# --- CONFIGURATION ---
DATA_FILE = 'Conversational_Transcript_Dataset.json'
MODEL_FILE = 'gru_causal_model.pth'
MATRIX_FILE = 'transition_matrix.npy'

# --- PART 1: STATE DEFINITIONS ---
STATE_MAP = {
    0: "Greeting/Neutral",
    1: "Info_Exchange",
    2: "Problem_Statement",
    3: "Solution_Offer",
    4: "Friction/Pushback",
    5: "Escalation/Anger"
}

def heuristic_labeler(text, speaker):
    text = text.lower()
    if any(w in text for w in ['manager', 'supervisor', 'lawyer', 'legal', 'complaint', 'threat', 'sue']): return 5
    if 'speak' in text and any(w in text for w in ['to', 'with', 'manager', 'someone']): return 5 
    if any(w in text for w in ['nothing', 'weeks', 'long', 'ridiculous', 'waste', 'patience', 'stop', 'fail']): return 4
    if 'dont' in text or "don't" in text or 'wait' in text: return 4
    if any(w in text for w in ['refund', 'replacement', 'credit', 'scheduled', 'fixed', 'shipped', 'reverse']): return 3
    if speaker == "Agent" and any(w in text for w in ['new', 'receive', 'days', 'today', 'send', 'ship', 'ill']): return 3
    if any(w in text for w in ['broken', 'outage', 'fraud', 'charge', 'delay', 'stuck', 'never', 'crack']): return 2
    if any(w in text for w in ['work', 'working', 'fix', 'issue', 'system']):
        if 'not' in text or 'cant' in text: return 2
    if any(w in text for w in ['number', 'email', 'address', 'verify', 'date', 'code', 'card', 'account']): return 1
    return 0

# --- PART 2: GPU-ACCELERATED PROCESSING ---
def process_data():
    print(f"Loading {DATA_FILE}...")
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: Data file not found. Please ensure JSON is in the folder.")
        return [], []

    all_turns = []
    all_sequences = [] 
    
    print("Initializing BERT on GPU...")
    # This automatically moves the Transformer to GPU if DEVICE is cuda/mps
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    
    print(f"Vectorizing {len(data['transcripts'])} transcripts (Parallelized)...")
    
    for transcript in data['transcripts']:
        t_id = transcript['transcript_id']
        sequence = []
        texts = [t['text'] for t in transcript['conversation']]
        
        # GPU Batch Processing happens here
        embeddings = embedder.encode(texts, convert_to_tensor=True, device=DEVICE)
        
        # Move back to CPU only for storage/list manipulation to save VRAM
        embeddings_cpu = embeddings.cpu().numpy()
        
        for i, turn in enumerate(transcript['conversation']):
            label = heuristic_labeler(turn['text'], turn['speaker'])
            sequence.append(label)
            
            all_turns.append({
                'transcript_id': t_id,
                'vector': embeddings_cpu[i], # Store as numpy
                'label': label,
                'text': turn['text']
            })
        all_sequences.append(sequence)
            
    return all_turns, all_sequences

# --- PART 3: BUILD MATRIX ---
def build_transition_matrix(sequences):
    print("Building Probability Matrix...")
    num_states = 6
    matrix = np.zeros((num_states, num_states))
    
    for seq in sequences:
        for i in range(len(seq) - 1):
            curr, next_s = seq[i], seq[i+1]
            matrix[curr][next_s] += 1
            
    row_sums = matrix.sum(axis=1, keepdims=True)
    prob_matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums!=0)
    
    np.save(MATRIX_FILE, prob_matrix)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(prob_matrix, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=STATE_MAP.values(), yticklabels=STATE_MAP.values())
    plt.title('Causal Transition Probabilities')
    plt.show() # Can comment this out if running on a headless server
    return prob_matrix

# --- PART 4: GPU-ACCELERATED GRU TRAINING ---
class ConversationGRU(nn.Module):
    def __init__(self):
        super(ConversationGRU, self).__init__()
        self.gru = nn.GRU(384, 128, batch_first=True) 
        self.fc = nn.Linear(128, 6) 

    def forward(self, x):
        # x is already on DEVICE
        out, _ = self.gru(x)
        # We take the last hidden state for prediction in this simple trainer
        # OR we can return full sequence. For causal analysis, we usually want full sequence.
        # Here we just return linear mapping of all outputs
        return self.fc(out)

def train_model(data_dicts):
    print(f"Training GRU Model on {DEVICE}...")
    
    # Prepare Data
    X = np.array([d['vector'] for d in data_dicts])
    y = np.array([d['label'] for d in data_dicts])
    
    # Move entire dataset to GPU (VRAM permitting) for speed
    # We add a fake batch dimension (1, Total_Turns, 384) just for this hackathon demo
    # In production, use DataLoader with mini-batches
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    model = ConversationGRU().to(DEVICE) # Move model to GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(100): # Increased epochs since GPU is fast
        optimizer.zero_grad()
        
        output = model(X_tensor) # Forward pass on GPU
        
        # Flatten for loss calculation
        loss = criterion(output.view(-1, 6), y_tensor.view(-1))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0: 
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
        
    torch.save(model.state_dict(), MODEL_FILE)
    print("Model Trained & Saved.")
    return model

# --- PART 5: GPU INFERENCE ---
def find_root_cause(transcript_text_list, matrix, model):
    print("\n--- ANALYZING NEW TRANSCRIPT ---")
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    
    # 1. Vectorize on GPU
    # encode() returns a tensor on CPU by default unless specified, 
    # but we can manually move it.
    vectors = embedder.encode(transcript_text_list) 
    vec_tensor = torch.tensor(vectors, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # 2. Predict on GPU
    model.eval()
    with torch.no_grad():
        logits = model(vec_tensor)
        # Move result back to CPU for python list processing
        predicted_path = torch.argmax(logits, dim=2).squeeze().cpu().tolist()
        
    # 3. CPU Analysis (Math is trivial here, no GPU needed)
    max_risk_jump = 0
    cause_index = -1
    
    print(f"Predicted Path: {[STATE_MAP[s] for s in predicted_path]}")
    
    for i in range(len(predicted_path)-1):
        curr_s = predicted_path[i]
        next_s = predicted_path[i+1]
        
        risk_curr = matrix[curr_s][5]
        risk_next = matrix[next_s][5]
        
        jump = risk_next - risk_curr
        if next_s == 5: jump = 1.0 
        
        if jump > max_risk_jump:
            max_risk_jump = jump
            cause_index = i + 1
            
    if cause_index != -1:
        print(f"\nROOT CAUSE FOUND at Turn {cause_index+1}:")
        print(f"Trigger Phrase: \"{transcript_text_list[cause_index]}\"")
        print(f"Explanation: Probability of Escalation spiked by {max_risk_jump*100:.1f}%.")
    else:
        print("No specific causal trigger found.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Process
    processed_data, sequences = process_data()
    
    if processed_data:
        # 2. Map
        matrix = build_transition_matrix(sequences)
        
        # 3. Train
        model = train_model(processed_data)
        
        # 4. Test
        test_call = [
            "Hello, thank you for calling.",
            "My internet is broken.",
            "I can help. Let me check.",
            "It's still not working!",
            "I can't fix it right now.",
            "This is ridiculous, I want a manager!"
        ]
        find_root_cause(test_call, matrix, model)