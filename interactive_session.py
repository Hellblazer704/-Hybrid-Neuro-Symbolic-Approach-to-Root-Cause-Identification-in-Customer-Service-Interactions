import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. SETUP & CONFIGURATION ---
MODEL_FILE = os.path.join('models', 'best_model.pth')
MATRIX_FILE = os.path.join('models', 'markov_chain.npz')

# Check for Gemini Library
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("⚠️  'google-generativeai' not installed. Running in Mock Mode.")

# --- 2. CORRECTED MODEL ARCHITECTURE ---
# Matching your checkpoint: Bidirectional GRU (256 hidden) -> FC (256) -> FC (6)
class ConversationGRU(nn.Module):
    def __init__(self):
        super(ConversationGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=384, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3 
        )
        self.fc1 = nn.Linear(512, 256) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 6)   

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

STATE_MAP = {
    0: "Greeting/Neutral", 1: "Info_Exchange", 2: "Problem_Statement",
    3: "Solution_Offer", 4: "Friction/Pushback", 5: "Escalation/Anger"
}

# --- 3. THE ANALYZER ENGINE ---
class AnalysisPipeline:
    def __init__(self, model_path, matrix_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔍 Loading model from: {os.path.abspath(model_path)}")
        
        if not os.path.exists(matrix_path):
            print(f"❌ ERROR: Matrix file not found at '{matrix_path}'")
            sys.exit(1)
            
        try:
            data = np.load(matrix_path, allow_pickle=True)
            if 'transition_matrix' in data:
                self.matrix = data['transition_matrix']
            elif 'arr_0' in data:
                 self.matrix = data['arr_0']
            else:
                self.matrix = np.ones((6,6)) / 6
        except Exception as e:
            print(f"❌ Error loading matrix: {e}")
            sys.exit(1)

        if not os.path.exists(model_path):
            print(f"❌ ERROR: Model file not found at '{model_path}'")
            sys.exit(1)
            
        self.model = ConversationGRU().to(self.device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("✅ Model loaded successfully.")
        except RuntimeError as e:
            print(f"\n❌ ARCHITECTURE ERROR: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
            
        print("⏳ Loading BERT Embedder...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

    def analyze(self, conversation_data):
        texts = [t['text'] for t in conversation_data]
        vectors = self.embedder.encode(texts, convert_to_tensor=True, device=self.device)
        vectors = vectors.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(vectors)
            predicted_path = torch.argmax(logits, dim=2).squeeze().cpu().tolist()
            
        max_jump = 0
        cause_idx = -1
        
        for i in range(len(predicted_path)-1):
            curr, next_s = predicted_path[i], predicted_path[i+1]
            risk_curr = self.matrix[curr][5] if curr < 6 else 0
            risk_next = 1.0 if next_s == 5 else (self.matrix[next_s][5] if next_s < 6 else 0)
            
            jump = risk_next - risk_curr
            if jump > max_jump:
                max_jump = jump
                cause_idx = i + 1
        
        return {
            'conversation': conversation_data,
            'predicted_states': predicted_path,
            'root_cause': {
                'turn_index': cause_idx,
                'escalated': 5 in predicted_path,
                'risk_spike': max_jump
            }
        }

# --- 4. THE LLM BRAIN (AUTO-DISCOVERY) ---
class GeminiBrain:
    def __init__(self, api_key):
        self.mock = True
        self.model = None
        
        if api_key and HAS_GEMINI:
            try:
                genai.configure(api_key=api_key)
                
                # AUTO-DISCOVERY LOGIC
                # We ask the API: "What models do I have access to?"
                print("🔎 Scanning for available Gemini models...")
                available_models = []
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        available_models.append(m.name)
                
                # Priority list: Try to find 'flash', then 'pro', then anything 'gemini'
                chosen_model_name = None
                for m in available_models:
                    if 'flash' in m: 
                        chosen_model_name = m
                        break
                if not chosen_model_name:
                    for m in available_models:
                        if 'pro' in m:
                            chosen_model_name = m
                            break
                if not chosen_model_name and available_models:
                    chosen_model_name = available_models[0]
                
                if chosen_model_name:
                    print(f"✅ Connected to Gemini Model: {chosen_model_name}")
                    self.model = genai.GenerativeModel(chosen_model_name)
                    self.mock = False
                else:
                    print("⚠️ No suitable Gemini model found in your account. Using Mock Mode.")
                    
            except Exception as e:
                print(f"⚠️ Gemini Connection Failed: {e}. Using Mock Mode.")

    def ask(self, query, analysis, history):
        conv = analysis['conversation']
        rc = analysis['root_cause']
        states = analysis['predicted_states']
        
        transcript = ""
        for i, turn in enumerate(conv):
            lbl = STATE_MAP[states[i]]
            tag = " [TRIGGER]" if i == rc['turn_index'] else ""
            transcript += f"Turn {i} ({turn['speaker']}): {turn['text']} ({lbl}){tag}\n"
            
        prompt = f"""
        You are an expert Analyst. 
        CONTEXT:
        {transcript}
        ANALYSIS:
        - Escalated: {rc['escalated']}
        - Root Cause Turn: {rc['turn_index']}
        - Confidence: {rc['risk_spike']:.2f}
        HISTORY:
        {history}
        User Question: {query}
        Answer concisely using the context.
        """
        
        if self.mock:
            return f"[MOCK MODE] Root cause found at Turn {rc['turn_index']}. (Set GEMINI_API_KEY to see real AI response)"
            
        try:
            return self.model.generate_content(prompt).text
        except Exception as e:
            return f"API Error: {e}"

# --- 5. MAIN INTERACTIVE LOOP ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  TASK 2: INTERACTIVE REASONING SESSION")
    print("="*50)

    # Robust Key Loading
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠️  GEMINI_API_KEY not found in environment.")
        print("   (Enter it below to enable AI mode)")
        api_key = input("🔑 Please paste your Gemini API Key here: ").strip()

    # Initialize
    pipeline = AnalysisPipeline(MODEL_FILE, MATRIX_FILE)
    brain = GeminiBrain(api_key)
    
    # Load Test Call
    test_call = [
        {"speaker": "Customer", "text": "My internet is down again."},
        {"speaker": "Agent", "text": "I can help with that. Let me check."},
        {"speaker": "Customer", "text": "This happens every week."},
        {"speaker": "Agent", "text": "I can't fix it right now. You have to wait."},
        {"speaker": "Customer", "text": "This is ridiculous! I want a manager!"}
    ]
    
    print("\n⚙️  Analyzing Conversation...")
    analysis = pipeline.analyze(test_call)
    print("✅ Call Analyzed. Root Cause Identified.")
    
    history = []
    print("\n💬 Session Started. Type 'exit' to quit or 'report' to save.")
    
    while True:
        try:
            q = input("\nAnalyst: ").strip()
        except EOFError:
            break
            
        if q.lower() == 'exit': break
        
        if q.lower() == 'report':
            with open("Session_Report.txt", "w") as f:
                f.write(str(analysis))
            print("📄 Report saved to Session_Report.txt")
            continue
            
        print("Thinking...", end="\r")
        ans = brain.ask(q, analysis, history)
        
        print(" " * 20, end="\r") 
        print(f"AI: {ans}")
        
        history.append(f"Q: {q}\nA: {ans}")