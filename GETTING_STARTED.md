# 🚀 Getting Started Guide

## Quick Start in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**This installs:**
- PyTorch (GPU support)
- Sentence-Transformers (BERT)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- tqdm

### Step 2: Run Quick Tests

```bash
python quick_test.py --test all
```

**This verifies:**
- ✓ Silver labeling works
- ✓ BERT embedding works
- ✓ Neural model works
- ✓ Markov chain works
- ✓ Data loading works

### Step 3: Run the Complete Pipeline

```bash
python main.py
```

**This will:**
1. Create demo data (if your dataset is missing)
2. Label conversations
3. Embed with BERT
4. Train the model (50 epochs)
5. Evaluate on test set
6. Analyze sample conversations

**Expected Runtime:** 30-60 minutes on RTX 4060

---

## What You Get

After running `main.py`, you'll have:

### 📁 Models Directory (`models/`)

- `best_model.pth` - Your trained model
- `markov_chain.npz` - Transition matrix
- `embeddings.npy` - Cached BERT embeddings
- `checkpoint_epoch_*.pth` - Training checkpoints

### 📊 Results Directory (`results/`)

- `confusion_matrix.png` - Model performance visualization
- `training_history.png` - Loss & accuracy curves
- `per_class_performance.png` - Per-state metrics
- `conversation_*_analysis.png` - Sample root cause visualizations

---

## Your Dataset Format

Place your data in `Conversational_Transcript_Dataset.json`:

```json
[
  {
    "transcript_id": "12345",
    "conversation": [
      {"speaker": "Customer", "text": "My internet is down"},
      {"speaker": "Agent", "text": "I'll help you with that"},
      {"speaker": "Customer", "text": "It's been 3 days!"},
      {"speaker": "Agent", "text": "Let me check your account"},
      {"speaker": "Customer", "text": "This is ridiculous!"}
    ]
  }
]
```

---

## Using the Trained Model

### Analyze New Conversations

```python
from inference import create_analyzer_pipeline

# Load your trained model
pipeline = create_analyzer_pipeline(
    model_path="models/best_model.pth",
    markov_path="models/markov_chain.npz"
)

# Analyze a conversation
conversation = [
    {"speaker": "Customer", "text": "My order is late"},
    {"speaker": "Agent", "text": "Let me check that"},
    # ... more turns
]

result = pipeline.analyze_conversation(conversation)
pipeline.print_detailed_analysis(result)
```

### Batch Analysis

```bash
python main.py --mode inference \
  --model models/best_model.pth \
  --markov models/markov_chain.npz \
  --data new_conversations.json
```

---

## Understanding the Output

### State Predictions

Your conversation is mapped to states:

| State | Meaning |
|-------|---------|
| **S0** | Greeting/Neutral - "Hello", "Thanks" |
| **S1** | Info Exchange - "Account number?", "Email?" |
| **S2** | Problem - "Broken", "Not working" |
| **S3** | Solution - "Refund", "I'll fix it" |
| **S4** | Friction - "Waste of time", "Frustrated" |
| **S5** | Escalation - "Manager!", "Lawyer!" |

### Root Cause

The system identifies the **specific turn** where risk jumped:

```
Turn 4: [Customer] - State: S4 (Friction/Pushback) ⚠️ ROOT CAUSE
  "I've been waiting for 3 days and nobody is helping!"
  
Risk Jump: 0.743
Risk Before: 0.123
Risk After: 0.866
```

This turn caused the conversation to move toward escalation.

---

## Customization

### Adjust Keywords for States

Edit `config.py`:

```python
STATE_KEYWORDS = {
    0: ["hello", "thanks", "okay"],
    1: ["account", "verify", "email"],
    2: ["broken", "issue", "problem"],
    3: ["refund", "fix", "solution"],
    4: ["frustrated", "waiting", "ridiculous"],
    5: ["manager", "sue", "cancel"]
}
```

### Change Training Parameters

Edit `config.py`:

```python
BATCH_SIZE = 32  # Reduce if GPU out of memory
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
USE_FOCAL_LOSS = True  # For class imbalance
```

### Hardware Settings

**RTX 4060 (8GB):**
```python
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 4
USE_MIXED_PRECISION = True
```

**Smaller GPU (4GB):**
```python
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 8
USE_MIXED_PRECISION = True
```

**CPU Only:**
```python
DEVICE = torch.device("cpu")
BATCH_SIZE = 8
USE_MIXED_PRECISION = False
```

---

## Troubleshooting

### "CUDA out of memory"

**Fix:** Reduce batch size in `config.py`

```python
BATCH_SIZE = 16  # Was 32
```

### "DataLoader freezes" (Windows)

**Fix:** Already set correctly

```python
NUM_WORKERS = 0  # Must be 0 on Windows
```

### Poor performance on S4/S5

**Fix:** Increase focal loss

```python
FOCAL_LOSS_GAMMA = 3.0  # Was 2.0
```

### Model predicts only S0 (Neutral Trap)

**Symptoms:**
```
⚠️ WARNING: Possible 'Neutral Trap' detected!
   S0 recall: 98.5%
   S4 recall: 12.3%
   S5 recall: 8.7%
```

**Fix:**
1. Increase focal loss gamma
2. Add more keywords for S4 and S5
3. Check if your data has enough S4/S5 examples

---

## Next Steps

1. **Test on Your Data:** Replace demo data with real conversations
2. **Tune Hyperparameters:** Adjust `config.py` for your use case
3. **Refine Keywords:** Add domain-specific keywords to `STATE_KEYWORDS`
4. **Deploy Model:** Use `inference.py` for production analysis
5. **Scale Up:** Train on larger datasets for better performance

---

## Interactive Demo

Try the interactive demo to test labeling:

```bash
python quick_test.py --test demo
```

This lets you type conversations turn-by-turn and see state predictions in real-time.

---

## Need Help?

- Check `README.md` for detailed documentation
- Run `python quick_test.py --test all` to verify setup
- Review training plots in `results/` directory

---

**Happy Analyzing! 🎯**
