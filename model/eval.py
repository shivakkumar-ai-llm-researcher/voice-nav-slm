import json
import torch
from pathlib import Path
from model import VoiceNavSLM

MODEL_DIR = Path(__file__).parent
EXPORT_DIR = MODEL_DIR.parent / "export"
VOCAB_PATH = MODEL_DIR / "tokenizer" / "vocab.json"
LABELS_PATH = EXPORT_DIR / "labels.json"
MODEL_PATH = MODEL_DIR / "voice_nav_slm.pt"

def load_inference_engine():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_meta = json.load(f)

    word2idx = meta["word2idx"]
    intent2idx = labels_meta["intent2idx"]
    idx2intent = {int(k): v for k, v in labels_meta["idx2intent"].items()}
    tag2idx = labels_meta["tag2idx"]
    idx2tag = {int(k): v for k, v in labels_meta["idx2tag"].items()}

    model = VoiceNavSLM(
        vocab_size=len(word2idx),
        num_intents=len(intent2idx),
        num_slots=len(tag2idx),
        d_model=128,
        num_layers=4,
        num_heads=4,
        d_ff=512,
        max_seq_len=64
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    return model, word2idx, idx2intent, idx2tag

def predict(text, model, word2idx, idx2intent, idx2tag):
    tokens = text.lower().split()
    input_tokens = ["<CLS>"] + tokens + ["<SEP>"]
    input_ids = [word2idx.get(w, word2idx["<UNK>"]) for w in input_tokens]

    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    mask = torch.ones_like(input_tensor)

    with torch.no_grad():
        intent_logits, slot_logits = model(input_tensor, mask)

    intent_id = intent_logits.argmax(dim=-1).item()
    predicted_intent = idx2intent[intent_id]

    slot_preds = slot_logits[0].argmax(dim=-1).tolist()

    # Align slots (excluding <CLS> and <SEP>)
    token_slots = []
    for token, tag_id in zip(tokens, slot_preds[1:-1]):
        tag = idx2tag[tag_id]
        token_slots.append((token, tag))

    return predicted_intent, token_slots

def main():
    model, word2idx, idx2intent, idx2tag = load_inference_engine()
    print("VoiceNavSLM Model and Tokenizer Loaded Successfully!")

    test_queries = [
        "take me to payroll",
        "search for John Doe in employees",
        "turn on dark mode",
        "show approved leave requests",
        "create new lead",
        "go back to previous screen",
        "scroll down"
    ]

    print("\n--- Model Predictions ---")
    for q in test_queries:
        intent, slots = predict(q, model, word2idx, idx2intent, idx2tag)
        print(f"\nUtterance: \"{q}\"")
        print(f"  Intent: {intent}")
        print(f"  Slots:  {slots}")

if __name__ == "__main__":
    main()
