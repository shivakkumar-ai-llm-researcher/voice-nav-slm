import os
import json
import torch
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_DIR / "model"
EXPORT_DIR = PROJECT_DIR / "export"

DEFAULT_CONFIDENCE_THRESHOLD = 0.85

class VoiceNavDesktopEngine:
    def __init__(self, labels_path=None, vocab_path=None, model_path=None, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        if labels_path is None:
            labels_path = EXPORT_DIR / "labels.json"
        if vocab_path is None:
            vocab_path = MODEL_DIR / "tokenizer" / "vocab.json"
        if model_path is None:
            model_path = EXPORT_DIR / "voice_nav_slm_traced.pt"

        self.confidence_threshold = confidence_threshold

        with open(labels_path, "r", encoding="utf-8") as f:
            labels_meta = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_meta = json.load(f)

        self.word2idx = vocab_meta["word2idx"]
        self.idx2intent = {int(k): v for k, v in labels_meta["idx2intent"].items()}
        self.idx2tag = {int(k): v for k, v in labels_meta["idx2tag"].items()}
        self.unk_idx = self.word2idx.get("<UNK>", 1)
        self.cls_idx = self.word2idx.get("<CLS>", 2)
        self.sep_idx = self.word2idx.get("<SEP>", 3)

        print(f"Loading Desktop TorchScript model from {model_path}...")
        self.model = torch.jit.load(str(model_path))
        self.model.eval()

    def predict(self, text: str):
        raw_tokens = text.lower().strip().split()
        tokens = ["<CLS>"] + raw_tokens + ["<SEP>"]

        input_ids = [self.word2idx.get(w, self.unk_idx) for w in tokens]
        if len(input_ids) > 64:
            input_ids = input_ids[:64]
        mask = [1] * len(input_ids)

        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        mask_tensor = torch.tensor([mask], dtype=torch.long)

        with torch.no_grad():
            intent_probs, slot_probs = self.model(input_tensor, mask_tensor)

        intent_id = intent_probs.argmax(dim=-1).item()
        confidence = intent_probs[0, intent_id].item()
        intent = self.idx2intent[intent_id]

        # Confidence Gating / Fallback Trigger
        requires_fallback = bool(confidence < self.confidence_threshold)

        slot_tag_ids = slot_probs[0].argmax(dim=-1).tolist()
        slots = []

        for token, tag_id in zip(raw_tokens, slot_tag_ids[1:-1]):
            slots.append({"token": token, "tag": self.idx2tag.get(tag_id, "O")})

        return {
            "query": text,
            "intent": intent if not requires_fallback else "FALLBACK_CLARIFICATION",
            "predicted_intent": intent,
            "confidence": round(confidence, 4),
            "confidence_threshold": self.confidence_threshold,
            "requires_fallback": requires_fallback,
            "slots": slots
        }

if __name__ == "__main__":
    engine = VoiceNavDesktopEngine()
    
    test_queries = [
        "take me to payroll",
        "search for John Doe in employees",
        "turn on dark mode",
        "create new lead",
        "go back to previous screen"
    ]

    print("\n--- Desktop Inference Engine Output ---")
    for q in test_queries:
        res = engine.predict(q)
        print(f"\nQuery: \"{q}\"")
        print(f"  Intent:    {res['intent']} (Conf: {res['confidence']})")
        print(f"  Fallback:  {res['requires_fallback']}")
        print(f"  Slots:     {res['slots']}")
