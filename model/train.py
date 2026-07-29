import os
import sys
import json
import traceback
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import VoiceNavSLM

DATA_DIR = Path(__file__).parent.parent / "data" / "generated"
LABELS_PATH = Path(__file__).parent.parent / "export" / "labels.json"
TOKENIZER_DIR = Path(__file__).parent / "tokenizer"
MODEL_SAVE_PATH = Path(__file__).parent / "voice_nav_slm.pt"

class VoiceNavDataset(Dataset):
    def __init__(self, data, word2idx, intent2idx, tag2idx, max_seq_len=64):
        self.samples = []
        self.max_seq_len = max_seq_len

        cls_id = word2idx.get("<CLS>", 2)
        sep_id = word2idx.get("<SEP>", 3)
        unk_id = word2idx.get("<UNK>", 1)
        pad_id = word2idx.get("<PAD>", 0)

        for item in data:
            tokens = item["tokens"]
            intent = item["intent"]
            bio_tags = item["bio_tags"]

            input_ids = [cls_id] + [
                word2idx.get(w.lower(), unk_id) for w in tokens
            ] + [sep_id]

            slot_tags = [-100] + [
                tag2idx.get(t, 0) for t in bio_tags
            ] + [-100]

            seq_len = len(input_ids)
            if seq_len > max_seq_len:
                input_ids = input_ids[:max_seq_len]
                slot_tags = slot_tags[:max_seq_len]
                attention_mask = [1] * max_seq_len
            else:
                pad_len = max_seq_len - seq_len
                attention_mask = [1] * seq_len + [0] * pad_len
                input_ids = input_ids + [pad_id] * pad_len
                slot_tags = slot_tags + [-100] * pad_len

            intent_id = intent2idx[intent]

            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "intent_id": torch.tensor(intent_id, dtype=torch.long),
                "slot_tags": torch.tensor(slot_tags, dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def evaluate(model, dataloader, device):
    model.eval()
    correct_intent = 0
    total_intent = 0
    correct_slots = 0
    total_slots = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_targets = batch["intent_id"].to(device)
            slot_targets = batch["slot_tags"].to(device)

            intent_logits, slot_logits = model(input_ids, attention_mask)

            intent_preds = intent_logits.argmax(dim=-1)
            correct_intent += (intent_preds == intent_targets).sum().item()
            total_intent += intent_targets.size(0)

            slot_preds = slot_logits.argmax(dim=-1)
            mask = (slot_targets != -100)
            correct_slots += ((slot_preds == slot_targets) & mask).sum().item()
            total_slots += mask.sum().item()

    intent_acc = (correct_intent / total_intent) * 100 if total_intent > 0 else 0
    slot_acc = (correct_slots / total_slots) * 100 if total_slots > 0 else 0
    return intent_acc, slot_acc

def train():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sys.stdout.write(f"Using device: {device}\n")
        sys.stdout.flush()

        # Load Single-Source-of-Truth Labels
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels_meta = json.load(f)

        intent2idx = labels_meta["intent2idx"]
        tag2idx = labels_meta["tag2idx"]

        # Load Tokenizer Vocab
        with open(TOKENIZER_DIR / "vocab.json", "r", encoding="utf-8") as f:
            vocab_meta = json.load(f)
        word2idx = vocab_meta["word2idx"]

        train_data = load_jsonl(DATA_DIR / "train.jsonl")
        val_data = load_jsonl(DATA_DIR / "val.jsonl")
        test_novel_data = load_jsonl(DATA_DIR / "test_novel.jsonl")

        sys.stdout.write(f"Vocab size: {len(word2idx)} | Intents: {len(intent2idx)} | Slot Tags: {len(tag2idx)}\n")
        sys.stdout.flush()

        train_dataset = VoiceNavDataset(train_data, word2idx, intent2idx, tag2idx)
        val_dataset = VoiceNavDataset(val_data, word2idx, intent2idx, tag2idx)
        test_dataset = VoiceNavDataset(test_novel_data, word2idx, intent2idx, tag2idx)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = VoiceNavSLM(
            vocab_size=len(word2idx),
            num_intents=len(intent2idx),
            num_slots=len(tag2idx),
            d_model=128,
            num_layers=4,
            num_heads=4,
            d_ff=512,
            max_seq_len=64,
            dropout=0.1
        ).to(device)

        intent_criterion = nn.CrossEntropyLoss()
        slot_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        epochs = 6
        sys.stdout.write("Starting training loop...\n")
        sys.stdout.flush()

        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                intent_targets = batch["intent_id"].to(device)
                slot_targets = batch["slot_tags"].to(device)

                optimizer.zero_grad()
                intent_logits, slot_logits = model(input_ids, attention_mask)

                l_intent = intent_criterion(intent_logits, intent_targets)
                l_slot = slot_criterion(slot_logits.view(-1, len(tag2idx)), slot_targets.view(-1))

                loss = l_intent + 1.5 * l_slot
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            val_intent_acc, val_slot_acc = evaluate(model, val_loader, device)

            msg = f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val Intent Acc: {val_intent_acc:.2f}% | Val Slot Acc: {val_slot_acc:.2f}%\n"
            sys.stdout.write(msg)
            sys.stdout.flush()

            if val_intent_acc + val_slot_acc > best_val_acc:
                best_val_acc = val_intent_acc + val_slot_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

        sys.stdout.write(f"\nModel saved to {MODEL_SAVE_PATH}\n")
        sys.stdout.flush()

        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        test_intent_acc, test_slot_acc = evaluate(model, test_loader, device)
        sys.stdout.write(f"\nFinal Performance on Held-Out test_novel.jsonl:\n")
        sys.stdout.write(f"  Test Intent Accuracy: {test_intent_acc:.2f}%\n")
        sys.stdout.write(f"  Test Slot Token Accuracy: {test_slot_acc:.2f}%\n")
        sys.stdout.flush()

    except Exception:
        sys.stdout.write("EXCEPTION OCCURRED:\n")
        sys.stdout.write(traceback.format_exc() + "\n")
        sys.stdout.flush()
        sys.exit(1)

if __name__ == "__main__":
    train()
