import sys
import json
import torch
from pathlib import Path

ROOT_DIR = Path("d:/tiny/voice-nav-slm")
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "model"))

from model import VoiceNavSLM

with open(ROOT_DIR / "export/labels.json") as f:
    labels = json.load(f)
with open(ROOT_DIR / "model/tokenizer/vocab.json") as f:
    vocab = json.load(f)

w2i = vocab["word2idx"]
i2idx = labels["intent2idx"]
idx2i = {int(k): v for k, v in labels["idx2intent"].items()}
t2idx = labels["tag2idx"]
idx2t = {int(k): v for k, v in labels["idx2tag"].items()}

model = VoiceNavSLM(len(w2i), len(i2idx), len(t2idx), 128, 4, 4, 512, 64)
model.load_state_dict(torch.load(ROOT_DIR / "model/voice_nav_slm.pt", map_location="cpu"))
model.eval()

samples = []
with open(ROOT_DIR / "data/generated/test_novel.jsonl") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

correct_intent = 0
true_pos = 0
false_pos = 0
false_neg = 0

with torch.no_grad():
    for item in samples:
        tokens = item["tokens"]
        input_ids = [2] + [w2i.get(w.lower(), 1) for w in tokens] + [3]
        if len(input_ids) > 64:
            input_ids = input_ids[:64]

        inp = torch.tensor([input_ids], dtype=torch.long)
        mask = torch.ones_like(inp)
        
        intent_logits, slot_logits = model(inp, mask)
        pred_intent = idx2i[intent_logits.argmax(-1).item()]
        if pred_intent == item["intent"]:
            correct_intent += 1
            
        slot_pred_ids = slot_logits[0].argmax(-1).tolist()
        pred_tags = [idx2t.get(idx, "O") for idx in slot_pred_ids[1:-1]]
        for true_t, pred_t in zip(item["bio_tags"], pred_tags):
            if true_t == "O" and pred_t == "O":
                continue
            if true_t == pred_t:
                true_pos += 1
            else:
                if pred_t != "O":
                    false_pos += 1
                if true_t != "O":
                    false_neg += 1

intent_acc = (correct_intent / len(samples)) * 100
prec = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
rec = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

out = f"Samples: {len(samples)} | Intent Acc: {intent_acc:.2f}% | Slot Precision: {prec*100:.2f}% | Slot Recall: {rec*100:.2f}% | Slot F1: {f1*100:.2f}%"
with open(ROOT_DIR / "eval/result.txt", "w", encoding="utf-8") as f:
    f.write(out + "\n")

print(out)
