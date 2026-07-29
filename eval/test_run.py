import sys
import json
import traceback
from pathlib import Path

ROOT_DIR = Path("D:/tiny/voice-nav-slm")
MODEL_DIR = ROOT_DIR / "model"
EXPORT_DIR = ROOT_DIR / "export"
DATA_DIR = ROOT_DIR / "data" / "generated"
EVAL_DIR = ROOT_DIR / "eval"

sys.path.append(str(ROOT_DIR))
sys.path.append(str(MODEL_DIR))

import torch
from model import VoiceNavSLM

log_file = EVAL_DIR / "eval_output.json"

try:
    labels_path = EXPORT_DIR / "labels.json"
    vocab_path = MODEL_DIR / "tokenizer" / "vocab.json"
    model_path = MODEL_DIR / "voice_nav_slm.pt"
    test_novel_path = DATA_DIR / "test_novel.jsonl"

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_meta = json.load(f)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_meta = json.load(f)

    word2idx = vocab_meta["word2idx"]
    intent2idx = labels_meta["intent2idx"]
    idx2intent = {int(k): v for k, v in labels_meta["idx2intent"].items()}
    tag2idx = labels_meta["tag2idx"]
    idx2tag = {int(k): v for k, v in labels_meta["idx2tag"].items()}

    cls_id = word2idx.get("<CLS>", 2)
    sep_id = word2idx.get("<SEP>", 3)
    unk_id = word2idx.get("<UNK>", 1)

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
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    samples = []
    with open(test_novel_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    correct_intent = 0
    total_intent = len(samples)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    confidence_scores = []

    with torch.no_grad():
        for item in samples:
            tokens = item["tokens"]
            true_intent = item["intent"]
            true_tags = item["bio_tags"]

            input_tokens = [cls_id] + [word2idx.get(w.lower(), unk_id) for w in tokens] + [sep_id]
            input_tensor = torch.tensor([input_tokens], dtype=torch.long)
            mask_tensor = torch.ones_like(input_tensor)

            intent_logits, slot_logits = model(input_tensor, mask_tensor)

            intent_probs = torch.softmax(intent_logits, dim=-1)
            pred_intent_id = intent_probs.argmax(dim=-1).item()
            conf = intent_probs[0, pred_intent_id].item()
            confidence_scores.append(conf)

            if idx2intent[pred_intent_id] == true_intent:
                correct_intent += 1

            slot_pred_ids = slot_logits[0].argmax(dim=-1).tolist()
            pred_tags = [idx2tag.get(idx, "O") for idx in slot_pred_ids[1:-1]]

            for true_t, pred_t in zip(true_tags, pred_tags):
                if true_t == "O" and pred_t == "O":
                    continue
                if true_t == pred_t:
                    true_positives += 1
                else:
                    if pred_t != "O":
                        false_positives += 1
                    if true_t != "O":
                        false_negatives += 1

    intent_acc = (correct_intent / total_intent) * 100.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_conf = sum(confidence_scores) / len(confidence_scores)

    results = {
        "total_test_samples": total_intent,
        "intent_accuracy_percent": round(intent_acc, 2),
        "slot_precision_percent": round(precision * 100, 2),
        "slot_recall_percent": round(recall * 100, 2),
        "slot_f1_percent": round(f1 * 100, 2),
        "average_confidence": round(avg_conf, 4),
        "passed_target_bar": bool(intent_acc >= 90.0)
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

except Exception as e:
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"EXCEPTION: {e}\n" + traceback.format_exc())
