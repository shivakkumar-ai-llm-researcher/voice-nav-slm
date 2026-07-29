import sys
import json
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
MODEL_DIR = ROOT_DIR / "model"
EXPORT_DIR = ROOT_DIR / "export"
DATA_DIR = ROOT_DIR / "data" / "generated"
EVAL_DIR = Path(__file__).parent

sys.path.append(str(ROOT_DIR))
sys.path.append(str(MODEL_DIR))

from model import VoiceNavSLM

def calculate_slot_f1(all_true_tags, all_pred_tags, ignore_tag="O"):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for true_seq, pred_seq in zip(all_true_tags, all_pred_tags):
        for true_t, pred_t in zip(true_seq, pred_seq):
            if true_t == ignore_tag and pred_t == ignore_tag:
                continue
            if true_t == pred_t:
                true_positives += 1
            else:
                if pred_t != ignore_tag:
                    false_positives += 1
                if true_t != ignore_tag:
                    false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def evaluate_novel_test_set():
    labels_path = EXPORT_DIR / "labels.json"
    vocab_path = MODEL_DIR / "tokenizer" / "vocab.json"
    model_path = MODEL_DIR / "voice_nav_slm.pt"
    test_novel_path = DATA_DIR / "test_novel.jsonl"

    if not test_novel_path.exists():
        msg = f"Error: Held-out novel test set not found at {test_novel_path}"
        print(msg)
        return {"error": msg}

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

    all_true_tags = []
    all_pred_tags = []
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

            pred_intent = idx2intent[pred_intent_id]
            if pred_intent == true_intent:
                correct_intent += 1

            slot_pred_ids = slot_logits[0].argmax(dim=-1).tolist()
            pred_tags = [idx2tag.get(idx, "O") for idx in slot_pred_ids[1:-1]]

            all_true_tags.append(true_tags)
            all_pred_tags.append(pred_tags)

    intent_acc = (correct_intent / total_intent) * 100.0
    prec, rec, f1 = calculate_slot_f1(all_true_tags, all_pred_tags)
    avg_conf = sum(confidence_scores) / len(confidence_scores)

    results = {
        "total_test_samples": total_intent,
        "intent_accuracy_percent": round(intent_acc, 2),
        "slot_precision_percent": round(prec * 100, 2),
        "slot_recall_percent": round(rec * 100, 2),
        "slot_f1_percent": round(f1 * 100, 2),
        "average_confidence": round(avg_conf, 4),
        "passed_target_bar": bool(intent_acc >= 90.0)
    }

    out_file = EVAL_DIR / "eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    evaluate_novel_test_set()
