import json
import yaml
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "intents.yaml"
OUTPUT_PATH = Path(__file__).parent / "labels.json"

def generate_labels():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    intents_config = schema.get("intents", {})
    intents = sorted(list(intents_config.keys()))

    slots = set()
    for intent_data in intents_config.values():
        for slot in intent_data.get("slots", []):
            slots.add(slot)

    slot_tags = ["O"]
    for slot in sorted(list(slots)):
        slot_tags.append(f"B-{slot}")
        slot_tags.append(f"I-{slot}")

    intent2idx = {intent: idx for idx, intent in enumerate(intents)}
    idx2intent = {idx: intent for intent, idx in intent2idx.items()}

    tag2idx = {tag: idx for idx, tag in enumerate(slot_tags)}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

    label_data = {
        "intents": intents,
        "intent2idx": intent2idx,
        "idx2intent": idx2intent,
        "slot_tags": slot_tags,
        "tag2idx": tag2idx,
        "idx2tag": idx2tag
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(label_data, f, indent=2)

    print(f"Generated single-source-of-truth label file: {OUTPUT_PATH}")
    print(f"  Intents ({len(intents)}): {intents}")
    print(f"  Slot Tags ({len(slot_tags)}): {slot_tags}")

if __name__ == "__main__":
    generate_labels()
