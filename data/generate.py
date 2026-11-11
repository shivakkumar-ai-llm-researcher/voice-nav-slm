import os
import json
import random
import yaml
from pathlib import Path

random.seed(42)

SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "intents.yaml"
OUTPUT_DIR = Path(__file__).parent / "generated"

PREFIXES = [
    "", "", "",
    "please ",
    "can you ",
    "could you ",
    "hey assistant ",
    "assistant ",
    "kindly ",
    "i want to "
]

SUFFIXES = [
    "", "", "",
    " please",
    " right now",
    " for me",
    " thanks"
]

FILLER_QUERIES = [
    "John Doe", "Jane Smith", "Acme Corp", "TechCorp", "Q3 Sales",
    "annual report", "pending leave", "marketing lead", "high priority",
    "software engineer", "invoice #1042", "project alpha", "client meeting",
    "unpaid salary", "sales pipeline", "product roadmap", "active deals"
]

FILTER_KEYS = ["status", "date", "priority", "department", "amount", "stage", "assignee"]
FILTER_VALUES = ["pending", "approved", "high", "urgent", "closed won", "marketing", "sales", "completed"]

def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def tokenize_and_tag(template, slot_replacements):
    import re
    tokens = []
    bio_tags = []
    slot_values = {}
    slots_info = []

    parts = re.split(r'(\{[\w_]+\})', template)

    for part in parts:
        if not part:
            continue
        match = re.match(r'\{([\w_]+)\}', part)
        if match:
            slot_name = match.group(1)
            phrase, canonical = slot_replacements[slot_name]
            slot_values[slot_name] = canonical
            
            phrase_words = phrase.split()
            start_token_idx = len(tokens)
            for idx, word in enumerate(phrase_words):
                tokens.append(word)
                tag = f"B-{slot_name}" if idx == 0 else f"I-{slot_name}"
                bio_tags.append(tag)
            end_token_idx = len(tokens)
            slots_info.append({
                "slot": slot_name,
                "value": canonical,
                "phrase": phrase,
                "start_token": start_token_idx,
                "end_token": end_token_idx
            })
        else:
            words = part.strip().split()
            for word in words:
                tokens.append(word)
                bio_tags.append("O")

    return tokens, bio_tags, slot_values, slots_info

def build_catalogs(schema):
    screens_catalog = []
    for sc in schema.get("screens", []):
        name = sc["name"]
        synonyms = [name.replace("_", " ")] + sc.get("synonyms", [])
        for syn in synonyms:
            screens_catalog.append((syn, name))

    features_catalog = []
    for ft in schema.get("features", []):
        name = ft["name"]
        synonyms = [name.replace("_", " ")] + ft.get("synonyms", [])
        for syn in synonyms:
            features_catalog.append((syn, name))

    items_catalog = []
    for item in schema.get("items", []):
        name = item["name"]
        synonyms = [name.replace("_", " ")] + item.get("synonyms", [])
        for syn in synonyms:
            items_catalog.append((syn, name))

    directions_catalog = [(d, d) for d in schema.get("directions", ["up", "down", "left", "right", "top", "bottom"])]

    toggle_values_catalog = []
    for mode, val_list in schema.get("toggle_values", {}).items():
        for v in val_list:
            toggle_values_catalog.append((v, mode))

    return screens_catalog, features_catalog, items_catalog, directions_catalog, toggle_values_catalog

def generate_samples(schema, catalogs, num_samples_per_intent=1000):
    screens_catalog, features_catalog, items_catalog, directions_catalog, toggle_values_catalog = catalogs
    dataset = []

    intents_config = schema.get("intents", {})

    for intent_name, intent_data in intents_config.items():
        # Standard training templates ONLY
        examples = intent_data.get("examples", [])

        for _ in range(num_samples_per_intent):
            template = random.choice(examples)
            prefix = random.choice(PREFIXES)
            suffix = random.choice(SUFFIXES)

            full_template = (prefix + template + suffix).strip()

            slot_replacements = {}
            if "{target_screen}" in full_template:
                syn, canonical = random.choice(screens_catalog)
                slot_replacements["target_screen"] = (syn, canonical)

            if "{target_feature}" in full_template:
                syn, canonical = random.choice(features_catalog)
                slot_replacements["target_feature"] = (syn, canonical)

            if "{target_item}" in full_template:
                syn, canonical = random.choice(items_catalog)
                slot_replacements["target_item"] = (syn, canonical)

            if "{direction}" in full_template:
                syn, canonical = random.choice(directions_catalog)
                slot_replacements["direction"] = (syn, canonical)

            if "{value}" in full_template:
                syn, canonical = random.choice(toggle_values_catalog)
                slot_replacements["value"] = (syn, canonical)

            if "{query}" in full_template:
                query_val = random.choice(FILLER_QUERIES)
                slot_replacements["query"] = (query_val, query_val)

            if "{filter_key}" in full_template:
                fk = random.choice(FILTER_KEYS)
                slot_replacements["filter_key"] = (fk, fk)

            if "{filter_value}" in full_template:
                fv = random.choice(FILTER_VALUES)
                slot_replacements["filter_value"] = (fv, fv)

            try:
                tokens, bio_tags, slot_vals, slots_info = tokenize_and_tag(full_template, slot_replacements)
                text = " ".join(tokens)

                dataset.append({
                    "text": text,
                    "tokens": tokens,
                    "intent": intent_name,
                    "bio_tags": bio_tags,
                    "slots": slots_info,
                    "slot_values": slot_vals
                })
            except KeyError:
                continue

    random.shuffle(dataset)
    return dataset

def main():
    schema = load_schema()
    catalogs = build_catalogs(schema)

    print("Generating standard training and validation synthetic dataset...")
    all_data = generate_samples(schema, catalogs, num_samples_per_intent=1200)

    total = len(all_data)
    train_end = int(0.85 * total)

    train_data = all_data[:train_end]
    val_data = all_data[train_end:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def write_jsonl(filepath, items):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    write_jsonl(OUTPUT_DIR / "train.jsonl", train_data)
    write_jsonl(OUTPUT_DIR / "val.jsonl", val_data)

    print(f"\nTraining/Validation JSONL Dataset Generation Complete!")
    print(f"  Train: {len(train_data):4d} ({OUTPUT_DIR / 'train.jsonl'})")
    print(f"  Val:   {len(val_data):4d} ({OUTPUT_DIR / 'val.jsonl'})")

if __name__ == "__main__":
    main()
