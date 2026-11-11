import json
import random
import yaml
from pathlib import Path

random.seed(999)

SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "intents.yaml"
OUTPUT_PATH = Path(__file__).parent / "generated" / "test_novel.jsonl"

# Adversarial & Novel phrasing templates (Completely distinct from training templates)
ADVERSARIAL_IN_DOMAIN_TEMPLATES = {
    "NAVIGATE": [
        "i'm looking to check up on {target_screen}",
        "pop open the {target_screen} view",
        "i want to see my {target_screen}",
        "can we head over to {target_screen}",
        "bring up the {target_screen} module",
        "take me straight to {target_screen}",
        "let me inspect {target_screen}",
        "jump to {target_screen} tab",
        "access {target_screen}",
        "pull up {target_screen} page"
    ],
    "SEARCH": [
        "query details for {query} in {target_screen}",
        "where can i find {query}",
        "look up records of {query} inside {target_screen}",
        "locate {query} from {target_screen}",
        "search for information regarding {query}",
        "find me {query} in {target_screen}",
        "pull up {query}"
    ],
    "TOGGLE": [
        "i'd like to {value} {target_feature}",
        "can we switch {target_feature} to {value}",
        "set {target_feature} as {value}",
        "turn {target_feature} {value} immediately",
        "make {target_feature} {value}",
        "change {target_feature} status to {value}"
    ],
    "FILTER": [
        "narrow down the list by {filter_key} {filter_value}",
        "only display items where {filter_key} is {filter_value}",
        "arrange entries by {filter_key}",
        "show records with {filter_key} set to {filter_value}",
        "filter list according to {filter_key} {filter_value}"
    ],
    "CREATE": [
        "i want to log a new {target_item}",
        "start a brand new {target_item}",
        "fill out form for {target_item}",
        "enter a {target_item} into the system",
        "submit details for {target_item}",
        "register a {target_item}"
    ],
    "BACK": [
        "get me out of this view",
        "return to where i was previously",
        "take one step backwards",
        "exit from here",
        "navigate back to main menu"
    ],
    "SCROLL": [
        "slide page {direction}",
        "pan the content {direction}",
        "swipe screen {direction}",
        "move view {direction}"
    ]
}

STT_NOISE_FILLERS = [
    "uh ", "um ", "err ", "like ", "you know ", "please uh ", "can you um "
]

OUT_OF_DOMAIN_SAMPLES = [
    "what is the weather forecast today",
    "order a pizza for lunch",
    "play some relaxing music",
    "call mom on her mobile",
    "who won the football match yesterday",
    "hello assistant how are you doing",
    "tell me a funny joke",
    "remind me to buy groceries later",
    "open YouTube app and search cat videos",
    "what is the capital of France",
    "book a flight to Mumbai",
    "set an alarm for 7 AM tomorrow",
    "calculate 450 divided by 15",
    "translate thank you to French",
    "turn off the living room light bulb",
    "what is the stock price of Google",
    "play latest news headlines",
    "who wrote Romeo and Juliet",
    "how far is the moon from Earth",
    "open Spotify player"
]

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

def generate_adversarial_test_set():
    schema = load_schema()
    catalogs = build_catalogs(schema)
    screens_catalog, features_catalog, items_catalog, directions_catalog, toggle_values_catalog = catalogs

    test_samples = []

    # 1. In-Domain Novel Paraphrases with STT Noise (800 samples)
    for intent_name, examples in ADVERSARIAL_IN_DOMAIN_TEMPLATES.items():
        for _ in range(120):
            template = random.choice(examples)
            filler = random.choice(STT_NOISE_FILLERS) if random.random() < 0.4 else ""
            full_template = (filler + template).strip()

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
                query_val = random.choice(["John Doe", "Jane Smith", "Acme Corp", "TechCorp", "Q3 Sales", "annual report"])
                slot_replacements["query"] = (query_val, query_val)

            if "{filter_key}" in full_template:
                fk = random.choice(["status", "date", "priority", "department", "amount"])
                slot_replacements["filter_key"] = (fk, fk)

            if "{filter_value}" in full_template:
                fv = random.choice(["pending", "approved", "high", "urgent", "closed won"])
                slot_replacements["filter_value"] = (fv, fv)

            try:
                tokens, bio_tags, slot_vals, slots_info = tokenize_and_tag(full_template, slot_replacements)
                text = " ".join(tokens)

                test_samples.append({
                    "text": text,
                    "tokens": tokens,
                    "intent": intent_name,
                    "bio_tags": bio_tags,
                    "slots": slots_info,
                    "slot_values": slot_vals,
                    "is_ood": False
                })
            except KeyError:
                continue

    # 2. Out-of-Domain (OOD) Ambiguous Queries (200 samples)
    for _ in range(200):
        ood_text = random.choice(OUT_OF_DOMAIN_SAMPLES)
        tokens = ood_text.split()
        bio_tags = ["O"] * len(tokens)

        test_samples.append({
            "text": ood_text,
            "tokens": tokens,
            "intent": "OUT_OF_DOMAIN",
            "bio_tags": bio_tags,
            "slots": [],
            "slot_values": {},
            "is_ood": True
        })

    random.shuffle(test_samples)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for sample in test_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Generated Genuinely Adversarial Test Dataset: {OUTPUT_PATH}")
    print(f"  Total Samples: {len(test_samples)}")
    print(f"  In-Domain Novel Samples:  {len([s for s in test_samples if not s['is_ood']])}")
    print(f"  Out-of-Domain (OOD) Samples: {len([s for s in test_samples if s['is_ood']])}")

if __name__ == "__main__":
    generate_adversarial_test_set()
