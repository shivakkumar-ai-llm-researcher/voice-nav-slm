import json
from pathlib import Path
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

DATA_PATH = Path(__file__).parent.parent / "data" / "generated" / "train.jsonl"
TOKENIZER_DIR = Path(__file__).parent / "tokenizer"

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>"]

def extract_corpus():
    corpus_file = TOKENIZER_DIR / "corpus.txt"
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    
    lines = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            lines.append(item["text"])

    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Extracted {len(lines)} sentences into {corpus_file}")
    return corpus_file

def train_wordpiece_tokenizer():
    corpus_file = extract_corpus()

    # Initialize WordPiece tokenizer
    tokenizer = Tokenizer(models.WordPiece(unk_token="<UNK>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordPieceTrainer(
        vocab_size=2500,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=1
    )

    print("Training WordPiece Tokenizer...")
    tokenizer.train(files=[str(corpus_file)], trainer=trainer)

    # Save tokenizer config and vocab.json
    tokenizer.save(str(TOKENIZER_DIR / "tokenizer.json"))
    
    vocab = tokenizer.get_vocab()

    # Generate custom vocab.json for simple array/dict runtime lookups
    word2idx = vocab
    idx2word = {idx: token for token, idx in vocab.items()}

    meta = {
        "word2idx": word2idx,
        "idx2word": idx2word,
        "vocab_size": len(vocab),
        "special_tokens": SPECIAL_TOKENS
    }

    with open(TOKENIZER_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Tokenizer trained successfully!")
    print(f"  Vocab Size: {len(vocab)}")
    print(f"  Saved artifacts to: {TOKENIZER_DIR}")

if __name__ == "__main__":
    train_wordpiece_tokenizer()
