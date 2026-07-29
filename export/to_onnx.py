import sys
import json
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
EXPORT_DIR = Path(__file__).parent

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(MODEL_DIR))

from model import VoiceNavSLM

VOCAB_PATH = MODEL_DIR / "tokenizer" / "vocab.json"
LABELS_PATH = EXPORT_DIR / "labels.json"
MODEL_PATH = MODEL_DIR / "voice_nav_slm.pt"
TORCHSCRIPT_PATH = EXPORT_DIR / "voice_nav_slm_traced.pt"
ONNX_PATH = EXPORT_DIR / "voice_nav_slm.onnx"
ONNX_INT8_PATH = EXPORT_DIR / "voice_nav_slm_int8.onnx"

class VoiceNavSLMExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        intent_logits, slot_logits = self.model(input_ids, attention_mask)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        slot_probs = torch.softmax(slot_logits, dim=-1)
        return intent_probs, slot_probs

def export_models():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_meta = json.load(f)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_meta = json.load(f)

    word2idx = vocab_meta["word2idx"]
    intent2idx = labels_meta["intent2idx"]
    tag2idx = labels_meta["tag2idx"]

    print(f"Loading PyTorch model from {MODEL_PATH}...")
    pytorch_model = VoiceNavSLM(
        vocab_size=len(word2idx),
        num_intents=len(intent2idx),
        num_slots=len(tag2idx),
        d_model=128,
        num_layers=4,
        num_heads=4,
        d_ff=512,
        max_seq_len=64
    )
    pytorch_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    pytorch_model.eval()

    export_model = VoiceNavSLMExportWrapper(pytorch_model)
    export_model.eval()

    batch_size = 1
    seq_len = 16
    dummy_input_ids = torch.randint(0, len(word2idx), (batch_size, seq_len), dtype=torch.long)
    dummy_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Exporting TorchScript model to {TORCHSCRIPT_PATH}...")
    traced_model = torch.jit.trace(export_model, (dummy_input_ids, dummy_mask))
    traced_model.save(str(TORCHSCRIPT_PATH))
    print(f"TorchScript export successful: {TORCHSCRIPT_PATH} ({TORCHSCRIPT_PATH.stat().st_size / (1024*1024):.2f} MB)")

    try:
        import onnx
        print(f"Exporting ONNX model to {ONNX_PATH}...")
        torch.onnx.export(
            export_model,
            (dummy_input_ids, dummy_mask),
            str(ONNX_PATH),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["intent_probs", "slot_probs"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "intent_probs": {0: "batch_size"},
                "slot_probs": {0: "batch_size", 1: "seq_len"}
            }
        )
        print(f"ONNX export successful: {ONNX_PATH} ({ONNX_PATH.stat().st_size / (1024*1024):.2f} MB)")
    except ImportError:
        print("\nNote: 'onnx' Python package is not installed. TorchScript model (.pt) ready for runtime deployment.")

if __name__ == "__main__":
    export_models()
