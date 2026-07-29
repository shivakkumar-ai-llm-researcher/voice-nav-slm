# AGENT.md — Voice Navigation Tiny NLU Model (voice-nav-slm)

This document is the project-level reference for any AI agent (Gemini,
Claude, or other) working on this repository. Read this alongside
`SKILL.md` before making changes — `SKILL.md` defines behavioral rules and
constraints; this file defines the project's architecture, structure, and
build order.

---

## 1. Project summary

A tiny (1–8M parameter) bidirectional transformer encoder that converts a
speech-to-text transcript into a structured in-app navigation command
(intent + slots) for **Firstclick HRMS** and **SalesPro CRM**. Runs fully
on-device across mobile, web, and desktop — no cloud inference, no
per-query cost, no paid SDK dependency.

This is a closed-domain classification problem, not open-domain language
generation. The full pipeline:

```
Mic input → whisper.cpp (STT, on-device) → voice-nav-slm (intent + slot
model, ONNX Runtime, on-device) → app action dispatch
```

Only the middle stage (intent + slot model) is built in this repo. STT
(whisper.cpp) and wake word (openWakeWord, optional) are external
dependencies, not built here.

---

## 2. Goals / non-goals

**Goals**
- ≥90% intent accuracy on novel (non-templated) phrasing
- Fully free/open-source stack, zero recurring cost
- Single trained model artifact deployed identically to Android, iOS, web,
  and desktop
- Tenant-aware: intent schema filterable per client based on which modules
  (Firstclick/SalesPro module-toggle config) are enabled for that tenant
- Fast enough for real-time on-device inference on a mid-range phone or
  laptop-class CPU

**Non-goals**
- Not a general-purpose voice assistant or chatbot
- Not doing STT or wake-word detection itself
- Not targeting microcontroller/no-OS embedded hardware
- Not trying to beat commercial SDKs (Picovoice Rhino, Dialogflow) on raw
  benchmark accuracy — the win condition is tenant-awareness, language
  coverage, app-state context, and zero cost, not head-to-head accuracy

---

## 3. Repository structure

```
voice-nav-slm/
├── AGENT.md                    # this file
├── SKILL.md                    # behavioral rules for AI agents working here
├── schema/
│   └── intents.yaml             # single source of truth: intents, slots, screens
├── data/
│   ├── generate.py              # synthetic data generator (schema -> training data)
│   └── generated/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test_novel.jsonl      # held-out, non-templated phrasing only
├── model/
│   ├── model.py                  # encoder + intent head + slot head
│   ├── tokenizer_train.py         # small WordPiece/BPE tokenizer trainer
│   ├── train.py                   # Colab-targeted training script
│   └── tokenizer/                  # trained tokenizer artifacts
├── export/
│   ├── to_onnx.py                  # ONNX export
│   └── quantize.py                 # int8 post-training quantization
├── eval/
│   └── evaluate.py                  # intent accuracy + slot F1, fp32 vs quantized
└── runtime/
    ├── android/                      # Kotlin + ONNX Runtime Mobile integration
    ├── ios/                          # Swift + ONNX Runtime Mobile integration
    ├── web/                          # ONNX Runtime Web (WASM), client-side only
    └── desktop/                      # ONNX Runtime (Node/Python/C++)
```

---

## 4. Architecture details

- **Encoder**: bidirectional transformer, 4–6 layers, hidden dim 128–256,
  4 heads, RoPE positional encoding, RMSNorm, SwiGLU FFN.
- **Intent head**: linear + softmax over the pooled `[CLS]`-equivalent
  representation, single-label classification.
- **Slot head**: linear + softmax per token over BIO tags, sharing the
  same encoder output as the intent head (single forward pass).
- **Tokenizer**: custom WordPiece/BPE, 2k–5k vocab, trained only on
  in-domain text (schema templates + any real logged transcripts once
  available). Not a reused general-purpose tokenizer.
- **Context conditioning (Phase 2)**: current screen/module ID embedded
  and concatenated to the encoder input, so ambiguous commands ("open
  report") resolve using live app state, not just the utterance text.
- **Confidence gating**: softmax confidence on the intent head is
  thresholded at inference time; below-threshold predictions should
  trigger a clarification prompt or fallback rather than a forced guess.

---

## 5. Approved dependencies

Only the following are approved for this project. Anything not on this
list requires a explicit decision before adding, since the project's core
constraint is zero recurring cost and no vendor lock-in.

| Purpose | Approved |
|---|---|
| STT | whisper.cpp (tiny/base) |
| Wake word (optional) | openWakeWord |
| TTS (optional) | Native OS TTS (Android `TextToSpeech`, iOS `AVSpeechSynthesizer`) or Piper TTS |
| Model training | PyTorch |
| Tokenizer | Hugging Face `tokenizers` library (WordPiece/BPE trainer only — not pulling pretrained vocab) |
| Model export | ONNX, `onnxruntime` |
| Web inference | `onnxruntime-web` |
| Mobile inference | ONNX Runtime Mobile (Android/iOS) |
| Training compute | Google Colab free tier (T4) |

**Explicitly not approved**: Picovoice (Porcupine/Rhino/Cheetah/Leopard),
Google Dialogflow, Amazon Lex, any cloud STT/NLU billed per query, any
paid TTS API.

---

## 6. Build order

Follow this sequence — later steps depend on earlier ones being stable,
not just present.

1. **Schema** (`schema/intents.yaml`) — enumerate all intents, slots, and
   the full screen/module list pulled from Firstclick HRMS and SalesPro
   CRM's module-toggle backend config. Nothing downstream starts until
   this is in a reviewable state.
2. **Synthetic data generator** (`data/generate.py`) — templates × schema
   → `train.jsonl` / `val.jsonl`, plus a hand-curated `test_novel.jsonl`
   of paraphrased, non-templated phrasing.
3. **Tokenizer** (`model/tokenizer_train.py`) — train the small
   WordPiece/BPE vocab on the generated corpus.
4. **Model + training script** (`model/model.py`, `model/train.py`) —
   build the encoder + two heads, train on Colab T4, checkpoint to Drive.
5. **Evaluation** (`eval/evaluate.py`) — intent accuracy + slot F1 on
   `test_novel.jsonl`. Do not proceed to export until the ≥90% intent
   accuracy bar is met on this held-out set specifically.
6. **Export + quantization** (`export/to_onnx.py`,
   `export/quantize.py`) — export ONNX graph with both heads in one
   forward pass, quantize to int8, re-run evaluation on the quantized
   model to confirm accuracy holds.
7. **Platform runtime integration** (`runtime/android`, `runtime/ios`,
   `runtime/web`, `runtime/desktop`) — wire the same `.onnx` artifact and
   label list into each platform, using each platform's ONNX Runtime
   binding.
8. **Context conditioning (Phase 2)** — add screen/module-state
   conditioning input once Phase 1 baseline accuracy is validated.
9. **Confidence-gated fallback (Phase 2)** — wire low-confidence routing
   to clarification/fallback UI once real usage data exists to calibrate
   the threshold.
10. **Feedback loop (Phase 3)** — log low-confidence/misrouted commands
    (opt-in, local), feed back into the synthetic data generator as new
    template variants, periodically retrain.

---

## 7. Conventions

- Screen/module names always sourced from `schema/intents.yaml` — never
  hardcoded elsewhere.
- Intent names: `UPPER_SNAKE_CASE`. Slot names: `lower_snake_case`.
- One canonical schema, filtered per tenant at inference time — no
  per-tenant schema forks.
- Label ordering (intent list, slot tag list) must be identical across
  all four platform runtimes and the exported ONNX model — a mismatch
  here corrupts predictions silently. Keep the label list generation as
  a single script output consumed by both training and all runtime
  integrations, not retyped per platform.
- Accuracy is always measured on `test_novel.jsonl` (non-templated), not
  on a random split of generated data.

---

## 8. Status / open decisions

- [ ] Full screen/module list not yet extracted from Firstclick HRMS /
      SalesPro CRM module-toggle config — required before `schema/intents.yaml`
      can be finalized.
- [ ] Wake word vs button-press-to-talk not yet decided per platform.
- [ ] whisper.cpp model size (tiny vs base) not yet benchmarked against
      real Tamil/Indian-accented English samples.
- [ ] Confidence threshold value not yet calibrated (no real usage data
      yet — placeholder only until Phase 2).
