---
name: voice-nav-slm
description: >
  Use this skill whenever working on the in-app voice navigation project
  (tiny intent/slot NLU model for Firstclick HRMS and SalesPro CRM voice
  commands). Triggers on: "voice nav", "intent model", "slot filling",
  "voice-nav-slm", schema files under schema/intents.yaml, synthetic data
  generation for commands, ONNX export of the NLU model, or any request to
  add a new screen/intent/slot to the voice navigation system. Do NOT use
  for the unrelated decoder-only ShivCoder/SLM-from-scratch project, or for
  the "slm" agentic coding CLI tool itself — this skill is scoped only to
  the voice-navigation encoder model and its pipeline.
---

# Voice Navigation Tiny NLU Model — Project Skill

## Purpose

Build a **tiny, on-device intent + slot classification model** that converts
a speech-to-text transcript into a structured app-navigation command for
Firstclick HRMS and SalesPro CRM. This is a closed-domain, bidirectional
encoder classification task — **NOT** open-domain text generation. Any
model working on this project must not drift toward building a
general-purpose chatbot, a decoder-only generator, or an open-vocabulary
system. The output space is always a fixed, enumerable set of
{intent, slots} defined in `schema/intents.yaml`.

Read this file fully before writing any code, generating any data, or
proposing any architecture change. If a request conflicts with a rule
below, follow this file and flag the conflict to the user rather than
silently choosing one.

---

## Non-negotiable architecture decisions

Do not re-litigate these; they were decided deliberately and changing them
without being asked breaks the accuracy target and platform-portability
goals of the project.

1. **Model type**: Bidirectional transformer **encoder** (BERT-style), NOT
   decoder-only. Two heads on top of a shared encoder:
   - Intent head: single-label softmax classification.
   - Slot head: per-token BIO tagging.
2. **Size target**: 1–8M parameters total. Hidden dim 128–256, 4–6 layers,
   4 heads. If a proposed change pushes past ~10M params, stop and flag it
   — the whole point is on-device, zero-cost, real-time inference on
   mobile/web/desktop.
3. **Positional encoding / norm / FFN**: RoPE, RMSNorm, SwiGLU — consistent
   with the user's existing from-scratch SLM conventions, reused here for
   familiarity and code reuse, not because they're strictly required for a
   model this small.
4. **Tokenizer**: small custom WordPiece/BPE, 2k–5k vocab, trained only on
   the app's own domain text (screen names, command phrasing, supported
   languages) — never reuse a large general-purpose tokenizer; it bloats
   the embedding table, which dominates parameter count at this scale.
5. **STT is out of scope for this model.** Speech-to-text is handled
   upstream by whisper.cpp (tiny/base). This skill's model only ever
   consumes text. Do not propose merging STT into this model.
6. **Deployment target**: mobile (Android/iOS), web (browser via WASM),
   desktop. No embedded/microcontroller constraint — quantization can
   target int8 for size/speed, but does not need to satisfy no-OS/RAM-
   starved microcontroller limits.
7. **Runtime**: export to ONNX, run via ONNX Runtime (+ ONNX Runtime Web
   for browser). One trained artifact, all four platforms.
8. **Fully free/open-source stack.** Do not introduce paid SDKs, per-query
   billed APIs, or licensed models anywhere in this pipeline (STT, wake
   word, TTS, training, or inference). If a suggested library or tool has
   a paid tier requirement for production use, flag it before adding it as
   a dependency.

---

## Schema is the single source of truth

`schema/intents.yaml` defines every intent, its slots, and example
templates, plus the full list of registered screens/modules pulled from
each app's module-toggle backend config.

**Rules for editing the schema:**
- Every new screen/action added to Firstclick HRMS or SalesPro CRM's
  module toggle system should have a corresponding entry added here before
  training data is regenerated. Do not hardcode screen names anywhere else
  in the codebase — always read from this file.
- Intent names are UPPER_SNAKE_CASE. Slot names are lower_snake_case.
- Keep the schema tenant-agnostic: it should describe the *union* of all
  possible screens across all tenants. Per-tenant filtering (based on which
  modules a given client has enabled) happens at inference/deployment time,
  not by maintaining separate schema files per tenant.
- When ambiguous whether something is a new intent or a new slot value
  under an existing intent (e.g. "open payroll" vs "open settings"), prefer
  reusing an existing intent (`NAVIGATE`) with a new slot value
  (`target_screen: payroll`) over creating a new intent. New intents should
  only be added for genuinely distinct action types (e.g. `TOGGLE` vs
  `NAVIGATE` vs `SEARCH`), not per-screen.

---

## Synthetic data generation rules

Location: `data/generate.py`.

- Generate from templates × schema entries, not from hand-written flat
  example lists. Every template must be traceable back to a schema intent.
- Include realistic noise on a meaningful fraction of examples (not all):
  filler words ("uh", "please", "can you"), reordering, minor typos,
  and STT-plausible near-homophone substitutions — since real input to
  this model comes after whisper.cpp, not clean typed text.
- Balance classes: no intent should have drastically fewer examples than
  others purely due to fewer hand-written templates. If a screen list has
  50 screens and one intent naturally covers all of them while another
  covers 3 actions, generate proportionally so the intent head doesn't
  learn a skewed prior.
- Always produce a held-out test set from phrasings **not seen in
  training templates** (paraphrases a real user might say), not just a
  random split of the same template family. A random split of
  template-generated data will overstate accuracy — measure the real
  target (90%+ intent accuracy) against genuinely novel phrasing, not
  templated data leakage.
- Output format: JSON lines, one example per line:
  `{"text": ..., "intent": ..., "slots": [{"slot": ..., "value": ..., "start": ..., "end": ...}]}`

---

## Training rules

Location: `model/train.py`, targeting Google Colab free T4.

- Multi-task loss: cross-entropy on intent head + token-level cross-entropy
  on slot BIO tags, summed (optionally weighted — start at 1:1, adjust only
  if one task's metric lags badly).
- Track both **intent accuracy** and **slot F1** separately every eval
  step — do not report a single blended metric, since they can diverge
  (e.g. intent right, slot boundary off by one token).
- 90% intent accuracy on the held-out novel-phrasing test set is the
  Phase 1 bar. Do not consider the model "done" based on validation loss
  alone — always check against the novel-phrasing set specifically.
- Save checkpoints to Drive per the user's existing training script
  convention (matches `train_colab.py` pattern from the from-scratch SLM
  project) so training can resume across Colab session limits.

---

## Export & quantization

Location: `export/to_onnx.py`.

- Export both heads in a single ONNX graph (one forward pass, two output
  tensors), not two separate models — avoids duplicate encoder inference
  cost at runtime.
- Quantize to int8 post-training. Benchmark accuracy before/after
  quantization on the same held-out set — if quantization drops intent
  accuracy more than ~1–2 points, flag it rather than silently shipping
  the quantized version.
- Verify the exported model runs identically via ONNX Runtime on at least
  one non-training environment (e.g. plain Python ONNX Runtime, not just
  the training framework) before wiring into any platform runtime.

---

## Platform integration conventions

- Android: Kotlin, fits existing Jetpack Compose stack. ONNX Runtime
  Mobile.
- iOS: Swift, ONNX Runtime Mobile.
- Web: ONNX Runtime Web (WASM), fully client-side — no backend round trip
  required for inference.
- Desktop: ONNX Runtime (Node/Python/C++ depending on the shipping app).
- All four platforms consume the **same** `.onnx` artifact and the
  **same** `intents.yaml`-derived label list — never fork label
  ordering or slot indices per platform; a mismatch here silently
  corrupts predictions without erroring.

---

## Accuracy checklist before declaring a milestone done

Before reporting a training run or pipeline change as successful, confirm:

- [ ] Intent accuracy ≥ 90% on the novel-phrasing held-out set (not the
      templated split)
- [ ] Slot F1 reported separately, not folded into intent accuracy
- [ ] Quantized (int8) model re-evaluated, not just the fp32 checkpoint
- [ ] Confidence threshold behavior checked: low-confidence inputs route
      to fallback/clarification, not a forced top-1 guess
- [ ] New schema entries (if any) reflected in regenerated training data
      before retraining, not left stale
- [ ] No paid/licensed dependency introduced anywhere in the pipeline

---

## What NOT to do

- Do not turn this into an open-domain generative assistant "for
  flexibility." The closed-domain constraint is the entire accuracy and
  size advantage — don't erode it.
- Do not fork the schema per tenant into separate files. Filter at
  inference time from one canonical schema.
- Do not add STT, wake-word, or TTS logic into this model or repo — those
  are separate pipeline stages with their own tooling (whisper.cpp,
  openWakeWord, native/Piper TTS).
- Do not introduce Picovoice, Dialogflow, Lex, or any other paid NLU/STT
  SDK as a dependency of this pipeline.
- Do not silently increase model size past ~10M params to chase marginal
  accuracy gains — flag the tradeoff to the user instead.
