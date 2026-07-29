/**
 * VoiceNavSLM Web Runtime Engine
 * Uses ONNX Runtime Web (onnxruntime-web) for in-browser intent classification and slot extraction.
 * Single source of truth for labels via labels.json.
 */

class VoiceNavInferenceEngine {
  constructor(labelsMeta, vocabMeta, confidenceThreshold = 0.85) {
    this.word2idx = vocabMeta.word2idx;
    this.intent2idx = labelsMeta.intent2idx;
    this.idx2intent = labelsMeta.idx2intent;
    this.tag2idx = labelsMeta.tag2idx;
    this.idx2tag = labelsMeta.idx2tag;
    this.confidenceThreshold = confidenceThreshold;
    this.session = null;
  }

  async loadModel(modelPathOrBuffer) {
    if (typeof ort === 'undefined') {
      throw new Error("onnxruntime-web ('ort') library is not loaded. Include ort.min.js in your web app.");
    }
    this.session = await ort.InferenceSession.create(modelPathOrBuffer);
    console.log("VoiceNavSLM ONNX Web Session initialized successfully.");
  }

  preprocess(text) {
    const rawTokens = text.toLowerCase().trim().split(/\s+/);
    const tokens = ["<CLS>", ...rawTokens, "<SEP>"];
    
    let inputIds = tokens.map(t => this.word2idx[t] !== undefined ? this.word2idx[t] : (this.word2idx["<UNK>"] || 1));
    if (inputIds.length > 64) {
      inputIds = inputIds.slice(0, 64);
    }
    const attentionMask = new Array(inputIds.length).fill(1);

    return {
      rawTokens,
      inputIds: new Int32Array(inputIds),
      attentionMask: new Int32Array(attentionMask),
      seqLen: inputIds.length
    };
  }

  async predict(text) {
    if (!this.session) {
      throw new Error("Model session not loaded. Call loadModel() first.");
    }

    const { rawTokens, inputIds, attentionMask, seqLen } = this.preprocess(text);

    const tensorInputIds = new ort.Tensor('int64', BigInt64Array.from(inputIds.map(x => BigInt(x))), [1, seqLen]);
    const tensorMask = new ort.Tensor('int64', BigInt64Array.from(attentionMask.map(x => BigInt(x))), [1, seqLen]);

    const results = await this.session.run({
      input_ids: tensorInputIds,
      attention_mask: tensorMask
    });

    const intentProbs = results.intent_probs.data;
    let maxIntentIdx = 0;
    let maxIntentProb = -1;
    for (let i = 0; i < intentProbs.length; i++) {
      if (intentProbs[i] > maxIntentProb) {
        maxIntentProb = intentProbs[i];
        maxIntentIdx = i;
      }
    }
    const predictedIntent = this.idx2intent[maxIntentIdx] || "UNKNOWN";
    const requiresFallback = maxIntentProb < this.confidenceThreshold;

    const slotProbs = results.slot_probs.data;
    const numSlots = Object.keys(this.tag2idx).length;
    const slots = [];

    for (let t = 1; t < seqLen - 1; t++) {
      let maxSlotIdx = 0;
      let maxSlotProb = -1;
      for (let s = 0; s < numSlots; s++) {
        const prob = slotProbs[t * numSlots + s];
        if (prob > maxSlotProb) {
          maxSlotProb = prob;
          maxSlotIdx = s;
        }
      }
      const tag = this.idx2tag[maxSlotIdx] || "O";
      slots.push({ token: rawTokens[t - 1], tag });
    }

    return {
      query: text,
      intent: requiresFallback ? "FALLBACK_CLARIFICATION" : predictedIntent,
      predictedIntent,
      confidence: parseFloat(maxIntentProb.toFixed(4)),
      confidenceThreshold: this.confidenceThreshold,
      requiresFallback,
      slots
    };
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = VoiceNavInferenceEngine;
}
