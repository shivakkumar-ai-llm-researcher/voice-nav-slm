package com.firstclick.voicenav

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.nio.LongBuffer

data class SlotTagResult(val token: String, val tag: String)
data class VoiceNavResult(val intent: String, val confidence: Float, val slots: List<SlotTagResult>)

class VoiceNavInferenceEngine(context: Context, vocabJsonAsset: String, modelAsset: String) {

    private val word2idx = mutableMapOf<String, Long>()
    private val idx2intent = mutableMapOf<Int, String>()
    private val idx2tag = mutableMapOf<Int, String>()
    private var unkIdx: Long = 1L

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        // Load Vocab
        val vocabStr = context.assets.open(vocabJsonAsset).bufferedReader().use { it.readText() }
        val jsonObj = JSONObject(vocabStr)

        val w2iObj = jsonObj.getJSONObject("word2idx")
        w2iObj.keys().forEach { k -> word2idx[k] = w2iObj.getLong(k) }
        unkIdx = word2idx["<UNK>"] ?: 1L

        val i2iObj = jsonObj.getJSONObject("idx2intent")
        i2iObj.keys().forEach { k -> idx2intent[k.toInt()] = i2iObj.getString(k) }

        val i2tObj = jsonObj.getJSONObject("idx2tag")
        i2tObj.keys().forEach { k -> idx2tag[k.toInt()] = i2tObj.getString(k) }

        // Load Model Bytes
        val modelBytes = context.assets.open(modelAsset).readBytes()
        session = env.createSession(modelBytes, OrtSession.SessionOptions())
    }

    fun predict(text: String): VoiceNavResult {
        val rawTokens = text.trim().lowercase().split("\\s+".toRegex())
        val tokens = listOf("<CLS>") + rawTokens + listOf("<SEP>")

        val inputIds = tokens.map { word2idx[it] ?: unkIdx }
        val attentionMask = List(tokens.size) { 1L }

        val shape = longArrayOf(1, tokens.size.toLong())
        val tensorInputIds = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds.toLongArray()), shape)
        val tensorMask = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask.toLongArray()), shape)

        val inputs = mapOf(
            "input_ids" to tensorInputIds,
            "attention_mask" to tensorMask
        )

        val outputs = session.run(inputs)
        val intentProbs = (outputs[0].value as Array<FloatArray>)[0]

        var maxIntentIdx = 0
        var maxProb = -1f
        for (i in intentProbs.indices) {
            if (intentProbs[i] > maxProb) {
                maxProb = intentProbs[i]
                maxIntentIdx = i
            }
        }
        val intent = idx2intent[maxIntentIdx] ?: "UNKNOWN"

        val slotProbs = (outputs[1].value as Array<Array<FloatArray>>)[0] // [seqLen, numSlots]
        val slots = mutableListOf<SlotTagResult>()

        for (t in 1 until tokens.size - 1) {
            val tokenProbs = slotProbs[t]
            var maxSlotIdx = 0
            var maxSlotProb = -1f
            for (s in tokenProbs.indices) {
                if (tokenProbs[s] > maxSlotProb) {
                    maxSlotProb = tokenProbs[s]
                    maxSlotIdx = s
                }
            }
            val tag = idx2tag[maxSlotIdx] ?: "O"
            slots.add(SlotTagResult(rawTokens[t - 1], tag))
        }

        return VoiceNavResult(intent, maxProb, slots)
    }
}
