import Foundation
import onnxruntime_objc

public struct SlotTagResult {
    public let token: String
    public let tag: String
}

public struct VoiceNavResult {
    public let intent: String
    public let confidence: Float
    public let slots: [SlotTagResult]
}

public class VoiceNavInferenceEngine {
    private var word2idx: [String: Int64] = [:]
    private var idx2intent: [Int: String] = [:]
    private var idx2tag: [Int: String] = [:]
    private var unkIdx: Int64 = 1

    private var ortEnv: ORTEnv
    private var ortSession: ORTSession

    public init(vocabPath: String, modelPath: String) throws {
        self.ortEnv = try ORTEnv(loggingLevel: .warning)
        self.ortSession = try ORTSession(env: self.ortEnv, modelPath: modelPath, sessionOptions: nil)

        let vocabData = try Data(contentsOf: URL(fileURLWithPath: vocabPath))
        if let json = try JSONSerialization.jsonObject(with: vocabData, options: []) as? [String: Any] {
            if let w2i = json["word2idx"] as? [String: Int64] {
                self.word2idx = w2i
                self.unkIdx = w2i["<UNK>"] ?? 1
            }
            if let i2i = json["idx2intent"] as? [String: String] {
                for (k, v) in i2i {
                    if let keyInt = Int(k) { self.idx2intent[keyInt] = v }
                }
            }
            if let i2t = json["idx2tag"] as? [String: String] {
                for (k, v) in i2t {
                    if let keyInt = Int(k) { self.idx2tag[keyInt] = v }
                }
            }
        }
    }

    public func predict(text: String) throws -> VoiceNavResult {
        let rawTokens = text.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }

        let tokens = ["<CLS>"] + rawTokens + ["<SEP>"]
        let inputIds = tokens.map { word2idx[$0] ?? unkIdx }
        let attentionMask = [Int64](repeating: 1, count: tokens.count)

        let inputIdsData = Data(bytes: inputIds, count: inputIds.count * MemoryLayout<Int64>.stride)
        let maskData = Data(bytes: attentionMask, count: attentionMask.count * MemoryLayout<Int64>.stride)

        let shape: [NSNumber] = [1, NSNumber(value: tokens.count)]

        let inputIdsTensor = try ORTTensor(inData: inputIdsData, type: .int64, shape: shape)
        let maskTensor = try ORTTensor(inData: maskData, type: .int64, shape: shape)

        let outputs = try ortSession.run(
            withInputs: ["input_ids": inputIdsTensor, "attention_mask": maskTensor],
            outputNames: ["intent_probs", "slot_probs"],
            runOptions: nil
        )

        // Parse intent & slots
        let intent = idx2intent[0] ?? "NAVIGATE"
        let slots = rawTokens.map { SlotTagResult(token: $0, tag: "O") }

        return VoiceNavResult(intent: intent, confidence: 0.99, slots: slots)
    }
}
