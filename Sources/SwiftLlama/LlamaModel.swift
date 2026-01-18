import Foundation
import llama

class LlamaModel {
    private let model: Model
    private let configuration: Configuration
    private let context: Context
    private let vocab: Vocab
    private let sampler: UnsafeMutablePointer<llama_sampler>
    private var batch: Batch
    private var tokens: [Token]
    private var maxTokenCountOverride: Int32?
    private var promptTokenCount: Int32 = 0
    private var generatedTokenAccount: Int32 = 0
    private var ended = false

    var shouldContinue: Bool {
        generatedTokenAccount < effectiveMaxTokenCount && !ended
    }

    private var effectiveMaxTokenCount: Int32 {
        maxTokenCountOverride ?? Int32(configuration.maxTokenCount)
    }

    init(path: String, configuration: Configuration = .init()) throws {
        self.configuration = configuration
        if Self.shouldEnableLogging() {
            Self.installLogCallbackIfNeeded()
        }
        llama_backend_init()
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED)

        var model_params = llama_model_default_params()
        #if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        #endif

        guard let model = llama_model_load_from_file(path, model_params) else {
            throw SwiftLlamaError.others("Cannot load model at path \(path)")
        }
        self.model = model

        guard let context = llama_init_from_model(model, configuration.contextParameters) else {
            throw SwiftLlamaError.others("Cannot load model context")
        }
        self.context = context
        guard let vocab = llama_model_get_vocab(model) else {
            throw SwiftLlamaError.others("Cannot load model vocabulary")
        }
        self.vocab = vocab

        self.tokens = []
        self.batch = llama_batch_init(Int32(configuration.batchSize * Configuration.historySize * 2), 0, 1)

        self.sampler = llama_sampler_chain_init(llama_sampler_chain_default_params())
        llama_sampler_chain_add(
            sampler,
            llama_sampler_init_penalties(
                Int32(configuration.penaltyLastN),
                configuration.repetitionPenalty,
                configuration.frequencyPenalty,
                configuration.presencePenalty
            )
        )
        let minKeep = max(0, configuration.minKeep)
        let shouldApplyTopP = configuration.topP > 0 && (configuration.topP < 1.0 || minKeep > 0)
        if configuration.temperature > 0 {
            if configuration.topK > 0 {
                llama_sampler_chain_add(sampler, llama_sampler_init_top_k(Int32(configuration.topK)))
            }
            if shouldApplyTopP {
                llama_sampler_chain_add(sampler, llama_sampler_init_top_p(configuration.topP, minKeep))
            }
            if configuration.minP > 0 {
                llama_sampler_chain_add(sampler, llama_sampler_init_min_p(configuration.minP, minKeep))
            }
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(configuration.temperature))
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(Self.resolveSeed(configuration.seed)))
        } else {
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy())
        }

        try checkContextLength(context: context, model: model)
    }

    private func checkContextLength(context: Context, model: Model) throws {
        let n_ctx = llama_n_ctx(context)
        let n_ctx_train = llama_model_n_ctx_train(model)
        if n_ctx > n_ctx_train {
            throw SwiftLlamaError.others("Model was trained on \(n_ctx_train) context but tokens \(n_ctx) specified")
        }
    }

    func start(for prompt: Prompt, maxOutputTokens: Int?) throws {
        ended = false
        llama_sampler_reset(sampler)
        tokens = tokenize(text: prompt.prompt, addBos: true)
        promptTokenCount = Int32(tokens.count)
        if let maxOutputTokens {
            let output = max(1, maxOutputTokens)
            let total = promptTokenCount + Int32(output)
            let contextLimit = Int32(llama_n_ctx(context))
            maxTokenCountOverride = min(total, contextLimit)
        } else {
            maxTokenCountOverride = nil
        }

        batch.clear()
        tokens.enumerated().forEach { index, token in
            batch.add(token: token, position: Int32(index), seqIDs: [0], logit: false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true

        if llama_decode(context, batch) != 0 {
            throw SwiftLlamaError.decodeError
        }
        generatedTokenAccount = batch.n_tokens
    }

    func `continue`() throws -> String {
        let newToken = llama_sampler_sample(sampler, context, batch.n_tokens - 1)

        if llama_vocab_is_eog(vocab, newToken) || generatedTokenAccount >= effectiveMaxTokenCount {
            ended = true
            return ""
        }

        let piece = tokenToString(token: newToken)

        batch.clear()
        batch.add(token: newToken, position: generatedTokenAccount, seqIDs: [0], logit: true)
        generatedTokenAccount += 1

        if llama_decode(context, batch) != 0 {
            throw SwiftLlamaError.decodeError
        }
        return piece
    }

    // MARK: - Helpers

    /// Convert a sampled token to a Swift String (valid UTF-8, no interleaved \0 bytes).
    private func tokenToString(token: llama_token) -> String {
        var cap: Int32 = 32
        var buf = [CChar](repeating: 0, count: Int(cap))

        // First attempt
        var written = buf.withUnsafeMutableBufferPointer { p -> Int32 in
            guard let base = p.baseAddress else { return 0 }
            return llama_token_to_piece(vocab, token, base, cap, 0, false)
        }

        // If negative, allocate required size and retry
        if written < 0 {
            cap = -written
            buf = [CChar](repeating: 0, count: Int(cap))
            written = buf.withUnsafeMutableBufferPointer { p -> Int32 in
                guard let base = p.baseAddress else { return 0 }
                return llama_token_to_piece(vocab, token, base, cap, 0, false)
            }
        }

        let count = Int(max(0, written))
        if count == 0 { return "" }

        // Decode exact byte count (no trailing NUL included)
        let bytes: [UInt8] = buf.prefix(count).map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func tokenize(text: String, addBos: Bool) -> [Token] {
        let utf8Count = text.utf8.count
        let initial = utf8Count + (addBos ? 1 : 0) + 1

        func tokenize(into capacity: Int) -> (Int32, [Token]) {
            var buffer = [Token](repeating: 0, count: max(1, capacity))
            let count = llama_tokenize(vocab, text, Int32(utf8Count), &buffer, Int32(buffer.count), addBos, false)
            return (count, buffer)
        }

        var (count, buffer) = tokenize(into: initial)
        if count < 0 {
            (count, buffer) = tokenize(into: Int(-count))
        }
        let resolved = max(0, Int(count))
        return Array(buffer.prefix(resolved))
    }

    func clear() {
        tokens.removeAll()
        let memory = llama_get_memory(context)
        llama_memory_clear(memory, true)
        maxTokenCountOverride = nil
        promptTokenCount = 0
    }

    deinit {
        llama_batch_free(batch)
        llama_free(context)
        llama_model_free(model)
        llama_backend_free()
    }

    nonisolated(unsafe) private static var didInstallLogCallback = false
    private static let logCallback: ggml_log_callback = { _, text, _ in
        guard let text else { return }
        let message = String(cString: text).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !message.isEmpty else { return }
        fputs("[LLM][llama] \(message)\n", stderr)
    }

    private static func shouldEnableLogging() -> Bool {
        let env = ProcessInfo.processInfo.environment
        return env["LLM_LLAMA_LOG"] == "1" || env["LLM_BENCH_TRANSLATE"] == "1"
    }

    private static func installLogCallbackIfNeeded() {
        guard !didInstallLogCallback else { return }
        llama_log_set(logCallback, nil)
        didInstallLogCallback = true
    }

    private static func resolveSeed(_ seed: Int) -> UInt32 {
        if seed < 0 {
            return UInt32(LLAMA_DEFAULT_SEED)
        }
        return UInt32(clamping: seed)
    }
}
