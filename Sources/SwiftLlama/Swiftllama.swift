import Foundation
import llama
import Combine

public class SwiftLlama {
    private let model: LlamaModel
    private let configuration: Configuration
    private var contentStarted = false
    private var sessionSupport = false {
        didSet {
            if !sessionSupport {
                session = nil
            }
        }
    }

    private var session: Session?
    private var generatedTokenCache = ""

    var maxLengthOfStopToken: Int {
        configuration.stopTokens.map { $0.count }.max() ?? 0
    }

    public init(
        modelPath: String,
        modelConfiguration: Configuration = .init()
    ) throws {
        self.model = try LlamaModel(path: modelPath, configuration: modelConfiguration)
        self.configuration = modelConfiguration
    }

    private func prepare(sessionSupport: Bool, for prompt: Prompt) -> Prompt {
        contentStarted = false
        generatedTokenCache = ""
        self.sessionSupport = sessionSupport
        if sessionSupport {
            if session == nil {
                session = Session(lastPrompt: prompt)
            } else {
                session?.lastPrompt = prompt
            }
            return session?.sessionPrompt ?? prompt
        } else {
            return prompt
        }
    }

    private func response(for prompt: Prompt,
                          maxOutputTokens: Int?,
                          output: (String) -> Void,
                          finish: (Error?) -> Void) {
        func flushBufferedOutput() {
            configuration.stopTokens.forEach {
                generatedTokenCache = generatedTokenCache.replacingOccurrences(of: $0, with: "")
            }
            if !generatedTokenCache.isEmpty {
                output(generatedTokenCache)
            }
            generatedTokenCache = ""
        }
        defer { model.clear() }
        do {
            try model.start(for: prompt, maxOutputTokens: maxOutputTokens)
            while model.shouldContinue {
                if Task.isCancelled {
                    throw CancellationError()
                }
                var delta = try model.continue()
                if contentStarted { // remove the prefix empty spaces
                    if needToStop(after: delta, output: output) {
                        break
                    }
                } else {
                    delta = delta.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !delta.isEmpty {
                        contentStarted = true
                        if needToStop(after: delta, output: output) {
                            break
                        }
                    }
                }
            }
            flushBufferedOutput()
            finish(nil)
        } catch {
            flushBufferedOutput()
            finish(error)
        }
    }

    /// Handling logic of StopToken
    private func needToStop(after delta: String, output: (String) -> Void) -> Bool {
        // If no stop tokens, just stream through
        guard maxLengthOfStopToken > 0 else {
            output(delta)
            return false
        }

        generatedTokenCache += delta

        // 1) If any stop token appears, cut output before it and stop
        if let stopRange = configuration.stopTokens
            .compactMap({ generatedTokenCache.range(of: $0) })
            .min(by: { $0.lowerBound < $1.lowerBound }) { // earliest occurrence
            let before = String(generatedTokenCache[..<stopRange.lowerBound])
            if !before.isEmpty { output(before) }
            generatedTokenCache.removeAll(keepingCapacity: false)
            return true
        }

        // 2) Stream everything except a small tail so split stop tokens are caught next time
        let tail = max(maxLengthOfStopToken - 1, 0)
        if generatedTokenCache.count > tail {
            let cut = generatedTokenCache.index(generatedTokenCache.endIndex, offsetBy: -tail)
            let safe = String(generatedTokenCache[..<cut])
            if !safe.isEmpty { output(safe) }
            generatedTokenCache.removeFirst(safe.count)
        }

        return false
    }

    @SwiftLlamaActor
    public func start(for prompt: Prompt,
                      sessionSupport: Bool = false,
                      maxOutputTokens: Int? = nil) -> AsyncThrowingStream<String, Error> {
        let sessionPrompt = prepare(sessionSupport: sessionSupport, for: prompt)
        return .init { continuation in
            let task = Task { @SwiftLlamaActor in
                response(for: sessionPrompt, maxOutputTokens: maxOutputTokens) { [weak self] delta in
                    continuation.yield(delta)
                    self?.session?.response(delta: delta)
                } finish: { [weak self] error in
                    if let error {
                        continuation.finish(throwing: error)
                        self?.session = nil
                    } else {
                        continuation.finish()
                        self?.session?.endResponse()
                    }
                }
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    @SwiftLlamaActor
    public func start(for prompt: Prompt,
                      sessionSupport: Bool = false,
                      maxOutputTokens: Int? = nil) -> AnyPublisher<String, Error> {
        let sessionPrompt = prepare(sessionSupport: sessionSupport, for: prompt)
        let subject = PassthroughSubject<String, Error>()
        let task = Task { @SwiftLlamaActor in
            response(for: sessionPrompt, maxOutputTokens: maxOutputTokens) { delta in
                subject.send(delta)
                session?.response(delta: delta)
            } finish: { error in
                if let error {
                    subject.send(completion: .failure(error))
                    session = nil
                } else {
                    subject.send(completion: .finished)
                    session?.endResponse()
                }
            }
        }
        return subject
            .handleEvents(receiveCancel: { task.cancel() })
            .eraseToAnyPublisher()
    }

    @SwiftLlamaActor
    public func start(for prompt: Prompt,
                      sessionSupport: Bool = false,
                      maxOutputTokens: Int? = nil) async throws -> String {
        var result = ""
        for try await value in start(for: prompt, sessionSupport: sessionSupport, maxOutputTokens: maxOutputTokens) {
            result += value
        }
        return result
    }
}
