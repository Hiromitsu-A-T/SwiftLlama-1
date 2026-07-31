import XCTest
@testable import SwiftLlama

final class SwiftLlamaTests: XCTestCase {
    func testBackendLifecycleReinitializesAfterLastRelease() {
        var initializationCount = 0
        var shutdownCount = 0
        let lifecycle = LlamaBackendLifecycle(
            initialize: { initializationCount += 1 },
            shutdown: { shutdownCount += 1 }
        )

        lifecycle.retain()
        lifecycle.retain()
        XCTAssertEqual(initializationCount, 1)
        XCTAssertEqual(
            lifecycle.snapshot(),
            .init(referenceCount: 2, isInitialized: true)
        )

        lifecycle.release()
        XCTAssertEqual(shutdownCount, 0)
        lifecycle.release()
        XCTAssertEqual(shutdownCount, 1)
        XCTAssertEqual(
            lifecycle.snapshot(),
            .init(referenceCount: 0, isInitialized: false)
        )

        lifecycle.retain()
        XCTAssertEqual(initializationCount, 2)
        lifecycle.release()
        XCTAssertEqual(shutdownCount, 2)
    }

    func testBackendLifecycleIgnoresUnbalancedRelease() {
        var shutdownCount = 0
        let lifecycle = LlamaBackendLifecycle(
            initialize: {},
            shutdown: { shutdownCount += 1 }
        )

        lifecycle.release()

        XCTAssertEqual(shutdownCount, 0)
        XCTAssertEqual(
            lifecycle.snapshot(),
            .init(referenceCount: 0, isInitialized: false)
        )
    }

    func testBackendLifecycleDoesNotRetainWhileShutdownIsInFlight() async {
        let shutdownStarted = DispatchSemaphore(value: 0)
        let allowShutdownToFinish = DispatchSemaphore(value: 0)
        let retainFinished = DispatchSemaphore(value: 0)
        let lifecycle = LlamaBackendLifecycle(
            initialize: {},
            shutdown: {
                shutdownStarted.signal()
                allowShutdownToFinish.wait()
            }
        )
        lifecycle.retain()

        DispatchQueue.global().async {
            lifecycle.release()
        }
        XCTAssertEqual(shutdownStarted.wait(timeout: .now() + 1), .success)

        DispatchQueue.global().async {
            lifecycle.retain()
            retainFinished.signal()
        }
        XCTAssertEqual(
            retainFinished.wait(timeout: .now() + 0.1),
            .timedOut,
            "retain must wait until llama_backend_free has completed"
        )

        allowShutdownToFinish.signal()
        XCTAssertEqual(retainFinished.wait(timeout: .now() + 1), .success)
        XCTAssertEqual(
            lifecycle.snapshot(),
            .init(referenceCount: 1, isInitialized: true)
        )
        allowShutdownToFinish.signal()
        lifecycle.release()
    }

}
