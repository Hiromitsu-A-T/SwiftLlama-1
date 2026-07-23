// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "SwiftLlama",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .watchOS(.v11),
        .tvOS(.v18),
        .visionOS(.v2)
    ],
    products: [
        .library(name: "SwiftLlama", targets: ["SwiftLlama"]),
    ],
    dependencies: [],
    targets: [
        .target(name: "SwiftLlama", 
                dependencies: [
                    "LlamaFramework"
                ]),
        .testTarget(name: "SwiftLlamaTests", dependencies: ["SwiftLlama"]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b10087/llama-b10087-xcframework.zip",
            checksum: "ea28e09d542f025686aec47df1b3312c2510f7ed884c476743ab76c31c924005"
        )
    ]
)
