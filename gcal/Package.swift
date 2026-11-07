// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "CalendarReminder",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "CalendarReminder", path: "Sources/CalendarReminder")
    ]
)
