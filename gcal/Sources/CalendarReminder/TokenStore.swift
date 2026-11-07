import Foundation

/// Stores each account's OAuth refresh token in a file readable only by the current user.
///
/// We avoid the Keychain because ad-hoc signed builds get a new identity on every rebuild,
/// which makes macOS prompt for the keychain password each time.
enum TokenStore {
    private static var fileURL: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CalendarReminder", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true,
                                                 attributes: [.posixPermissions: 0o700])
        return dir.appendingPathComponent("tokens.json")
    }

    private static func load() -> [String: String] {
        guard let data = try? Data(contentsOf: fileURL),
              let tokens = try? JSONDecoder().decode([String: String].self, from: data)
        else { return [:] }
        return tokens
    }

    private static func save(_ tokens: [String: String]) {
        guard let data = try? JSONEncoder().encode(tokens) else { return }
        let url = fileURL
        try? data.write(to: url, options: .atomic)
        try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    static func set(_ value: String, for account: String) {
        var tokens = load()
        tokens[account] = value
        save(tokens)
    }

    static func get(_ account: String) -> String? {
        load()[account]
    }

    static func delete(_ account: String) {
        var tokens = load()
        tokens[account] = nil
        save(tokens)
    }
}
