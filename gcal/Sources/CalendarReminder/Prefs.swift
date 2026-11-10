import Foundation

struct OAuthClient {
    let id: String
    let secret: String
}

struct AppError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

enum Prefs {
    private static var d: UserDefaults { .standard }

    static func registerDefaults() {
        d.register(defaults: [
            "leadMinutes": 2,
            "soundName": "Glass",
            "repeatSound": true,
            "includeAllCalendars": true,
        ])
    }

    /// The client entered in Settings, falling back to the one bundled at build time.
    static var client: OAuthClient? {
        let id = (d.string(forKey: "clientID") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let secret = (d.string(forKey: "clientSecret") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return id.isEmpty ? bundledClient : OAuthClient(id: id, secret: secret)
    }

    static let bundledClient: OAuthClient? = {
        guard let url = Bundle.main.url(forResource: "client_secret", withExtension: "json"),
              let data = try? Data(contentsOf: url)
        else { return nil }
        return parseClientJSON(data)
    }()

    /// Parses a client JSON downloaded from Google Cloud Console.
    static func parseClientJSON(_ data: Data) -> OAuthClient? {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let creds = (root["installed"] ?? root["web"]) as? [String: Any],
              let id = creds["client_id"] as? String
        else { return nil }
        return OAuthClient(id: id, secret: creds["client_secret"] as? String ?? "")
    }

    static var leadMinutes: Int { d.integer(forKey: "leadMinutes") }
    static var soundName: String { d.string(forKey: "soundName") ?? "Glass" }
    static var repeatSound: Bool { d.bool(forKey: "repeatSound") }
    static var includeAllCalendars: Bool { d.bool(forKey: "includeAllCalendars") }

    static var pausedUntil: Date? {
        get { d.object(forKey: "pausedUntil") as? Date }
        set { d.set(newValue, forKey: "pausedUntil") }
    }

    /// Meetings already alerted (key → meeting end), so a restart doesn't pop them up again.
    static var firedAlerts: [String: Date] {
        get { d.dictionary(forKey: "firedAlerts") as? [String: Date] ?? [:] }
        set { d.set(newValue, forKey: "firedAlerts") }
    }

    /// Snoozed meetings (key → time to alert again).
    static var snoozedAlerts: [String: Date] {
        get { d.dictionary(forKey: "snoozedAlerts") as? [String: Date] ?? [:] }
        set { d.set(newValue, forKey: "snoozedAlerts") }
    }

    static var accounts: [String] {
        get { d.stringArray(forKey: "accounts") ?? [] }
        set { d.set(newValue, forKey: "accounts") }
    }

    static var systemSounds: [String] {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: "/System/Library/Sounds")) ?? []
        return files.map { ($0 as NSString).deletingPathExtension }.sorted()
    }
}
