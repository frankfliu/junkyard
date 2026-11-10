import AppKit
import Foundation
import ServiceManagement

@MainActor
final class AppModel: ObservableObject {
    @Published private(set) var accounts: [String] = Prefs.accounts
    @Published private(set) var meetings: [Meeting] = []
    @Published private(set) var errors: [String: String] = [:]
    @Published private(set) var lastRefresh: Date?
    @Published private(set) var signingIn = false
    @Published var signInStatus: String?
    @Published private(set) var loginError: String?
    @Published var launchAtLogin = SMAppService.mainApp.status == .enabled {
        didSet { setLaunchAtLogin(launchAtLogin) }
    }

    /// Called whenever the menu bar title may need updating.
    var onChange: (() -> Void)?

    private let alerts = AlertController()
    private var tokens: [String: (token: String, expiry: Date)] = [:]
    // Persisted so restarting the app doesn't re-alert meetings already handled.
    private var fired = Prefs.firedAlerts.filter { $0.value > Date() } {
        didSet { Prefs.firedAlerts = fired.filter { $0.value > Date() } }
    }
    private var snoozed = Prefs.snoozedAlerts.filter { $0.value > Date().addingTimeInterval(-86400) } {
        didSet { Prefs.snoozedAlerts = snoozed }
    }
    private var refreshing = false
    private var timers: [Timer] = []

    private static let refreshInterval: TimeInterval = 5 * 60
    /// Still pop an alert if we notice a meeting up to this long after it started (e.g. after wake).
    private static let lateGrace: TimeInterval = 5 * 60

    func start() {
        schedule(every: 5) { $0.tick() }
        schedule(every: Self.refreshInterval) { model in Task { await model.refresh() } }
        NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didWakeNotification, object: nil, queue: .main
        ) { [weak self] _ in
            // Give the network a moment to come back after wake.
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 10_000_000_000)
                await self?.refresh()
            }
        }
        Task { await refresh() }
    }

    private func schedule(every interval: TimeInterval, _ body: @escaping (AppModel) -> Void) {
        let timer = Timer(timeInterval: interval, repeats: true) { [weak self] _ in
            Task { @MainActor in if let self { body(self) } }
        }
        RunLoop.main.add(timer, forMode: .common)  // keep firing while the menu is open
        timers.append(timer)
    }

    // MARK: - Fetching

    func refresh() async {
        guard !refreshing else { return }
        refreshing = true
        defer { refreshing = false }

        guard let client = Prefs.client else {
            errors = ["": "Set up the Google OAuth client in Settings"]
            onChange?()
            return
        }

        var all: [Meeting] = []
        var errs: [String: String] = [:]
        for email in accounts {
            do {
                let token = try await accessToken(email, client)
                all += try await CalendarAPI.upcoming(token: token, account: email,
                                                      allCalendars: Prefs.includeAllCalendars)
            } catch {
                tokens[email] = nil
                errs[email] = error.localizedDescription
                all += meetings.filter { $0.account == email }  // keep alerting from last good data
            }
        }

        // The same meeting can show up on several accounts/calendars; keep one.
        var seen = Set<String>()
        meetings = all.sorted { $0.start < $1.start }.filter { seen.insert($0.key).inserted }
        errors = errs
        lastRefresh = Date()
        tick()
    }

    private func accessToken(_ email: String, _ client: OAuthClient) async throws -> String {
        if let cached = tokens[email], cached.expiry > Date().addingTimeInterval(60) {
            return cached.token
        }
        guard let refreshToken = TokenStore.get(email) else {
            throw AppError(message: "No saved sign-in. Re-authenticate this account.")
        }
        let t = try await GoogleAuth.refresh(client: client, refreshToken: refreshToken)
        tokens[email] = (t.access_token, Date().addingTimeInterval(TimeInterval(t.expires_in)))
        return t.access_token
    }

    // MARK: - Alerts

    /// When set, alert popups are suppressed until this time.
    var pausedUntil: Date? {
        if let until = Prefs.pausedUntil, until > Date() { return until }
        return nil
    }

    func pause(hours: Int) {
        Prefs.pausedUntil = Date().addingTimeInterval(TimeInterval(hours * 3600))
        onChange?()
    }

    func resume() {
        Prefs.pausedUntil = nil
        tick()
    }

    func tick() {
        let now = Date()
        if let until = Prefs.pausedUntil, until <= now {
            Prefs.pausedUntil = nil
        }
        guard pausedUntil == nil else {
            onChange?()
            return
        }
        let lead = TimeInterval(Prefs.leadMinutes * 60)
        for m in meetings where fired[m.key] == nil {
            let snoozeUntil = snoozed[m.key]
            let alertAt = snoozeUntil ?? m.start.addingTimeInterval(-lead)
            let deadline = snoozeUntil != nil ? m.end : m.start.addingTimeInterval(Self.lateGrace)
            if now >= alertAt && now < deadline {
                fired[m.key] = m.end
                snoozed[m.key] = nil
                alerts.show(m) { [weak self] until in self?.snooze(m, until: until) }
            }
        }
        onChange?()
    }

    private func snooze(_ m: Meeting, until: Date) {
        fired[m.key] = nil
        snoozed[m.key] = until
    }

    func joined(_ m: Meeting) {
        alerts.joined(m)
    }

    func testAlert() {
        let start = Date().addingTimeInterval(TimeInterval(Prefs.leadMinutes * 60))
        let m = Meeting(id: "test-\(UUID())", title: "Test meeting", start: start,
                        end: start.addingTimeInterval(1800), account: accounts.first ?? "you@example.com",
                        calendar: "Test", joinURL: URL(string: "https://meet.google.com/"), eventURL: nil)
        alerts.show(m) { _ in }
    }

    private func setLaunchAtLogin(_ on: Bool) {
        do {
            if on {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
            loginError = nil
        } catch {
            loginError = error.localizedDescription
        }
    }

    // MARK: - Accounts

    func addAccount() async {
        guard let client = Prefs.client else {
            signInStatus = "Enter the OAuth client ID first."
            return
        }
        signingIn = true
        signInStatus = "Complete sign-in in your browser…"
        defer { signingIn = false }
        do {
            let (email, refreshToken) = try await GoogleAuth.signIn(client: client)
            TokenStore.set(refreshToken, for: email)
            tokens[email] = nil
            if !accounts.contains(email) {
                accounts.append(email)
                Prefs.accounts = accounts
            }
            signInStatus = "Signed in as \(email)"
            NSApp.activate(ignoringOtherApps: true)
            await refresh()
        } catch {
            signInStatus = "Sign-in failed: \(error.localizedDescription)"
        }
    }

    func removeAccount(_ email: String) {
        if let token = TokenStore.get(email) {
            Task { await GoogleAuth.revoke(token) }
        }
        TokenStore.delete(email)
        tokens[email] = nil
        accounts.removeAll { $0 == email }
        Prefs.accounts = accounts
        meetings.removeAll { $0.account == email }
        errors[email] = nil
        onChange?()
    }
}
