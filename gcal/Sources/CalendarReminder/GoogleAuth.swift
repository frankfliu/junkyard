import AppKit
import CryptoKit
import Foundation
import Network

struct TokenResponse: Decodable {
    let access_token: String
    let expires_in: Int
    let refresh_token: String?
    let id_token: String?
}

/// Google OAuth for installed apps: browser sign-in with a loopback redirect and PKCE.
enum GoogleAuth {
    static let scopes = "openid email https://www.googleapis.com/auth/calendar.readonly"
    private static let authURL = "https://accounts.google.com/o/oauth2/v2/auth"
    private static let tokenURL = URL(string: "https://oauth2.googleapis.com/token")!
    private static let revokeURL = URL(string: "https://oauth2.googleapis.com/revoke")!

    static func signIn(client: OAuthClient) async throws -> (email: String, refreshToken: String) {
        let server = try LoopbackServer()
        defer { server.stop() }
        let port = try await server.start()
        let redirect = "http://127.0.0.1:\(port)"
        let verifier = randomString(32)
        let challenge = base64url(Data(SHA256.hash(data: Data(verifier.utf8))))
        let state = randomString(16)

        var c = URLComponents(string: authURL)!
        c.queryItems = [
            URLQueryItem(name: "client_id", value: client.id),
            URLQueryItem(name: "redirect_uri", value: redirect),
            URLQueryItem(name: "response_type", value: "code"),
            URLQueryItem(name: "scope", value: scopes),
            URLQueryItem(name: "code_challenge", value: challenge),
            URLQueryItem(name: "code_challenge_method", value: "S256"),
            URLQueryItem(name: "access_type", value: "offline"),
            URLQueryItem(name: "prompt", value: "consent select_account"),
            URLQueryItem(name: "state", value: state),
        ]
        NSWorkspace.shared.open(c.url!)

        let params = try await server.waitForCallback(timeout: 300)
        if let err = params["error"] {
            throw AppError(message: "Google returned: \(err)")
        }
        guard params["state"] == state, let code = params["code"] else {
            throw AppError(message: "Invalid OAuth callback")
        }

        let token = try await HTTP.json(TokenResponse.self, HTTP.form(tokenURL, [
            "code": code,
            "client_id": client.id,
            "client_secret": client.secret,
            "redirect_uri": redirect,
            "grant_type": "authorization_code",
            "code_verifier": verifier,
        ]))
        guard let refresh = token.refresh_token else {
            throw AppError(message: "Google did not return a refresh token")
        }
        guard let idToken = token.id_token, let email = email(fromIDToken: idToken) else {
            throw AppError(message: "Could not determine account email")
        }
        return (email, refresh)
    }

    static func refresh(client: OAuthClient, refreshToken: String) async throws -> TokenResponse {
        try await HTTP.json(TokenResponse.self, HTTP.form(tokenURL, [
            "client_id": client.id,
            "client_secret": client.secret,
            "refresh_token": refreshToken,
            "grant_type": "refresh_token",
        ]))
    }

    static func revoke(_ token: String) async {
        _ = try? await URLSession.shared.data(for: HTTP.form(revokeURL, ["token": token]))
    }

    private static func email(fromIDToken jwt: String) -> String? {
        let parts = jwt.split(separator: ".")
        guard parts.count >= 2 else { return nil }
        var b64 = parts[1].replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
        b64 += String(repeating: "=", count: (4 - b64.count % 4) % 4)
        guard let data = Data(base64Encoded: b64),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return obj["email"] as? String
    }

    private static func randomString(_ bytes: Int) -> String {
        var data = Data(count: bytes)
        _ = data.withUnsafeMutableBytes { SecRandomCopyBytes(kSecRandomDefault, bytes, $0.baseAddress!) }
        return base64url(data)
    }

    private static func base64url(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
}

/// Minimal HTTP server on 127.0.0.1 that captures the OAuth redirect.
/// All mutable state is only touched on `queue`.
final class LoopbackServer: @unchecked Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "CalendarReminder.loopback")
    private var continuation: CheckedContinuation<[String: String], Error>?
    private var pending: Result<[String: String], Error>?
    private var done = false
    private var started = false

    init() throws {
        let params = NWParameters.tcp
        params.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        listener = try NWListener(using: params)
    }

    func start() async throws -> UInt16 {
        try await withCheckedThrowingContinuation { cont in
            listener.stateUpdateHandler = { [self] state in
                guard !started else { return }
                switch state {
                case .ready:
                    started = true
                    cont.resume(returning: listener.port?.rawValue ?? 0)
                case .failed(let error):
                    started = true
                    cont.resume(throwing: error)
                default:
                    break
                }
            }
            listener.newConnectionHandler = { [weak self] conn in self?.handle(conn) }
            listener.start(queue: queue)
        }
    }

    func stop() {
        listener.cancel()
    }

    func waitForCallback(timeout: TimeInterval) async throws -> [String: String] {
        queue.asyncAfter(deadline: .now() + timeout) { [weak self] in
            self?.deliver(.failure(AppError(message: "Timed out waiting for browser sign-in")))
        }
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                if let pending = self.pending {
                    cont.resume(with: pending)
                } else {
                    self.continuation = cont
                }
            }
        }
    }

    // Runs on `queue`.
    private func deliver(_ result: Result<[String: String], Error>) {
        guard !done else { return }
        done = true
        if let cont = continuation {
            continuation = nil
            cont.resume(with: result)
        } else {
            pending = result
        }
    }

    private func handle(_ conn: NWConnection) {
        conn.start(queue: queue)
        conn.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, _, _ in
            guard let self else { return }
            let request = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
            let path = request.split(separator: " ").dropFirst().first.map(String.init) ?? "/"
            let items = URLComponents(string: "http://localhost\(path)")?.queryItems ?? []
            var params: [String: String] = [:]
            for item in items { params[item.name] = item.value ?? "" }

            guard params["code"] != nil || params["error"] != nil else {
                self.respond(conn, status: "404 Not Found", body: "")
                return
            }
            let ok = params["code"] != nil
            let body = """
            <html><body style="font-family:-apple-system;text-align:center;margin-top:80px">
            <h2>\(ok ? "Signed in to Calendar Reminder" : "Sign-in failed")</h2>
            <p>You can close this tab and return to the app.</p></body></html>
            """
            self.respond(conn, status: "200 OK", body: body)
            self.deliver(.success(params))
        }
    }

    private func respond(_ conn: NWConnection, status: String, body: String) {
        let bytes = Data(body.utf8)
        let head = "HTTP/1.1 \(status)\r\nContent-Type: text/html; charset=utf-8\r\n"
            + "Content-Length: \(bytes.count)\r\nConnection: close\r\n\r\n"
        conn.send(content: Data(head.utf8) + bytes, completion: .contentProcessed { _ in conn.cancel() })
    }
}
