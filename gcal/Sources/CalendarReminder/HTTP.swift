import Foundation

enum HTTP {
    static func json<T: Decodable>(_ type: T.Type, _ request: URLRequest) async throws -> T {
        let (data, response) = try await URLSession.shared.data(for: request)
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            if body.contains("invalid_grant") {
                throw AppError(message: "Sign-in expired or revoked. Re-authenticate this account.")
            }
            throw AppError(message: "HTTP \(code): \(body.prefix(300))")
        }
        return try JSONDecoder().decode(T.self, from: data)
    }

    static func get(_ url: URL, token: String) -> URLRequest {
        var req = URLRequest(url: url)
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        return req
    }

    static func form(_ url: URL, _ params: [String: String]) -> URLRequest {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        req.httpBody = params.map { "\(escape($0.key))=\(escape($0.value))" }
            .joined(separator: "&").data(using: .utf8)
        return req
    }

    static func escape(_ s: String) -> String {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        return s.addingPercentEncoding(withAllowedCharacters: allowed) ?? s
    }
}
