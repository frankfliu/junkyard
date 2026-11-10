import Foundation

struct Meeting: Identifiable, Hashable {
    let id: String
    let title: String
    let start: Date
    let end: Date
    let account: String
    let calendar: String
    let joinURL: URL?
    let eventURL: URL?

    /// Identifies one occurrence; changes if the meeting is rescheduled so it alerts again.
    var key: String { "\(id)|\(start.timeIntervalSince1970)" }
}

private struct CalendarList: Decodable {
    struct Entry: Decodable {
        let id: String
        let summary: String?
        let summaryOverride: String?
        let selected: Bool?
        let primary: Bool?
    }
    let items: [Entry]
}

private struct EventList: Decodable {
    let items: [Event]
}

private struct Event: Decodable {
    struct Time: Decodable {
        let dateTime: String?
    }
    struct Attendee: Decodable {
        let isSelf: Bool?
        let responseStatus: String?
        enum CodingKeys: String, CodingKey {
            case isSelf = "self"
            case responseStatus
        }
    }
    struct Conference: Decodable {
        struct EntryPoint: Decodable {
            let entryPointType: String?
            let uri: String?
        }
        let entryPoints: [EntryPoint]?
    }

    let id: String
    let iCalUID: String?
    let status: String?
    let summary: String?
    let location: String?
    let description: String?
    let htmlLink: String?
    let hangoutLink: String?
    let eventType: String?
    let start: Time
    let end: Time
    let attendees: [Attendee]?
    let conferenceData: Conference?
}

enum CalendarAPI {
    private static let base = "https://www.googleapis.com/calendar/v3/"

    /// Timed events from now through the end of tomorrow, across the account's calendars.
    static func upcoming(token: String, account: String, allCalendars: Bool) async throws -> [Meeting] {
        let list = try await HTTP.json(CalendarList.self,
                                       HTTP.get(URL(string: base + "users/me/calendarList?minAccessRole=reader")!, token: token))
        let calendars = list.items.filter { cal in
            cal.primary == true || (allCalendars && cal.selected == true)
        }

        let now = Date()
        let endOfTomorrow = Calendar.current.date(byAdding: .day, value: 2, to: Calendar.current.startOfDay(for: now))!
        let iso = ISO8601DateFormatter()
        var meetings: [Meeting] = []
        for cal in calendars {
            var c = URLComponents(string: base + "calendars/\(encodePath(cal.id))/events")!
            c.queryItems = [
                URLQueryItem(name: "timeMin", value: iso.string(from: now)),
                URLQueryItem(name: "timeMax", value: iso.string(from: endOfTomorrow)),
                URLQueryItem(name: "singleEvents", value: "true"),
                URLQueryItem(name: "orderBy", value: "startTime"),
                URLQueryItem(name: "maxResults", value: "250"),
            ]
            let events: EventList
            do {
                events = try await HTTP.json(EventList.self, HTTP.get(c.url!, token: token))
            } catch where cal.primary != true {
                continue  // a broken secondary calendar shouldn't hide the rest
            }
            let name = cal.summaryOverride ?? cal.summary ?? cal.id
            meetings += events.items.compactMap { meeting(from: $0, account: account, calendar: name) }
        }
        return meetings
    }

    private static func meeting(from e: Event, account: String, calendar: String) -> Meeting? {
        guard e.status != "cancelled",
              let start = e.start.dateTime.flatMap(parseDate),
              let end = e.end.dateTime.flatMap(parseDate)  // all-day events have no dateTime
        else { return nil }
        if ["workingLocation", "outOfOffice"].contains(e.eventType ?? "") { return nil }
        // Events with guests must be accepted by me; events without guests are my own.
        if let attendees = e.attendees, !attendees.isEmpty,
           attendees.first(where: { $0.isSelf == true })?.responseStatus != "accepted" {
            return nil
        }

        return Meeting(
            id: e.iCalUID ?? e.id,
            title: e.summary ?? "(No title)",
            start: start,
            end: end,
            account: account,
            calendar: calendar,
            joinURL: joinURL(e).map { withAuthUser($0, account) },
            eventURL: e.htmlLink.flatMap(URL.init(string:)).map { withAuthUser($0, account) }
        )
    }

    /// Google links open in whichever account the browser defaults to; `authuser` picks the right one.
    private static func withAuthUser(_ url: URL, _ account: String) -> URL {
        guard let host = url.host?.lowercased(), host == "google.com" || host.hasSuffix(".google.com"),
              var comps = URLComponents(url: url, resolvingAgainstBaseURL: false)
        else { return url }
        var items = (comps.queryItems ?? []).filter { $0.name != "authuser" }
        items.append(URLQueryItem(name: "authuser", value: account))
        comps.queryItems = items
        return comps.url ?? url
    }

    private static let linkPattern = try! NSRegularExpression(
        pattern: #"https://[^\s"'<>]*(zoom\.us|teams\.microsoft\.com|teams\.live\.com|meet\.google\.com|webex\.com|chime\.aws)/[^\s"'<>]*"#)

    private static func joinURL(_ e: Event) -> URL? {
        if let link = e.hangoutLink, let url = URL(string: link) { return url }
        if let uri = e.conferenceData?.entryPoints?.first(where: { $0.entryPointType == "video" })?.uri,
           let url = URL(string: uri) {
            return url
        }
        for text in [e.location, e.description].compactMap({ $0 }) {
            let range = NSRange(text.startIndex..., in: text)
            if let m = linkPattern.firstMatch(in: text, range: range), let r = Range(m.range, in: text) {
                return URL(string: String(text[r]))
            }
        }
        return nil
    }

    private static let isoPlain = ISO8601DateFormatter()
    private static let isoFractional: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    private static func parseDate(_ s: String) -> Date? {
        isoPlain.date(from: s) ?? isoFractional.date(from: s)
    }

    private static func encodePath(_ s: String) -> String {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~@")
        return s.addingPercentEncoding(withAllowedCharacters: allowed) ?? s
    }
}
