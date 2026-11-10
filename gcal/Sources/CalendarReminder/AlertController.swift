import AppKit
import SwiftUI

/// Shows a floating window per meeting and plays the alert sound while any are open.
@MainActor
final class AlertController: NSObject, NSWindowDelegate {
    private var windows: [String: NSWindow] = [:]
    /// Alerts that ring; meetings without a join link alert silently.
    private var ringing: Set<String> = []
    private var soundTimer: Timer?
    private var soundStarted = Date.distantPast
    /// End of the latest meeting joined through this app; stay quiet until then.
    private var inMeetingUntil = Date.distantPast

    private static let soundRepeatInterval: TimeInterval = 3
    private static let soundMaxDuration: TimeInterval = 60
    private static let silentAutoDismiss: TimeInterval = 5 * 60

    func show(_ meeting: Meeting, onSnooze: @escaping (Date) -> Void) {
        guard windows[meeting.key] == nil else { return }
        let key = meeting.key

        let view = AlertView(
            meeting: meeting,
            join: { [weak self] url in
                NSWorkspace.shared.open(url)
                self?.joined(meeting)
                self?.close(key)
            },
            openEvent: { [weak self] url in
                NSWorkspace.shared.open(url)
                self?.close(key)
            },
            snooze: { [weak self] until in
                onSnooze(until)
                self?.close(key)
            },
            dismiss: { [weak self] in self?.close(key) }
        )
        let hosting = NSHostingView(rootView: view)
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: hosting.fittingSize),
                              styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.title = "Upcoming meeting"
        window.contentView = hosting
        window.identifier = NSUserInterfaceItemIdentifier(key)
        window.isReleasedWhenClosed = false
        window.level = .floating
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.delegate = self

        if let screen = NSScreen.main?.visibleFrame {
            let offset = CGFloat(windows.count) * 30
            window.setFrameTopLeftPoint(NSPoint(x: screen.maxX - window.frame.width - 20 - offset,
                                                y: screen.maxY - 20 - offset))
        }
        windows[key] = window
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        if meeting.joinURL != nil {
            ringing.insert(key)
            startSound()
        } else {
            DispatchQueue.main.asyncAfter(deadline: .now() + Self.silentAutoDismiss) { [weak self, weak window] in
                // Only close this exact window, not a later alert for the same meeting (e.g. after snooze).
                if let window, self?.windows[key] === window { window.close() }
            }
        }
    }

    /// Records that the user joined `meeting`, silencing alerts until it's scheduled to end.
    func joined(_ meeting: Meeting) {
        inMeetingUntil = max(inMeetingUntil, meeting.end)
        stopSound()
    }

    private func close(_ key: String) {
        windows[key]?.close()
    }

    func windowWillClose(_ notification: Notification) {
        guard let key = (notification.object as? NSWindow)?.identifier?.rawValue else { return }
        windows[key] = nil
        ringing.remove(key)
        if ringing.isEmpty { stopSound() }
    }

    private func startSound() {
        guard soundTimer == nil else { return }
        soundStarted = Date()
        playOnce()
        guard Prefs.repeatSound else { return }
        soundTimer = Timer.scheduledTimer(withTimeInterval: Self.soundRepeatInterval, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                if Date().timeIntervalSince(self.soundStarted) > Self.soundMaxDuration {
                    self.stopSound()
                } else {
                    self.playOnce()
                }
            }
        }
    }

    private func stopSound() {
        soundTimer?.invalidate()
        soundTimer = nil
    }

    private func playOnce() {
        // Don't ring while already on a call (joined this meeting, or the previous one ran over).
        guard Prefs.soundName != "None", Date() >= inMeetingUntil, !Microphone.inUse else { return }
        NSSound(named: NSSound.Name(Prefs.soundName))?.play()
    }
}

private struct AlertView: View {
    let meeting: Meeting
    let join: (URL) -> Void
    let openEvent: (URL) -> Void
    let snooze: (Date) -> Void
    let dismiss: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "bell.badge.fill")
                    .font(.system(size: 28))
                    .foregroundStyle(.orange)
                VStack(alignment: .leading, spacing: 4) {
                    Text(meeting.title)
                        .font(.title2.bold())
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                    TimelineView(.periodic(from: .now, by: 1)) { context in
                        Text(countdown(now: context.date))
                            .font(.headline)
                            .foregroundStyle(meeting.start <= context.date ? .red : .primary)
                    }
                    Text("\(timeRange) · \(meeting.calendar)")
                        .foregroundStyle(.secondary)
                    Text(meeting.account)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            HStack {
                if let url = meeting.joinURL {
                    Button("Join") { join(url) }
                        .keyboardShortcut(.defaultAction)
                }
                if let url = meeting.eventURL {
                    Button("Open Event") { openEvent(url) }
                }
                Spacer()
                Menu("Snooze") {
                    Button("1 minute") { snooze(Date().addingTimeInterval(60)) }
                    Button("5 minutes") { snooze(Date().addingTimeInterval(300)) }
                    if meeting.start > Date() {
                        Button("Until start time") { snooze(meeting.start) }
                    }
                }
                .fixedSize()
                Button("Dismiss") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }
        }
        .padding(20)
        .frame(width: 440)
    }

    private var timeRange: String {
        let f = DateFormatter()
        f.dateStyle = .none
        f.timeStyle = .short
        return "\(f.string(from: meeting.start)) – \(f.string(from: meeting.end))"
    }

    private func countdown(now: Date) -> String {
        let secs = Int(meeting.start.timeIntervalSince(now).rounded())
        if secs <= 0 {
            let ago = -secs / 60
            return ago == 0 ? "Starting now" : "Started \(ago) min ago"
        }
        return String(format: "Starts in %d:%02d", secs / 60, secs % 60)
    }
}
