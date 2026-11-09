import AppKit
import SwiftUI

@main
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    private let model = AppModel()
    private var statusItem: NSStatusItem!
    private var settingsWindow: NSWindow?

    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.accessory)  // menu bar only, no Dock icon
        withExtendedLifetime(delegate) { app.run() }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        Prefs.registerDefaults()
        installMainMenu()

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        let menu = NSMenu()
        menu.delegate = self
        statusItem.menu = menu

        model.onChange = { [weak self] in self?.updateStatusItem() }
        model.start()
        updateStatusItem()

        if Prefs.client == nil || model.accounts.isEmpty {
            openSettings()
        }
    }

    // MARK: - Status item

    private func updateStatusItem() {
        guard let button = statusItem?.button else { return }
        let now = Date()
        let symbol = model.accounts.isEmpty || model.pausedUntil != nil
            ? "bell.slash" : (model.errors.isEmpty ? "bell" : "bell.badge")
        button.image = NSImage(systemSymbolName: symbol, accessibilityDescription: "Calendar Reminder")
        button.imagePosition = .imageLeading

        var title = ""
        if let next = model.meetings.first(where: { $0.start > now }) {
            let mins = Int((next.start.timeIntervalSince(now) / 60).rounded(.up))
            if mins <= 60 {
                title = " \(mins)m"
            } else if Calendar.current.isDateInToday(next.start) {
                title = " " + Self.timeFormatter.string(from: next.start)
            }
        }
        button.title = title
    }

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .none
        f.timeStyle = .short
        return f
    }()

    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        let now = Date()

        let upcoming = model.meetings.filter { $0.end > now }.prefix(15)
        if model.accounts.isEmpty {
            menu.addItem(disabled("Open Settings to add a Google account"))
        } else if upcoming.isEmpty {
            menu.addItem(disabled("No meetings in the next 24 hours"))
        }

        var lastDay: String?
        for m in upcoming {
            let day = Calendar.current.isDateInToday(m.start) ? "Today" : "Tomorrow"
            if day != lastDay {
                if lastDay != nil { menu.addItem(.separator()) }
                menu.addItem(disabled(day))
                lastDay = day
            }
            let when = m.start <= now ? "Now" : Self.timeFormatter.string(from: m.start)
            let item = NSMenuItem(title: "\(when)   \(m.title)", action: #selector(openMeeting(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = m.joinURL ?? m.eventURL
            item.toolTip = "\(m.calendar) · \(m.account)"
            if m.joinURL != nil {
                item.image = NSImage(systemSymbolName: "video", accessibilityDescription: "Join")
            }
            menu.addItem(item)
        }

        if !model.errors.isEmpty {
            menu.addItem(.separator())
            for (account, err) in model.errors.sorted(by: { $0.key < $1.key }) {
                let item = disabled("⚠︎ \(account.isEmpty ? "" : account + ": ")\(err)")
                item.toolTip = err
                menu.addItem(item)
            }
        }

        menu.addItem(.separator())
        if let last = model.lastRefresh {
            menu.addItem(disabled("Updated \(Self.timeFormatter.string(from: last))"))
        }
        if let until = model.pausedUntil {
            menu.addItem(action("Unpause (paused until \(Self.timeFormatter.string(from: until)))", #selector(resume), ""))
        } else {
            let pause = NSMenuItem(title: "Pause", action: nil, keyEquivalent: "")
            let sub = NSMenu()
            for hours in [1, 2, 4, 8] {
                let item = action("\(hours) hour\(hours == 1 ? "" : "s")", #selector(pause(_:)), "")
                item.tag = hours
                sub.addItem(item)
            }
            pause.submenu = sub
            menu.addItem(pause)
        }
        menu.addItem(action("Refresh Now", #selector(refreshNow), "r"))
        menu.addItem(action("Settings…", #selector(openSettings), ","))
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(title: "Quit Calendar Reminder", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
    }

    private func disabled(_ title: String) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        item.isEnabled = false
        return item
    }

    private func action(_ title: String, _ selector: Selector, _ key: String) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: selector, keyEquivalent: key)
        item.target = self
        return item
    }

    @objc private func openMeeting(_ sender: NSMenuItem) {
        if let url = sender.representedObject as? URL {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func pause(_ sender: NSMenuItem) {
        model.pause(hours: sender.tag)
    }

    @objc private func resume() {
        model.resume()
    }

    @objc private func refreshNow() {
        Task { await model.refresh() }
    }

    @objc private func openSettings() {
        if settingsWindow == nil {
            let window = NSWindow(contentViewController: NSHostingController(rootView: SettingsView(model: model)))
            window.title = "Calendar Reminder Settings"
            window.styleMask = [.titled, .closable]
            window.isReleasedWhenClosed = false
            window.center()
            settingsWindow = window
        }
        NSApp.activate(ignoringOtherApps: true)
        settingsWindow?.makeKeyAndOrderFront(nil)
    }

    /// Accessory apps have no menu bar, but text fields need an Edit menu for ⌘C/⌘V to work.
    private func installMainMenu() {
        let main = NSMenu()

        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Close Window", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        appMenu.addItem(withTitle: "Quit Calendar Reminder", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let appItem = NSMenuItem()
        appItem.submenu = appMenu
        main.addItem(appItem)

        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        let editItem = NSMenuItem()
        editItem.submenu = edit
        main.addItem(editItem)

        NSApp.mainMenu = main
    }
}
