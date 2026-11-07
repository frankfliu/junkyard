import AppKit
import SwiftUI

struct SettingsView: View {
    @ObservedObject var model: AppModel
    @AppStorage("clientID") private var clientID = ""
    @AppStorage("clientSecret") private var clientSecret = ""
    @AppStorage("leadMinutes") private var leadMinutes = 2
    @AppStorage("soundName") private var soundName = "Glass"
    @AppStorage("repeatSound") private var repeatSound = true
    @AppStorage("includeAllCalendars") private var includeAllCalendars = true

    var body: some View {
        Form {
            Section("Google accounts") {
                HStack {
                    Button("Add Google Account…") { Task { await model.addAccount() } }
                        .disabled((clientID.isEmpty && Prefs.bundledClient == nil) || model.signingIn)
                    if let status = model.signInStatus {
                        Text(status).foregroundStyle(.secondary).lineLimit(2)
                    }
                }
                ScrollView {
                    VStack(spacing: 0) {
                        if model.accounts.isEmpty {
                            Text("No accounts yet.")
                                .foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.vertical, 6)
                        }
                        ForEach(model.accounts, id: \.self) { email in
                            accountRow(email)
                            if email != model.accounts.last {
                                Divider()
                            }
                        }
                    }
                    .padding(.horizontal, 8)
                }
                .frame(height: Self.accountListHeight)
                .background(Color(nsColor: .textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(nsColor: .separatorColor)))
            }

            Section("Alerts") {
                Stepper("Alert \(leadMinutes) min before a meeting starts", value: $leadMinutes, in: 0...30)
                Picker("Sound", selection: $soundName) {
                    ForEach(Prefs.systemSounds + ["None"], id: \.self) { Text($0).tag($0) }
                }
                .onChange(of: soundName) { name in NSSound(named: NSSound.Name(name))?.play() }
                Toggle("Repeat sound until dismissed (max 1 min)", isOn: $repeatSound)
                Toggle("Watch all calendars shown in Google Calendar (off: primary only)", isOn: $includeAllCalendars)
                    .onChange(of: includeAllCalendars) { _ in Task { await model.refresh() } }
                Toggle("Launch at login", isOn: $model.launchAtLogin)
                if let loginError = model.loginError {
                    Text(loginError).font(.caption).foregroundStyle(.red)
                }
            }

            Section("Google OAuth client") {
                HStack {
                    if !clientID.isEmpty {
                        Label("Configured (imported)", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    } else if Prefs.bundledClient != nil {
                        Label("Configured (bundled with this build)", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    } else {
                        Label("Not configured", systemImage: "xmark.circle.fill")
                            .foregroundStyle(.red)
                    }
                    Spacer()
                    if !clientID.isEmpty {
                        Button("Clear") {
                            clientID = ""
                            clientSecret = ""
                        }
                    }
                    Button("Import client_secret.json…") { importClientJSON() }
                }
                HStack {
                    Text("Create a \"Desktop app\" OAuth client with the Calendar API enabled. See README.md.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Test Alert") { model.testAlert() }
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 600, height: 625)
    }

    /// Fits about three accounts; more scroll.
    private static let accountListHeight: CGFloat = 114

    private func accountRow(_ email: String) -> some View {
        HStack {
            Text(email)
            if let err = model.errors[email] {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.yellow)
                    .help(err)
            }
            Spacer()
            Button("Re-authenticate") { Task { await model.addAccount() } }
                .disabled(model.signingIn)
            Button("Remove") { model.removeAccount(email) }
        }
        .padding(.vertical, 6)
    }

    private func importClientJSON() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url,
              let data = try? Data(contentsOf: url),
              let client = Prefs.parseClientJSON(data)
        else {
            model.signInStatus = "Could not read client ID from that file."
            return
        }
        clientID = client.id
        clientSecret = client.secret
        model.signInStatus = "Imported OAuth client."
    }
}
