# Calendar Reminder

A macOS menu bar app that watches Google Calendar for one or more Google accounts and pops up an
alert (with sound) a couple of minutes before each meeting starts, similar to the Calendar Bell
Chrome extension.

- Menu bar shows a countdown to the next meeting and lists the next 24 hours of meetings.
- Alert window floats above all apps and Spaces, has **Join** (Meet/Zoom/Teams/Webex link),
  **Open Event**, **Snooze** and **Dismiss** buttons, and plays a system sound.
- Multiple Google accounts; a meeting that shows up on several accounts alerts only once.
- Only alerts for meetings you've accepted (or your own events without guests). Declined,
  tentative and unanswered invites, all-day events, working-location and out-of-office entries
  are ignored.
- Meetings without a video link (Meet/Zoom/Teams/Webex/Chime) alert silently, and the window
  closes on its own after 5 minutes.
- Refresh tokens are stored in `~/Library/Application Support/CalendarReminder/tokens.json` (mode 600);
  calendars are polled every 5 minutes and after wake from sleep.

Requires macOS 13+ and the Swift toolchain (Xcode Command Line Tools are enough).

## 1. Create a Google OAuth client (one time)

Google requires every app that reads Calendar to have its own OAuth client.

1. Open <https://console.cloud.google.com/> and create a project (e.g. "Calendar Reminder").
2. **APIs & Services → Library** → enable **Google Calendar API**.
3. **Google Auth Platform → Branding**: fill in app name and your email.
4. **Audience**:
   - User type **External** (needed if you use personal Gmail or accounts from other orgs).
   - While in **Testing**, add every Google account you'll sign in with under **Test users**.
     Note: in Testing mode Google expires refresh tokens after **7 days**, so you'd have to
     re-authenticate weekly. To avoid that, click **Publish app** (it's fine to stay unverified
     for personal use; you'll just click through an "unverified app" warning when signing in).
   - If all your accounts are in one Google Workspace org, you can pick **Internal** instead and
     skip both of the above.
5. **Data Access** → Add scope `https://www.googleapis.com/auth/calendar.readonly`.
6. **Clients → Create client** → Application type **Desktop app** → Create →
   **Download JSON**.

## 2. Build and run

```bash
./build.sh            # builds "build/Calendar Reminder.app"
./build.sh install    # also copies it to /Applications and launches it
```

If `client_secrets.json` (or `client_secret.json`) is in the project folder, `build.sh` bundles it
into the app as the default OAuth client. It's git-ignored; don't commit it.

On first launch the Settings window opens:

1. If you didn't bundle a client: **Import client_secret.json…** and pick the JSON from step 1.6.
   An imported client overrides the bundled one; **Clear** reverts to the bundled client.
2. **Add Google Account…** — your browser opens to Google sign-in. Repeat for each account.
3. Optionally pick the lead time (default 2 min), sound, and **Launch at login**.

Use **Test Alert** in Settings to preview the popup and sound.

## Notes

- "Launch at login" works best when the app lives in `/Applications`.
- Settings live in `~/Library/Preferences/com.github.frankfliu.CalendarReminder.plist`.
