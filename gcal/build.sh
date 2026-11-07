#!/bin/bash
# Builds "Calendar Reminder.app" into ./build. Pass "install" to also copy it to /Applications.
set -euo pipefail
cd "$(dirname "$0")"

swift build -c release

APP="build/Calendar Reminder.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$(swift build -c release --show-bin-path)/CalendarReminder" "$APP/Contents/MacOS/"
cp Resources/Info.plist "$APP/Contents/"
# Bundle a default OAuth client if one is present (kept out of git).
for f in client_secrets.json client_secret.json; do
    if [[ -f "$f" ]]; then
        cp "$f" "$APP/Contents/Resources/client_secret.json"
        echo "Bundled OAuth client from $f"
        break
    fi
done
codesign --force --sign - "$APP"
echo "Built $APP"

if [[ "${1:-}" == "install" ]]; then
    pkill -x CalendarReminder || true
    rm -rf "/Applications/Calendar Reminder.app"
    cp -R "$APP" /Applications/
    echo "Installed /Applications/Calendar Reminder.app"
    open "/Applications/Calendar Reminder.app"
fi
