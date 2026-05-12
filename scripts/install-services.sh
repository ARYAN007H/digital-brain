#!/usr/bin/env bash
# ── Install Digital Brain systemd user services ──────────
# Enables auto-start on login and auto-restart on failure.
# Run: ./scripts/install-services.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SYSTEMD_DIR="$HOME/.config/systemd/user"

echo "🧠 Installing Digital Brain systemd services..."
echo ""

# Create systemd user directory
mkdir -p "$SYSTEMD_DIR"

# ── Patch service files with actual paths ────────────────
for svc in digital-brain-pocketbase.service \
           digital-brain-watcher.service \
           digital-brain-nightly.service \
           digital-brain-nightly.timer; do
    src="$SCRIPT_DIR/$svc"
    dst="$SYSTEMD_DIR/$svc"

    if [ ! -f "$src" ]; then
        echo "  ⚠ Missing: $svc"
        continue
    fi

    # Replace %h with actual HOME and copy
    sed "s|%h|$HOME|g" "$src" > "$dst"
    echo "  ✅ Installed: $svc"
done

# ── Reload systemd ──────────────────────────────────────
systemctl --user daemon-reload

# ── Enable services ─────────────────────────────────────
echo ""
echo "▸ Enabling services..."

systemctl --user enable digital-brain-pocketbase.service
echo "  ✅ PocketBase (auto-start on login)"

systemctl --user enable digital-brain-watcher.service
echo "  ✅ Inbox watcher (auto-start on login)"

systemctl --user enable digital-brain-nightly.timer
echo "  ✅ Nightly surfacing timer (2am daily)"

# ── Start services now ──────────────────────────────────
echo ""
echo "▸ Starting services..."

systemctl --user start digital-brain-pocketbase.service
sleep 2  # give PocketBase time to start
systemctl --user start digital-brain-watcher.service
systemctl --user start digital-brain-nightly.timer

echo ""
echo "════════════════════════════════════════════"
echo "✅ Services installed and running!"
echo ""
echo "Manage with:"
echo "  systemctl --user status digital-brain-pocketbase"
echo "  systemctl --user status digital-brain-watcher"
echo "  systemctl --user status digital-brain-nightly.timer"
echo "  journalctl --user -u digital-brain-watcher -f"
echo ""
echo "Disable with:"
echo "  systemctl --user disable --now digital-brain-pocketbase"
echo "  systemctl --user disable --now digital-brain-watcher"
echo "  systemctl --user disable --now digital-brain-nightly.timer"
echo "════════════════════════════════════════════"
