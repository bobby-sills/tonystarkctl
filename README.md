# tonystarkctl

Control your computer with your hands (like tony stark).

Hand-tracked input for Hyprland: webcam → MediaPipe → `hyprctl` / `ydotool`.

## Modes

Toggle by holding a closed fist on either hand (or pressing `m`):

- **mouse** — cursor follows your left hand; pinch = hold left mouse button.
- **window** — left-hand pinch grabs and drags the active window; left double-tap toggles floating; both hands pinched = two-handed resize. Right hand finger-count (1/2/3) holds switch workspace. Left-hand three-finger hold launches walker. Left-hand V-sign hold = voxtype push-to-talk.

`h` toggles the HUD, `ESC` quits.

## Requirements

- **Hyprland** (uses `hyprctl dispatch`)
- **A webcam** at index 0 (override `CAM_INDEX` in `handctl.py`)
- **ydotool + ydotoold** — needed for mouse button down/up (real click events that support drag). The script expects the daemon socket at `/run/ydotoold/socket`.
- **Python 3.9–3.12** (MediaPipe only ships wheels for these)

Optional, only used by specific gestures:

- `walker` — three-finger hold in window mode
- `voxtype` — V-sign hold push-to-talk

## Install & run

### NixOS (recommended)

`shell.nix` provides an FHS environment with the C libs the pip wheels need, and bootstraps a project-local `.venv` with `mediapipe` and `opencv-python` on first entry (~150 MB download).

```bash
git clone https://github.com/<you>/tonystarkctl
cd tonystarkctl
nix-shell                # first entry installs the venv
python handctl.py
```

Enable `ydotoold` system-wide in your NixOS config:

```nix
programs.ydotool.enable = true;
```

This creates `/run/ydotoold/socket` automatically. Add your user to the `ydotool` group if your distro requires it (NixOS does not).

If the venv ever gets out of sync with the current nixpkgs Python (e.g. after `nix-collect-garbage`), `rm -rf .venv` and re-enter `nix-shell`.

### Other Linux

You'll need to provide the runtime libs yourself. Roughly:

```bash
# Debian/Ubuntu
sudo apt install python3-venv libgl1 libglib2.0-0 libxcb1 libxkbcommon0 \
                 ydotool v4l-utils

python3 -m venv .venv
source .venv/bin/activate
pip install "mediapipe==0.10.21" opencv-python

# Start ydotoold (one-time, needs root for /dev/uinput)
sudo ydotoold &
export YDOTOOL_SOCKET=/tmp/.ydotool_socket   # or wherever yours lives

python handctl.py
```

If OpenCV's bundled Qt can't find a platform plugin, set `QT_QPA_PLATFORM=xcb`.
