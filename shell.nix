{ pkgs ? import <nixpkgs> { } }:

# mediapipe isn't packaged in nixpkgs, so we run inside a buildFHSEnv
# that provides the C libs the pip wheels expect, and pull mediapipe +
# opencv-python into a project-local .venv on first entry.

(pkgs.buildFHSEnv {
  name = "handctl";

  targetPkgs = p: with p; [
    python312  # mediapipe only ships wheels for 3.9-3.12
    # C runtime libs the wheels link against
    stdenv.cc.cc.lib
    zlib
    glib
    libGL
    libglvnd
    # GUI for cv2.imshow + webcam
    gtk3
    v4l-utils
    ydotool  # used by handctl.py for mouse button down/up (supports drag)
    # opencv-python wheel bundles Qt; needs X11/xcb at runtime
    libxcb
    libx11
    libxext
    libxrender
    libsm
    libice
    libxcb-util
    libxcb-image
    libxcb-keysyms
    libxcb-render-util
    libxcb-wm
    libxkbcommon
    dbus
    fontconfig
    freetype
  ];

  profile = ''
    if [ ! -f "$PWD/.venv/.handctl-ready" ]; then
      echo "installing mediapipe + opencv-python into .venv (~150MB)..."
      rm -rf "$PWD/.venv"
      python -m venv .venv \
        && .venv/bin/pip install --upgrade pip \
        && .venv/bin/pip install "mediapipe==0.10.21" opencv-python \
        && touch "$PWD/.venv/.handctl-ready"
    fi
    source "$PWD/.venv/bin/activate"
    # opencv-python's bundled Qt only ships the xcb plugin; force it
    # instead of letting Qt try (and fail to find) the wayland plugin.
    export QT_QPA_PLATFORM=xcb
    # NixOS ydotoold listens on /run/ydotoold/socket, not the default.
    export YDOTOOL_SOCKET=/run/ydotoold/socket
    echo "handctl shell ready — run: python handctl.py"
  '';

  runScript = "bash";
}).env
