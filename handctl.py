#!/usr/bin/env python3
"""Hand-tracked input for Hyprland.

Three modes, toggle by holding a closed fist on either hand (or `m`):
  - mouse:     cursor follows your left hand; pinch = hold left button
  - window:    left pinch grabs the active window and drags it;
               left double-tap toggles floating;
               BOTH hands pinched = two-handed resize (pull apart / push together)
  - workspace: left pinch + sweep left/right cycles workspaces

ESC quits.
"""
import json
import subprocess
import sys
import time

import cv2
import mediapipe as mp

# Hysteresis: pinch engages only when fingers are clearly close (ENGAGE),
# and releases only when they're clearly apart (RELEASE). The gap is a
# dead zone where the previous state is held — kills threshold flicker.
PINCH_ENGAGE = 0.04
PINCH_RELEASE = 0.07

# Fraction of frame on each edge that's outside the active region. A
# smaller central box maps to the full screen so corners are reachable
# without sending your hand to the frame edge where tracking fails.
ACTIVE_MARGIN = 0.2

# 0 = raw (jittery), 1 = frozen. ~0.75 is heavy low-pass.
SMOOTHING = 0.75

CAM_INDEX = 0

# Hold a closed fist for this long to toggle modes.
FIST_HOLD_SECONDS = 0.6
# A finger counts as curled if tip-to-MCP distance is less than this
# fraction of the palm size (wrist→middle MCP).
FIST_CURL_RATIO = 0.5
# A finger counts as extended if tip-to-MCP distance exceeds this fraction
# of palm size. Loose enough that a pinching index still reads as extended.
FINGER_EXTEND_RATIO = 0.65

MODES = ["mouse", "window", "workspace"]

# Horizontal hand displacement (normalized 0..1) during a pinch that
# triggers one workspace step in workspace mode.
SWIPE_THRESHOLD = 0.15

# Window mode: a pinch shorter than TAP_MAX counts as a tap (no drag).
# Two taps inside DOUBLE_TAP_WINDOW toggle floating on the active window.
TAP_MAX = 0.25
DOUBLE_TAP_WINDOW = 0.4

# Don't let two-handed resize collapse a window below this many pixels.
MIN_WIN_SIZE = 100

# Hold a V-sign (index + middle extended) for this long to start voxtype
# recording. Dropping the gesture stops + transcribes. Mirrors Super+V.
V_HOLD_SECONDS = 0.4

# Hold a three-finger pose (index + middle + ring) in window mode for this
# long to launch walker. Short enough to feel snappy, long enough to ignore
# transient finger states.
WALKER_HOLD_SECONDS = 0.3

# Hold a finger-count pose on the right hand in window mode for this long
# to switch workspace. 1 finger = ws 1, V-sign = ws 2, three fingers = ws 3.
WS_HOLD_SECONDS = 0.4

# ydotool button bytes: low nibble = button (0x0 = left), high nibble =
# action flag (0x40 = down, 0x80 = up). 0xC0 would be a full click.
YDOTOOL_LEFT_DOWN = "0x40"
YDOTOOL_LEFT_UP = "0x80"


def hypr_dispatch(*args):
    subprocess.run(
        ["hyprctl", "dispatch", *map(str, args)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def hypr_json(query):
    out = subprocess.run(
        ["hyprctl", "-j", query], capture_output=True, text=True, check=False
    )
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None


def focused_monitor_size():
    for m in hypr_json("monitors") or []:
        if m.get("focused"):
            return m["width"], m["height"]
    return 1920, 1080


def active_window_size():
    w = hypr_json("activewindow") or {}
    sz = w.get("size") or [800, 600]
    return sz[0], sz[1]


def mouse_down():
    subprocess.Popen(
        ["ydotool", "click", YDOTOOL_LEFT_DOWN],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def mouse_up():
    subprocess.Popen(
        ["ydotool", "click", YDOTOOL_LEFT_UP],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def dist2d(ax, ay, bx, by):
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def remap(v):
    """Map ACTIVE_MARGIN..(1-ACTIVE_MARGIN) into 0..1, clamped."""
    return max(0.0, min(1.0, (v - ACTIVE_MARGIN) / (1 - 2 * ACTIVE_MARGIN)))


def is_fist(lm):
    """All four non-thumb fingers curled toward the palm."""
    palm = dist(lm[0], lm[9])  # wrist → middle-finger MCP, scale reference
    if palm == 0:
        return False
    for tip_idx, mcp_idx in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        if dist(lm[tip_idx], lm[mcp_idx]) > FIST_CURL_RATIO * palm:
            return False
    return True


def is_point(lm):
    """Pointing pose: index extended, middle/ring/pinky curled."""
    palm = dist(lm[0], lm[9])
    if palm == 0:
        return False
    if dist(lm[8], lm[5]) < FINGER_EXTEND_RATIO * palm:
        return False
    for tip_idx, mcp_idx in [(12, 9), (16, 13), (20, 17)]:
        if dist(lm[tip_idx], lm[mcp_idx]) > FIST_CURL_RATIO * palm:
            return False
    return True


def is_v_sign(lm):
    """Peace/V sign: index + middle extended, ring + pinky curled."""
    palm = dist(lm[0], lm[9])
    if palm == 0:
        return False
    if dist(lm[8], lm[5]) < FINGER_EXTEND_RATIO * palm:
        return False
    if dist(lm[12], lm[9]) < FINGER_EXTEND_RATIO * palm:
        return False
    for tip_idx, mcp_idx in [(16, 13), (20, 17)]:
        if dist(lm[tip_idx], lm[mcp_idx]) > FIST_CURL_RATIO * palm:
            return False
    return True


def is_three(lm):
    """Three-finger pose: index + middle + ring extended, pinky curled."""
    palm = dist(lm[0], lm[9])
    if palm == 0:
        return False
    for tip_idx, mcp_idx in [(8, 5), (12, 9), (16, 13)]:
        if dist(lm[tip_idx], lm[mcp_idx]) < FINGER_EXTEND_RATIO * palm:
            return False
    if dist(lm[20], lm[17]) > FIST_CURL_RATIO * palm:
        return False
    return True


def voxtype_start():
    subprocess.Popen(
        ["voxtype", "record", "start"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def voxtype_stop():
    subprocess.Popen(
        ["voxtype", "record", "stop"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def empty_hand():
    return {
        "present": False,
        "pinched": False,
        "was_pinched": False,
        "sx": None,
        "sy": None,
        "fist": False,
        "point": False,
        "v_sign": False,
        "three": False,
        "pinch_start_time": None,
    }


def main():
    screen_w, screen_h = focused_monitor_size()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        sys.exit("could not open webcam")

    # WINDOW_GUI_NORMAL strips Qt's built-in toolbar/statusbar (the
    # zoom/pan/save chrome that opencv-python adds by default).
    cv2.namedWindow("handctl", cv2.WINDOW_GUI_NORMAL)

    # Per-hand state, indexed by physical side (after mirror-flip correction).
    H = {"left": empty_hand(), "right": empty_hand()}

    mode_idx = 0
    fist_start = None           # time fist closed (either hand), or None
    fist_consumed = False       # already triggered switch during this fist
    swipe_anchor_x = None       # L[sx] at last pinch-start (workspace mode)
    win_w, win_h = 800, 600     # window size cached on grab (window mode drag)
    last_tap_time = 0.0         # release time of last short tap (window mode)
    # Left-hand pinch overlapped with a right-hand pinch at some point.
    # When set, suppress tap-detection on left release so two-handed
    # resize sessions don't accidentally trigger togglefloating.
    left_pinch_paired = False
    # While both hands are pinched in window mode: snapshot at gesture start.
    resize_state = None         # {"start_dist", "start_w", "start_h"} or None
    show_hud = True             # press 'h' to toggle overlays on/off
    positioned = False          # have we parked the preview yet?
    v_start = None              # time V-sign first seen (either hand), or None
    v_active = False            # we've fired voxtype record start, awaiting release
    walker_start = None         # time three-finger pose first seen, or None
    walker_fired = False        # already launched walker for this hold
    ws_pose = None              # current right-hand workspace pose (1/2/3) or None
    ws_start = None             # when ws_pose first detected
    ws_fired = False            # already switched for this pose hold

    def switch_mode():
        nonlocal mode_idx, resize_state, left_pinch_paired, last_tap_time
        nonlocal walker_start, walker_fired, ws_pose, ws_start, ws_fired
        if MODES[mode_idx] == "mouse" and H["left"]["was_pinched"]:
            mouse_up()
        for h in H.values():
            h["pinched"] = False
            h["was_pinched"] = False
            h["pinch_start_time"] = None
        resize_state = None
        left_pinch_paired = False
        last_tap_time = 0.0
        walker_start = None
        walker_fired = False
        ws_pose = None
        ws_start = None
        ws_fired = False
        mode_idx = (mode_idx + 1) % len(MODES)

    with mp.solutions.hands.Hands(
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            mode = MODES[mode_idx]

            # Roll forward "previous frame" pinch state, then mark all hands
            # absent — anything actually detected will set present=True below.
            for h in H.values():
                h["was_pinched"] = h["pinched"]
                h["present"] = False

            if res.multi_hand_landmarks and res.multi_handedness:
                for landmarks, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = hd.classification[0].label
                    # cv2.flip mirrored the frame, so MediaPipe's "Left"
                    # label corresponds to the user's physical right hand.
                    side = "right" if label == "Left" else "left"
                    h = H[side]
                    h["present"] = True
                    lm = landmarks.landmark
                    h["fist"] = is_fist(lm)
                    h["point"] = is_point(lm)
                    h["v_sign"] = is_v_sign(lm)
                    h["three"] = is_three(lm)
                    d = dist(lm[4], lm[8])
                    h["pinched"] = (
                        d < PINCH_RELEASE if h["was_pinched"] else d < PINCH_ENGAGE
                    )
                    anchor = lm[5]  # index MCP knuckle — stable when fingers move
                    if h["sx"] is None:
                        h["sx"], h["sy"] = anchor.x, anchor.y
                    else:
                        h["sx"] = SMOOTHING * h["sx"] + (1 - SMOOTHING) * anchor.x
                        h["sy"] = SMOOTHING * h["sy"] + (1 - SMOOTHING) * anchor.y

            # Hands that vanished this frame: drop state so a reappearing
            # hand doesn't smooth from a stale position.
            for h in H.values():
                if not h["present"]:
                    h["sx"] = h["sy"] = None
                    h["pinched"] = False
                    h["fist"] = False
                    h["point"] = False
                    h["v_sign"] = False
                    h["three"] = False

            # Fist-hold mode toggle (either hand). Suppresses pinch input
            # while held so closing your hand can't double as a pinch.
            now = time.time()
            any_fist = any(h["fist"] for h in H.values())
            if any_fist:
                if fist_start is None:
                    fist_start = now
                    fist_consumed = False
                elif not fist_consumed and now - fist_start >= FIST_HOLD_SECONDS:
                    switch_mode()
                    fist_consumed = True
                for h in H.values():
                    h["pinched"] = False
            else:
                fist_start = None
                fist_consumed = False

            # V-sign hold (LEFT hand only): push-to-talk for voxtype.
            # Same contract as Super+V. Left-hand-only because right-hand
            # V-sign is repurposed for workspace 2 switching in window mode.
            any_v = H["left"]["v_sign"]
            if any_v:
                if v_start is None:
                    v_start = now
                elif not v_active and now - v_start >= V_HOLD_SECONDS:
                    voxtype_start()
                    v_active = True
            else:
                if v_active:
                    voxtype_stop()
                    v_active = False
                v_start = None

            L = H["left"]
            R = H["right"]

            if mode == "mouse":
                # Cursor follows the dominant hand whenever it's visible,
                # regardless of finger pose.
                active = L if L["present"] else (R if R["present"] else None)
                if active is not None:
                    cx = int(remap(active["sx"]) * screen_w)
                    cy = int(remap(active["sy"]) * screen_h)
                    hypr_dispatch("movecursor", f"{cx} {cy}")
                if L["pinched"] and not L["was_pinched"]:
                    mouse_down()
                elif not L["pinched"] and L["was_pinched"]:
                    mouse_up()

            elif mode == "window":
                # Three-finger hold on LEFT hand launches walker.
                if L["three"]:
                    if walker_start is None:
                        walker_start = now
                    elif (not walker_fired
                            and now - walker_start >= WALKER_HOLD_SECONDS):
                        hypr_dispatch("exec", "walker")
                        walker_fired = True
                else:
                    walker_start = None
                    walker_fired = False

                # Workspace switching on RIGHT hand: hold N fingers extended,
                # fires hyprctl workspace N. Each pose is checked in order
                # of specificity (3-finger first, then 2, then 1).
                ws_n = None
                if R["present"]:
                    if R["three"]:
                        ws_n = 3
                    elif R["v_sign"]:
                        ws_n = 2
                    elif R["point"]:
                        ws_n = 1
                if ws_n is not None:
                    if ws_pose != ws_n:
                        ws_pose = ws_n
                        ws_start = now
                        ws_fired = False
                    elif (not ws_fired
                            and now - ws_start >= WS_HOLD_SECONDS):
                        hypr_dispatch("workspace", str(ws_n))
                        ws_fired = True
                else:
                    ws_pose = None
                    ws_start = None
                    ws_fired = False

                both_pinched = L["pinched"] and R["pinched"]
                both_were_pinched = L["was_pinched"] and R["was_pinched"]

                if both_pinched and not both_were_pinched:
                    # Two-handed resize begins.
                    cur_dist = max(0.01, dist2d(L["sx"], L["sy"], R["sx"], R["sy"]))
                    ww, hh = active_window_size()
                    resize_state = {
                        "start_dist": cur_dist,
                        "start_w": ww,
                        "start_h": hh,
                    }
                    left_pinch_paired = True

                if both_pinched and resize_state is not None:
                    cur_dist = dist2d(L["sx"], L["sy"], R["sx"], R["sy"])
                    scale = cur_dist / resize_state["start_dist"]
                    new_w = max(MIN_WIN_SIZE, int(resize_state["start_w"] * scale))
                    new_h = max(MIN_WIN_SIZE, int(resize_state["start_h"] * scale))
                    mx = (L["sx"] + R["sx"]) / 2
                    my = (L["sy"] + R["sy"]) / 2
                    tx = int(remap(mx) * screen_w - new_w / 2)
                    ty = int(remap(my) * screen_h - new_h / 2)
                    hypr_dispatch("resizeactive", "exact", f"{new_w} {new_h}")
                    hypr_dispatch("moveactive", "exact", f"{tx} {ty}")
                elif both_were_pinched and not both_pinched:
                    # Resize gesture just ended. If left hand is still
                    # pinched, re-baseline so the drag doesn't snap.
                    resize_state = None
                    if L["pinched"]:
                        L["pinch_start_time"] = now
                        win_w, win_h = active_window_size()
                else:
                    # Single-hand drag (left only). Track when right
                    # joins/leaves so we know whether to count a tap.
                    if L["pinched"] and R["pinched"]:
                        left_pinch_paired = True
                    if L["pinched"] and not L["was_pinched"]:
                        L["pinch_start_time"] = now
                        win_w, win_h = active_window_size()
                        left_pinch_paired = R["pinched"]
                    # Hold past TAP_MAX before dragging — short pinches
                    # are taps, not nudges.
                    if (L["pinched"] and L["sx"] is not None
                            and L["pinch_start_time"] is not None
                            and now - L["pinch_start_time"] > TAP_MAX):
                        tx = int(remap(L["sx"]) * screen_w - win_w / 2)
                        ty = int(remap(L["sy"]) * screen_h - win_h / 2)
                        hypr_dispatch("moveactive", "exact", f"{tx} {ty}")
                    # Left release: tap accounting.
                    if (not L["pinched"] and L["was_pinched"]
                            and L["pinch_start_time"] is not None):
                        duration = now - L["pinch_start_time"]
                        if not left_pinch_paired and duration < TAP_MAX:
                            if now - last_tap_time < DOUBLE_TAP_WINDOW:
                                hypr_dispatch("togglefloating")
                                last_tap_time = 0.0
                            else:
                                last_tap_time = now
                        else:
                            last_tap_time = 0.0
                        L["pinch_start_time"] = None
                        left_pinch_paired = False

            elif mode == "workspace":
                # Left hand pinch + horizontal sweep.
                if L["pinched"] and not L["was_pinched"]:
                    swipe_anchor_x = L["sx"]
                elif not L["pinched"]:
                    swipe_anchor_x = None
                if L["pinched"] and L["sx"] is not None and swipe_anchor_x is not None:
                    dxs = L["sx"] - swipe_anchor_x
                    if dxs > SWIPE_THRESHOLD:
                        hypr_dispatch("workspace", "+1")
                        swipe_anchor_x = L["sx"]
                    elif dxs < -SWIPE_THRESHOLD:
                        hypr_dispatch("workspace", "-1")
                        swipe_anchor_x = L["sx"]

            # HUD (toggle with 'h').
            if show_hud:
                fh, fw = frame.shape[:2]
                cv2.rectangle(
                    frame,
                    (int(ACTIVE_MARGIN * fw), int(ACTIVE_MARGIN * fh)),
                    (int((1 - ACTIVE_MARGIN) * fw), int((1 - ACTIVE_MARGIN) * fh)),
                    (80, 80, 80),
                    1,
                )

                def hand_tag(h):
                    if not h["present"]:
                        return "-"
                    if h["fist"]:
                        return "FIST"
                    return "PINCH" if h["pinched"] else "open"

                cv2.putText(
                    frame,
                    f"[{mode}] L:{hand_tag(L)} R:{hand_tag(R)}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0) if (L["pinched"] or R["pinched"]) else (200, 200, 200),
                    2,
                )
                cv2.putText(
                    frame,
                    "fist or 'm' to toggle mode    'h' HUD    ESC: quit",
                    (10, fh - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 180),
                    1,
                )

                # Per-hand dot at the smoothed anchor (left = green dominant, right = magenta).
                for side, color in (("left", (0, 255, 0)), ("right", (255, 0, 255))):
                    h = H[side]
                    if h["present"] and h["sx"] is not None:
                        px = int(h["sx"] * fw)
                        py = int(h["sy"] * fh)
                        cv2.circle(frame, (px, py), 8, color, 2)
                        if h["pinched"]:
                            cv2.circle(frame, (px, py), 4, color, -1)

                # Window mode: bar between hands while resizing.
                if (mode == "window" and resize_state is not None
                        and L["present"] and R["present"]):
                    lx, ly = int(L["sx"] * fw), int(L["sy"] * fh)
                    rx, ry = int(R["sx"] * fw), int(R["sy"] * fh)
                    cv2.line(frame, (lx, ly), (rx, ry), (0, 200, 255), 2)

                # Workspace mode: swipe progress indicator (only while pinched).
                if (mode == "workspace" and L["pinched"]
                        and swipe_anchor_x is not None and L["sx"] is not None):
                    dxs = L["sx"] - swipe_anchor_x
                    frac = max(-1.0, min(1.0, dxs / SWIPE_THRESHOLD))
                    cx0 = fw // 2
                    cy0 = 70
                    half = int(0.25 * fw)
                    cv2.line(frame, (cx0 - half, cy0), (cx0 + half, cy0), (80, 80, 80), 1)
                    cv2.line(frame, (cx0, cy0 - 6), (cx0, cy0 + 6), (80, 80, 80), 1)
                    cv2.circle(frame, (cx0 + int(frac * half), cy0), 6, (0, 200, 255), -1)

                # Fist hold progress bar.
                if any_fist and fist_start is not None and not fist_consumed:
                    progress = min(1.0, (now - fist_start) / FIST_HOLD_SECONDS)
                    bar_w = int(0.4 * fw)
                    bar_x = (fw - bar_w) // 2
                    bar_y = 70
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10),
                                  (80, 80, 80), 1)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + int(bar_w * progress), bar_y + 10),
                                  (0, 200, 255), -1)

                # V-sign hold progress + recording indicator.
                if v_active:
                    cv2.putText(frame, "REC", (fw - 90, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                elif any_v and v_start is not None:
                    progress = min(1.0, (now - v_start) / V_HOLD_SECONDS)
                    bar_w = int(0.4 * fw)
                    bar_x = (fw - bar_w) // 2
                    bar_y = 90
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10),
                                  (80, 80, 80), 1)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + int(bar_w * progress), bar_y + 10),
                                  (0, 100, 255), -1)

                # Walker hold progress (window mode three-finger).
                if (mode == "window" and walker_start is not None
                        and not walker_fired):
                    progress = min(1.0, (now - walker_start) / WALKER_HOLD_SECONDS)
                    bar_w = int(0.4 * fw)
                    bar_x = (fw - bar_w) // 2
                    bar_y = 110
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10),
                                  (80, 80, 80), 1)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + int(bar_w * progress), bar_y + 10),
                                  (255, 200, 0), -1)

                # Workspace hold progress + target label.
                if (mode == "window" and ws_pose is not None
                        and ws_start is not None and not ws_fired):
                    progress = min(1.0, (now - ws_start) / WS_HOLD_SECONDS)
                    bar_w = int(0.4 * fw)
                    bar_x = (fw - bar_w) // 2
                    bar_y = 130
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10),
                                  (80, 80, 80), 1)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + int(bar_w * progress), bar_y + 10),
                                  (0, 255, 200), -1)
                    cv2.putText(frame, f"WS {ws_pose}",
                                (bar_x + bar_w + 8, bar_y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 200), 1)

            cv2.imshow("handctl", frame)
            # First frame just mapped the window — park it bottom-right
            # with enough margin to stay clear of bars / reserved areas.
            if not positioned:
                tx = screen_w - 400
                ty = screen_h - 320
                hypr_dispatch("movewindowpixel",
                              f"exact {tx} {ty},title:^(handctl)$")
                positioned = True

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("m"):
                switch_mode()
            if key == ord("h"):
                show_hud = not show_hud

    if H["left"]["was_pinched"] and MODES[mode_idx] == "mouse":
        mouse_up()
    if v_active:
        voxtype_stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
