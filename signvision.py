"""
SignVision AI  —  ASL Sign Language Translator
==============================================
No training needed. Just run it.

INSTALL:
    pip install opencv-python mediapipe numpy pyttsx3

RUN:
    python signvision.py

GESTURES  (hold still ~1 second to type):
    A  Fist, thumb to side
    B  Four fingers up, thumb tucked
    C  Curved C shape
    D  Index up, others curl to thumb
    I  Only pinky up
    L  Index + thumb make L shape
    O  All fingers form O circle
    U  Two fingers up, together
    V  Peace sign, fingers spread
    W  Three fingers up
    Y  Thumb + pinky out
    ✋ Open palm = SPACE

KEYS:
    ENTER  speak the sentence
    Z      delete last letter
    C      clear everything
    Q      quit
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

import os, sys, time, threading, urllib.request
from collections import deque, Counter


# ─────────────────────────────────────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX    = 0       # change to 1 or 2 if camera not found
HOLD_FRAMES     = 22      # frames to hold gesture before typing
LETTER_COOLDOWN = 1.3     # seconds before same letter can fire again
MIN_CONFIDENCE  = 0.50    # minimum score to show a prediction


# ─────────────────────────────────────────────────────────────────────────────
#  MEDIAPIPE LANDMARK IDs  (these numbers are fixed by MediaPipe)
# ─────────────────────────────────────────────────────────────────────────────
WRIST      = 0
THUMB_TIP  = 4;  THUMB_IP  = 3;  THUMB_MCP  = 2
INDEX_TIP  = 8;  INDEX_PIP = 6;  INDEX_MCP  = 5
MIDDLE_TIP = 12; MIDDLE_PIP= 10; MIDDLE_MCP = 9
RING_TIP   = 16; RING_MCP  = 13
PINKY_TIP  = 20; PINKY_MCP = 17

SKELETON = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ─────────────────────────────────────────────────────────────────────────────
#  COLOURS  (BGR format for OpenCV)
# ─────────────────────────────────────────────────────────────────────────────
TEAL    = (160, 210, 20)
WHITE   = (235, 235, 235)
GREY    = (130, 130, 130)
DKGREY  = (45,  45,  45)
DKBG    = (18,  18,  18)
PANEL   = (25,  25,  25)
GREEN   = (60,  200, 60)
AMBER   = (30,  160, 255)
RED     = (50,  50,  210)
BLACK   = (0,   0,   0)

FONT = cv2.FONT_HERSHEY_DUPLEX
SANS = cv2.FONT_HERSHEY_SIMPLEX
MONO = cv2.FONT_HERSHEY_PLAIN


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — DOWNLOAD HAND MODEL  (26 MB, one time only)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FILE = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

def download_model():
    if os.path.exists(MODEL_FILE):
        return
    print("Downloading hand model (~26 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Download complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — CREATE HAND DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
def create_detector():
    opts = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.50,
    )
    return HandLandmarker.create_from_options(opts)


def detect_hand(frame, detector):
    """Run MediaPipe on one frame. Returns 21 landmarks or None."""
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(image)
    return result.hand_landmarks[0] if result.hand_landmarks else None


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — FINGER GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def dist(a, b):
    """3D distance between two MediaPipe landmarks."""
    return ((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2) ** 0.5

def is_up(lms, tip, mcp):
    """
    Finger is extended when tip is farther from wrist than the knuckle.
    Works at any hand angle.
    """
    return dist(lms[tip], lms[WRIST]) > dist(lms[mcp], lms[WRIST]) * 1.2

def curl(lms, tip, mcp):
    """How curled is a finger? 0.0 = open, 1.0 = fully curled."""
    ratio = dist(lms[tip], lms[WRIST]) / (dist(lms[mcp], lms[WRIST]) * 1.8 + 1e-6)
    return 1.0 - min(max(ratio, 0.0), 1.0)

def tip_dist(lms, a, b):
    """Tip-to-tip distance normalized by hand size."""
    hand_size = dist(lms[WRIST], lms[MIDDLE_MCP]) + 1e-6
    return dist(lms[a], lms[b]) / hand_size

def thumb_palm(lms):
    """How far is the thumb tip from the center of the palm?"""
    px = (lms[INDEX_MCP].x + lms[MIDDLE_MCP].x +
          lms[RING_MCP].x  + lms[PINKY_MCP].x) / 4
    py = (lms[INDEX_MCP].y + lms[MIDDLE_MCP].y +
          lms[RING_MCP].y  + lms[PINKY_MCP].y) / 4
    s  = dist(lms[WRIST], lms[MIDDLE_MCP]) + 1e-6
    return ((lms[THUMB_TIP].x - px)**2 + (lms[THUMB_TIP].y - py)**2)**0.5 / s


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — ASL RECOGNITION  (geometry rules, no ML, no training)
# ─────────────────────────────────────────────────────────────────────────────
def recognize(lms):
    """
    Score each letter against the current hand shape.
    Each rule adds points when features match.
    Highest scorer above MIN_CONFIDENCE wins.
    Returns (letter, confidence) or None.
    """
    # Which fingers are extended?
    idx = is_up(lms, INDEX_TIP,  INDEX_MCP)
    mid = is_up(lms, MIDDLE_TIP, MIDDLE_MCP)
    rng = is_up(lms, RING_TIP,   RING_MCP)
    pnk = is_up(lms, PINKY_TIP,  PINKY_MCP)
    thm = is_up(lms, THUMB_TIP,  THUMB_MCP)

    # How curled is each finger? (0=open, 1=curled)
    ic = curl(lms, INDEX_TIP,  INDEX_MCP)
    mc = curl(lms, MIDDLE_TIP, MIDDLE_MCP)
    rc = curl(lms, RING_TIP,   RING_MCP)
    pc = curl(lms, PINKY_TIP,  PINKY_MCP)

    # Key distances between fingertips
    ti = tip_dist(lms, THUMB_TIP, INDEX_TIP)    # thumb to index
    tm = tip_dist(lms, THUMB_TIP, MIDDLE_TIP)   # thumb to middle
    im = tip_dist(lms, INDEX_TIP, MIDDLE_TIP)   # index to middle
    tp = thumb_palm(lms)                         # thumb to palm center

    scores = {}

    # A — fist with thumb resting beside index (not tucked inside)
    if not idx and not mid and not rng and not pnk:
        s = 0.40
        if ic > 0.55: s += 0.10
        if mc > 0.55: s += 0.10
        if tp > 0.35: s += 0.25   # thumb visible to side
        if not thm:   s += 0.15
        scores['A'] = s

    # B — four fingers straight up, thumb folded in
    if idx and mid and rng and pnk:
        s = 0.45
        if not thm:   s += 0.25
        if tp < 0.55: s += 0.20   # thumb tucked
        if ic < 0.30: s += 0.05
        if mc < 0.30: s += 0.05
        scores['B'] = s

    # C — hand curves into a C, all fingers partially bent
    if (not idx and not mid and
            0.30 < ic < 0.75 and 0.30 < mc < 0.75 and
            0.30 < rc < 0.75 and 0.30 < pc < 0.75):
        s = 0.45
        if ti > 0.45: s += 0.25
        if tp > 0.40: s += 0.20
        if 0.35 < curl(lms, THUMB_TIP, THUMB_MCP) < 0.80: s += 0.10
        scores['C'] = s

    # D — index pointing up, others curl around to touch thumb
    if idx and not mid and not rng and not pnk:
        s = 0.40
        if tm < 0.60: s += 0.30   # others touching thumb
        if ic < 0.30: s += 0.15
        if not thm:   s += 0.15
        scores['D'] = s

    # E — all fingers bent flat, thumb tucked under
    if not idx and not mid and not rng and not pnk:
        s = 0.25
        if tp < 0.45: s += 0.35   # thumb tucked inside
        if ic > 0.55: s += 0.15
        if mc > 0.55: s += 0.15
        if not thm:   s += 0.10
        scores['E'] = s

    # F — index+thumb touch, other three fingers up
    if not idx and mid and rng and pnk:
        s = 0.35
        if ti < 0.40: s += 0.45   # index and thumb touching
        if mc < 0.30: s += 0.10
        if pc < 0.30: s += 0.10
        scores['F'] = s

    # I — only pinky raised
    if not idx and not mid and not rng and pnk:
        s = 0.50
        if not thm:   s += 0.20
        if pc < 0.30: s += 0.20
        if ic > 0.60: s += 0.10
        scores['I'] = s

    # L — index up + thumb out = L shape
    if idx and not mid and not rng and not pnk:
        s = 0.30
        if thm:       s += 0.45   # thumb must also be out
        if tp > 0.60: s += 0.15
        if ic < 0.30: s += 0.10
        scores['L'] = s

    # O — all fingers curve to touch thumb making a circle
    if not idx and not mid and not rng and not pnk:
        s = 0.15
        if ti < 0.38: s += 0.45   # thumb touching index tip
        if tm < 0.50: s += 0.20
        if 0.35 < ic < 0.72: s += 0.10
        if tp > 0.30: s += 0.10
        scores['O'] = s

    # R — index and middle crossed/pressed together
    if idx and mid and not rng and not pnk:
        s = 0.25
        if not thm:   s += 0.20
        if im < 0.18: s += 0.45   # fingers very close = crossed
        if ti > 0.50: s += 0.10
        scores['R'] = s

    # S — tight fist, thumb wrapped across front of fingers
    if not idx and not mid and not rng and not pnk:
        s = 0.25
        if 0.25 < tp < 0.65: s += 0.35   # thumb across front
        if ic > 0.50:         s += 0.15
        if not thm:           s += 0.25
        scores['S'] = s

    # U — index + middle up, held close together
    if idx and mid and not rng and not pnk:
        s = 0.30
        if not thm:   s += 0.15
        if im < 0.22: s += 0.42   # fingers together (U not V)
        if ic < 0.30: s += 0.07
        if mc < 0.30: s += 0.06
        scores['U'] = s

    # V — index + middle up, spread apart (peace sign)
    if idx and mid and not rng and not pnk:
        s = 0.30
        if not thm:   s += 0.15
        if im > 0.22: s += 0.42   # fingers spread (V not U)
        if ic < 0.30: s += 0.07
        if mc < 0.30: s += 0.06
        scores['V'] = s

    # W — index, middle, ring all up
    if idx and mid and rng and not pnk:
        s = 0.55
        if not thm:   s += 0.25
        if ic < 0.30 and mc < 0.30 and rc < 0.30: s += 0.20
        scores['W'] = s

    # X — index finger hooked (bent but not fully curled)
    if not idx and not mid and not rng and not pnk:
        s = 0.15
        if 0.42 < ic < 0.72: s += 0.45   # index hooked
        if mc < 0.45:         s += 0.20
        if tp > 0.50:         s += 0.20
        scores['X'] = s

    # Y — thumb + pinky out, others folded
    if not idx and not mid and not rng and pnk:
        s = 0.35
        if thm:       s += 0.45   # thumb also out
        if pc < 0.30: s += 0.10
        if ic > 0.60: s += 0.10
        scores['Y'] = s

    # SPACE — open palm, all five fingers fully spread
    if idx and mid and rng and pnk and thm:
        s = 0.50
        if ic < 0.25: s += 0.12
        if mc < 0.25: s += 0.12
        if rc < 0.25: s += 0.12
        if pc < 0.25: s += 0.12
        if tp > 0.60: s += 0.12
        scores['SPACE'] = s

    if not scores:
        return None

    best = max(scores, key=scores.get)
    conf = scores[best]
    return (best, min(conf, 1.0)) if conf >= MIN_CONFIDENCE else None


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — SMOOTHER  (prevents flickering between frames)
# ─────────────────────────────────────────────────────────────────────────────
class Smoother:
    def __init__(self, size=10):
        self.buf = deque(maxlen=size)

    def update(self, pred):
        self.buf.append(pred[0] if pred else None)

    def stable(self):
        """Return the letter only if it's the majority vote."""
        valid = [x for x in self.buf if x is not None]
        if not valid or len(valid) < len(self.buf) // 2:
            return None
        top, count = Counter(valid).most_common(1)[0]
        return top if count >= len(self.buf) * 0.50 else None

    def clear(self):
        self.buf.clear()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — TEXT TO SPEECH  (background thread so UI doesn't freeze)
# ─────────────────────────────────────────────────────────────────────────────
def speak(text):
    def _go():
        try:
            import pyttsx3
            e = pyttsx3.init()
            e.setProperty('rate', 155)
            e.say(text)
            e.runAndWait()
        except Exception:
            print("Speech unavailable. Run:  pip install pyttsx3")
    threading.Thread(target=_go, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def draw_box(img, x1, y1, x2, y2, color=PANEL, alpha=0.88):
    """Semi-transparent filled rectangle."""
    y1, y2 = max(0,y1), min(img.shape[0],y2)
    x1, x2 = max(0,x1), min(img.shape[1],x2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color)
    img[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0)

def draw_text(img, s, x, y, scale=0.6, color=WHITE, thick=1, font=FONT):
    """Text with drop shadow."""
    cv2.putText(img, s, (x+1,y+1), font, scale, BLACK, thick+1, cv2.LINE_AA)
    cv2.putText(img, s, (x,  y  ), font, scale, color, thick,   cv2.LINE_AA)

def draw_bar(img, x, y, w, h, pct, color=TEAL):
    """Horizontal progress bar."""
    cv2.rectangle(img, (x,y), (x+w,y+h), DKGREY, -1)
    fill = int(w * min(max(pct,0.0),1.0))
    if fill > 0:
        cv2.rectangle(img, (x,y), (x+fill,y+h), color, -1)
    cv2.rectangle(img, (x,y), (x+w,y+h), (60,60,60), 1)

def draw_ring(img, cx, cy, r, pct, color=TEAL):
    """Circular hold-progress ring."""
    cv2.circle(img, (cx,cy), r, DKGREY, 3, cv2.LINE_AA)
    if pct > 0:
        sweep = int(360 * pct)
        c = GREEN if pct > 0.85 else color
        cv2.ellipse(img, (cx,cy), (r,r), -90, 0, sweep, c, 5, cv2.LINE_AA)

def draw_skeleton(img, lms, w, h):
    """Hand skeleton overlay on camera image."""
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in lms]
    for a,b in SKELETON:
        cv2.line(img, pts[a], pts[b], (70,70,70), 2, cv2.LINE_AA)
    for i,p in enumerate(pts):
        if i == INDEX_TIP:
            cv2.circle(img, p, 12, TEAL,  -1, cv2.LINE_AA)
            cv2.circle(img, p, 15, DKGREY, 2, cv2.LINE_AA)
        elif i == WRIST:
            cv2.circle(img, p, 6, GREY, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, p, 4, (190,190,190), -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 8 — MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    download_model()

    # ── Open camera ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}.")
        print("Change CAMERA_INDEX at the top of this file to 1 or 2.")
        sys.exit(1)

    # Use the camera's native resolution (640x480 from your test)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Read one frame to confirm the actual size
    ret, test_frame = cap.read()
    if not ret:
        print("ERROR: Camera opened but cannot read frames.")
        sys.exit(1)

    CAM_H, CAM_W = test_frame.shape[:2]
    print(f"Camera running at {CAM_W}x{CAM_H}")

    # ── Window layout  (fits 640x480 camera + 300px sidebar) ─────────────────
    SIDE_W  = 300          # sidebar width
    BOT_H   = 80           # text bar height at bottom
    WIN_W   = CAM_W + SIDE_W   # 940
    WIN_H   = CAM_H + BOT_H    # 560
    SX      = CAM_W             # sidebar starts here (x = 640)

    cv2.namedWindow("SignVision AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SignVision AI", WIN_W, WIN_H)

    # ── App state ─────────────────────────────────────────────────────────────
    detector = create_detector()
    smoother = Smoother(size=10)

    typed    = ""        # text the user has typed
    hold_ltr = ""        # letter currently being held
    hold_n   = 0         # consecutive frames held
    last_ltr = ""        # last letter added
    last_t   = 0.0       # when it was added

    notif    = ""        # notification message
    notif_t  = 0.0       # when it expires

    fps      = 28.0
    t_prev   = time.perf_counter()

    GUIDE = [
        ("A", "Fist, thumb to side"),
        ("B", "4 fingers up"),
        ("C", "Curved C shape"),
        ("D", "Index up, circle"),
        ("I", "Only pinky up"),
        ("L", "Index+thumb = L"),
        ("O", "All fingers = O"),
        ("U", "2 fingers, together"),
        ("V", "Peace sign, spread"),
        ("W", "3 fingers up"),
        ("Y", "Thumb + pinky out"),
        ("Palm", "Open hand = SPACE"),
    ]

    print("SignVision AI — ready!")
    print("Hold any ASL letter gesture still to type it.\n")

    # ─────────────────────────────────────────────────────────────────────────
    #  MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)   # mirror horizontally

        # FPS
        t_now  = time.perf_counter()
        fps    = fps * 0.90 + (1.0 / max(t_now - t_prev, 1e-6)) * 0.10
        t_prev = t_now

        # Detect hand + recognize letter
        lms      = detect_hand(frame, detector)
        hand_ok  = lms is not None
        raw_pred = recognize(lms) if hand_ok else None

        smoother.update(raw_pred)
        stable = smoother.stable()
        conf   = raw_pred[1] if raw_pred else 0.0

        if not hand_ok:
            smoother.clear()
            hold_ltr = ""
            hold_n   = 0

        # Hold-to-type logic
        progress = 0.0
        if stable:
            if stable == hold_ltr:
                hold_n += 1
            else:
                hold_ltr = stable
                hold_n   = 1

            progress = min(hold_n / HOLD_FRAMES, 1.0)
            cd_ok = (stable != last_ltr or
                     time.time() - last_t > LETTER_COOLDOWN)

            if hold_n >= HOLD_FRAMES and cd_ok:
                hold_n   = 0
                last_ltr = stable
                last_t   = time.time()
                if stable == "SPACE":
                    typed  += " "
                    notif   = "[ SPACE ]"
                else:
                    typed  += stable
                    notif   = f"Typed:  {stable}"
                notif_t = time.time() + 1.0
        else:
            hold_ltr = ""
            hold_n   = 0

        # ── Build canvas ──────────────────────────────────────────────────────
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        canvas[:] = DKBG

        # Camera image (top-left)
        cam = frame.copy()
        cam = (cam.astype(np.float32) * 0.78).astype(np.uint8)   # slight dim

        # Draw hand on camera image
        if hand_ok:
            draw_skeleton(cam, lms, CAM_W, CAM_H)
            tx = int(lms[INDEX_TIP].x * CAM_W)
            ty = int(lms[INDEX_TIP].y * CAM_H)
            draw_ring(cam, tx, ty, 30, progress)

        canvas[0:CAM_H, 0:CAM_W] = cam

        # Small letter badge on camera (top-left corner)
        if stable and hand_ok:
            lbl = "SPC" if stable == "SPACE" else stable
            draw_box(canvas, 8, 8, 100, 82, DKBG, 0.82)
            cv2.rectangle(canvas, (8,8), (100,82), TEAL, 2)
            ltr_col = GREEN if progress > 0.85 else TEAL
            draw_text(canvas, lbl, 18, 72, 1.8, ltr_col, thick=3)

        # Divider lines
        cv2.line(canvas, (SX, 0),    (SX, WIN_H),  (50,50,50), 1)
        cv2.line(canvas, (0,  CAM_H),(WIN_W, CAM_H),(50,50,50), 1)

        # ── Right sidebar ─────────────────────────────────────────────────────
        draw_box(canvas, SX, 0, WIN_W, WIN_H, PANEL, 0.97)

        # Title
        draw_text(canvas, "Sign", SX+10, 34, 0.80, TEAL,  thick=2)
        draw_text(canvas, "Vision", SX+78, 34, 0.80, WHITE, thick=2)
        cv2.line(canvas, (SX+10,44),(WIN_W-10,44),(50,50,50),1)

        # Status dot
        dot_c = GREEN if hand_ok else RED
        cv2.circle(canvas, (SX+20, 65), 7, dot_c, -1, cv2.LINE_AA)
        draw_text(canvas, "Hand detected" if hand_ok else "Show your hand",
                  SX+36, 70, 0.48, WHITE if hand_ok else GREY)

        # FPS
        fps_c = GREEN if fps>22 else AMBER if fps>14 else RED
        draw_text(canvas, f"{fps:.0f}fps", WIN_W-58, 70, 0.46, fps_c)

        cv2.line(canvas,(SX+10,82),(WIN_W-10,82),(50,50,50),1)

        # Big letter
        if stable and hand_ok:
            lbl = "SPC" if stable=="SPACE" else stable
            lc  = GREEN if progress>0.85 else TEAL
            cv2.putText(canvas, lbl, (SX+52, 200), FONT, 4.0, BLACK, 12, cv2.LINE_AA)
            cv2.putText(canvas, lbl, (SX+50, 198), FONT, 4.0, lc,    7,  cv2.LINE_AA)

            # Confidence bar
            draw_text(canvas, f"Conf: {conf*100:.0f}%",
                      SX+10, 218, 0.45, GREY)
            draw_bar(canvas, SX+10, 224, SIDE_W-20, 8, conf,
                     GREEN if conf>0.80 else AMBER)

            # Hold bar
            hp  = int(progress*100)
            hl  = "Typed!" if progress>=1.0 else f"Hold... {hp}%"
            hc  = GREEN    if progress>=1.0 else (AMBER if hp>50 else GREY)
            draw_text(canvas, hl, SX+10, 250, 0.50, hc)
            draw_bar(canvas, SX+10, 256, SIDE_W-20, 7, progress,
                     GREEN if progress>0.85 else TEAL)
        else:
            draw_text(canvas, "-", SX+120, 190, 3.5, DKGREY, thick=5)
            msg = "No gesture" if hand_ok else "Waiting..."
            draw_text(canvas, msg, SX+50, 225, 0.48, (65,65,65))

        cv2.line(canvas,(SX+10,272),(WIN_W-10,272),(50,50,50),1)

        # Gesture guide
        draw_text(canvas, "Gesture Guide", SX+10, 292, 0.52, GREY)
        gy = 312
        for lk, desc in GUIDE:
            lk_c   = TEAL  if lk==stable else (70,70,70)
            desc_c = WHITE if lk==stable else (88,88,88)
            draw_text(canvas, lk,   SX+10, gy, 0.46, lk_c,  font=FONT)
            draw_text(canvas, desc, SX+58, gy, 0.40, desc_c, font=SANS)
            gy += 20

        cv2.line(canvas,(SX+10,gy+4),(WIN_W-10,gy+4),(50,50,50),1)
        gy += 18

        # Shortcuts
        for key_s, act in [("ENTER","speak"),("Z","delete"),
                            ("C","clear"),("Q","quit")]:
            draw_text(canvas, key_s, SX+10, gy, 0.44, TEAL, font=MONO)
            draw_text(canvas, act,   SX+75, gy, 0.42, (80,80,80), font=MONO)
            gy += 18

        # ── Bottom text bar ───────────────────────────────────────────────────
        BY = CAM_H
        draw_box(canvas, 0, BY, CAM_W, WIN_H, DKBG, 0.93)
        cv2.line(canvas,(0,BY),(CAM_W,BY),(55,55,55),1)

        draw_text(canvas, "TEXT", 12, BY+28, 0.50, GREY)

        show = typed + ">"
        max_ch = (CAM_W - 90) // 14
        if len(show) > max_ch:
            show = "..." + show[-max_ch:]
        draw_text(canvas, show, 80, BY+28, 0.70, WHITE)

        wc = len(typed.split()) if typed.strip() else 0
        draw_text(canvas, f"{len(typed)} chars  {wc} words  |  ENTER=speak  Z=delete  C=clear",
                  12, BY+58, 0.38, (65,65,65), font=MONO)

        # Notification
        if notif and time.time() < notif_t:
            (tw,th),_ = cv2.getTextSize(notif, FONT, 0.75, 2)
            nx = max(10, (CAM_W-tw)//2)
            ny = BY - 22
            draw_box(canvas, nx-18, ny-th-8, nx+tw+18, ny+10, (0,50,60), 0.90)
            draw_text(canvas, notif, nx, ny, 0.75, TEAL, thick=2)

        # ── Show frame ────────────────────────────────────────────────────────
        cv2.imshow("SignVision AI", canvas)

        # ── Keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key == 13:
            t = typed.strip()
            if t:
                speak(t)
                notif   = f'Speaking...'
                notif_t = time.time() + 2.0
        elif key == ord(' '):
            typed  += " "
            notif   = "[ SPACE ]"
            notif_t = time.time() + 0.6
        elif key in (ord('z'), ord('Z')):
            typed   = typed[:-1]
            notif   = "Deleted"
            notif_t = time.time() + 0.5
        elif key in (ord('c'), ord('C')):
            typed    = ""
            hold_ltr = ""
            hold_n   = 0
            smoother.clear()
            notif    = "Cleared"
            notif_t  = time.time() + 0.7

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()