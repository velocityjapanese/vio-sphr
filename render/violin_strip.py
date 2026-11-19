"""Per-frame violin view: bowed strings at a held-instrument angle.

The fingerboard is drawn flat in local space (four silver strings on ebony,
gliding finger dots, a bow crossing near the bridge), then the whole layer is
rotated so the instrument reads as held: scroll upper left, body lower right.
Finger stops use real first-position geometry (the speaking length halves every
twelve semitones), the bow tracks the melody, string letters G/D/A/E sit by the
nut, and a small badge names the sounding note and the finger playing it.
"""
import math
import queue
import random
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import FPS, H, W
from render.scene import key_table, load_font, make_spark_sprite, palette
from render.score_layer import PANEL_H, PANEL_W, PANEL_X, PANEL_Y

# Low string first. G3, D4, A4, E5 are spaced seven semitones apart, so the
# highest open string at or below the pitch keeps every stop in first position.
STRINGS = (("G", 55), ("D", 62), ("A", 69), ("E", 76))
STRING_Y = (649, 611, 573, 535)       # local pixels, G bottom .. E top
MID = 592.0                           # centreline of the string band
LOCAL_W, LOCAL_H = 3400, 1200         # scroll to tail end, plus the bout width
NECK_Y0, NECK_Y1 = 500, 684
NUT_X, BRIDGE_X, BOW_X = 738, 2466, 2310
BOARD_X0, BOARD_END_X = 700, 2232     # fingerboard stops well short of the bridge
BODY_X0 = 1439.0                      # body shoulder joins the neck here
BOW_X_MIN, BOW_X_MAX = 2090, 2415
BOW_EXCURSION = 46.0
BOW_ANGLE_RANGE = 14

# The plate is a real violin: a CC0 photograph of "The Ex Kurtz" violin from
# The Metropolitan Museum of Art (DP147092), warped so the plate edge at the
# neck joint lands on BODY_X0 and the top of the bridge on BRIDGE_X. It gives
# the true outline, purfling, f-holes, bridge and tailpiece; the ebony board,
# strings, scroll and pegs are drawn on top in local space.
BODY_PHOTO = Path(__file__).resolve().parents[1] / "assets" / "violin_body.jpg"
PHOTO_SHOULDER_Y = 378.0     # plate edge at the neck joint, photo pixels
PHOTO_BRIDGE_Y = 1022.0      # top edge of the bridge, where the strings end
PHOTO_CENTER_X = 655.0       # instrument centreline
# String lines between the board end and the bridge, inpainted out so the
# drawn strings (which drive finger stops and bowing) are the only ones.
PHOTO_STRINGS = (
    ((598.5, 868.0), (598.0, 1022.0)),
    ((631.0, 868.0), (632.5, 1022.0)),
    ((665.0, 868.0), (667.5, 1022.0)),
    ((707.5, 868.0), (712.5, 1022.0)),
)

# Whole-layer placement: clockwise tilt so the scroll sits up-left and the
# body runs to the lower right; scaled so the full instrument fits in frame.
TILT_DEG = 36.0
TILT_SCALE = 0.43
ANCHOR_LOCAL = (float(NUT_X), MID)
ANCHOR_SCREEN = (390.0, 292.0)

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FINGER_NAMES = ("1st", "2nd", "3rd", "4th")


_BODY_CACHE = {}


def _body_photo():
    """The CC0 violin photo warped into the local body footprint (2x)."""
    cached = _BODY_CACHE.get("warp")
    if cached is not None:
        return cached
    img = cv2.imread(str(BODY_PHOTO), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"violin body photo missing: {BODY_PHOTO}")
    img = img[:, :, ::-1]
    src_h, src_w = img.shape[:2]

    mn = img.min(axis=2)
    mx = img.max(axis=2)
    whiteish = ((mn >= 235) & ((mx - mn) <= 14)).astype(np.uint8)
    flood = whiteish.copy()
    ffmask = np.zeros((src_h + 2, src_w + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 2)
    fg = (flood != 2).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    fg = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)

    cut = np.zeros((src_h, src_w), np.uint8)
    for (xa, ya), (xb, yb) in PHOTO_STRINGS:
        cv2.line(cut, (int(round(xa)), int(round(ya))),
                 (int(round(xb)), int(round(yb))), 255, 11)
    img = cv2.inpaint(img, cv2.dilate(cut, np.ones((5, 5), np.uint8)), 6,
                      cv2.INPAINT_TELEA)

    S = 2
    k = (BRIDGE_X - BODY_X0) / (PHOTO_BRIDGE_Y - PHOTO_SHOULDER_Y)
    size = (LOCAL_W * S, LOCAL_H * S)
    # Photo pixels to local 2x pixels: local +x runs shoulder to tail, +y runs
    # across the plate. warpAffine inverts internally, so this is the forward map.
    fwd = np.array([[0.0, k * S, (BODY_X0 - k * PHOTO_SHOULDER_Y) * S],
                    [k * S, 0.0, (MID - k * PHOTO_CENTER_X) * S]])
    rgb = cv2.warpAffine(img, fwd, size, flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    alpha = cv2.warpAffine(fg * 255, fwd, size, flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha[:, :int((BODY_X0 - 4) * S)] = 0
    alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.1 * S)

    rgb = rgb.astype(np.float32) * 0.74
    out = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha])
    _BODY_CACHE["warp"] = out
    return out


def make_violin_base(palette_name):
    pal = palette(palette_name)
    S = 2
    w, h = LOCAL_W * S, LOCAL_H * S
    accent = np.array(pal["accent"], dtype=np.float32)
    y0, y1 = NECK_Y0 * S, NECK_Y1 * S
    be = BOARD_END_X * S

    canvas = _body_photo().copy()
    plate = canvas[:, :, :3].astype(np.float32)
    canvas[:, :, :3] = np.clip(plate + accent[None, None, :] * 0.02,
                               0, 255).astype(np.uint8)

    # Ebony fingerboard from the nut, stopping short of the bridge.
    b0 = BOARD_X0 * S
    bw = be - b0
    neck_h = y1 - y0
    gt = np.linspace(0.0, 1.0, neck_h, dtype=np.float32)[:, None, None]
    ebony = (np.array((36, 32, 36), dtype=np.float32)[None, None, :] * (1.0 - gt)
             + np.array((11, 10, 13), dtype=np.float32)[None, None, :] * gt)
    canvas[y0:y1, b0:be, :3] = np.clip(ebony, 0, 255).astype(np.uint8)
    canvas[y0:y1, b0:be, 3] = 255
    canvas[y0:y0 + 2, b0:be, :3] = (74, 70, 82)
    canvas[y0 + 2:y0 + 5, b0:be, :3] = (40, 37, 42)
    sheen_b = np.exp(-((np.arange(neck_h) - neck_h * 0.30) / 26.0) ** 2)[:, None, None]
    canvas[y0:y1, b0:be, :3] = np.clip(
        canvas[y0:y1, b0:be, :3].astype(np.float32)
        + accent[None, None, :] * sheen_b * 0.035, 0, 255).astype(np.uint8)
    grng = np.random.default_rng(7)
    grain_b = grng.normal(0, 1.6, (neck_h, bw, 1)).astype(np.float32)
    streak = (np.sin(np.arange(bw, dtype=np.float32) * 0.011) * 1.1
              + np.sin(np.arange(bw, dtype=np.float32) * 0.0037) * 2.0)[None, :, None]
    canvas[y0:y1, b0:be, :3] = np.clip(
        canvas[y0:y1, b0:be, :3].astype(np.float32) + grain_b + streak,
        0, 255).astype(np.uint8)

    # Bone nut; the pegbox continues left toward the scroll.
    nx = NUT_X * S
    canvas[y0:y1, nx - 9 * S:nx + 9 * S, :3] = (206, 198, 176)
    canvas[y0:y1, nx + 9 * S:nx + 12 * S, :3] = (120, 112, 96)
    canvas[y0:y1, nx - 13 * S:nx - 9 * S, :3] = (58, 52, 48)

    # Rounded fingerboard end cap (the plate itself comes from the photo).
    over = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    d.ellipse((be - 30 * S, y0, be + 12 * S, y1), fill=(30, 27, 32, 255))
    d.arc((be - 30 * S, y0, be + 12 * S, y1), start=-90, end=90,
          fill=(120, 106, 96, 150), width=2 * S)

    base_img = Image.alpha_composite(Image.fromarray(canvas, "RGBA"), over)
    canvas = np.array(base_img, dtype=np.uint8)

    # Strings over the plate, wound G through plain-steel E, with a soft shadow.
    string_cols = ((150, 118, 84), (148, 142, 130), (168, 170, 176), (198, 202, 208))
    xa, xb = NUT_X * S, BRIDGE_X * S
    for i, row in enumerate(STRING_Y):
        yy2 = row * S
        col = np.array(string_cols[i], dtype=np.float32)
        canvas[yy2 - 3:yy2 + 3, xa:xb, :3] = col.astype(np.uint8)
        canvas[yy2 - 3, xa:xb, :3] = np.clip(col * 1.35, 0, 255).astype(np.uint8)
        canvas[yy2 + 2, xa:xb, :3] = np.clip(col * 0.45, 0, 255).astype(np.uint8)
        shadow = canvas[yy2 + 4:yy2 + 7, xa:xb, :3]
        shadow[:] = (shadow.astype(np.float32) * 0.55).astype(np.uint8)

    # Pegbox, tuning pegs, and the scroll left of the nut.
    base_img = Image.fromarray(canvas, "RGBA")
    pegs = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(pegs)
    nx2 = NUT_X * S
    px1, px0 = nx2 - 12 * S, 306 * S
    d.polygon([(px1, (MID - 92) * S), (px1, (MID + 92) * S),
               (px0, (MID + 76) * S), (px0, (MID - 76) * S)],
              fill=(32, 20, 15, 255))
    d.polygon([(px1, (MID - 92) * S), (px0, (MID - 76) * S),
               (px0, (MID - 68) * S), (px1, (MID - 82) * S)],
              fill=(62, 40, 28, 255))

    # Four tuning pegs, two to each side, shafts running out of the box.
    for i, cx2 in enumerate((652, 566, 480, 394)):
        top = i % 2 == 0
        if top:
            sy0, sy1 = (MID - 150) * S, (MID - 62) * S
            y_head = (MID - 152) * S
        else:
            sy0, sy1 = (MID + 62) * S, (MID + 150) * S
            y_head = (MID + 152) * S
        d.rounded_rectangle((cx2 * S - 12 * S, sy0, cx2 * S + 12 * S, sy1),
                            radius=6 * S, fill=(42, 27, 20, 255))
        d.line(((cx2 * S - 7 * S, sy0), (cx2 * S - 7 * S, sy1)),
               fill=(76, 50, 35, 255), width=3 * S)
        d.ellipse((cx2 * S - 20 * S, y_head - 17 * S, cx2 * S + 20 * S, y_head + 17 * S),
                  fill=(28, 19, 15, 255))
        d.ellipse((cx2 * S - 14 * S, y_head - 11 * S, cx2 * S + 9 * S, y_head + 8 * S),
                  fill=(62, 42, 30, 255))

    # Scroll: turned volute past the end of the pegbox.
    scx, scy, sr = 226 * S, MID * S, 94 * S
    d.ellipse((scx - sr * 0.94, scy - sr, scx + sr * 0.94, scy + sr),
              fill=(36, 23, 17, 255))
    d.arc((scx - sr * 0.94, scy - sr, scx + sr * 0.94, scy + sr),
          start=110, end=320, fill=(120, 76, 48, 190), width=4 * S)
    spiral = []
    for k in range(160):
        u = k / 159.0
        ang = math.radians(210.0) + u * 2 * math.pi * 2.05
        rr = (77.0 - 58.0 * u) * S
        spiral.append((scx + 0.94 * rr * math.cos(ang), scy + rr * math.sin(ang)))
    d.line(spiral, fill=(14, 9, 7, 240), width=5 * S, joint="curve")
    er = 11 * S
    d.ellipse((scx - er, scy - er, scx + er, scy + er), fill=(10, 7, 6, 255))
    d.arc((scx - 22 * S, scy - 22 * S, scx + 22 * S, scy + 22 * S),
          start=130, end=250, fill=(98, 62, 40, 180), width=3 * S)
    instrument = Image.alpha_composite(base_img, pegs)

    # Drop shadow from the instrument silhouette, offset down-right.
    alpha = instrument.getchannel("A")
    blur = np.array(alpha.filter(ImageFilter.GaussianBlur(15 * S)), dtype=np.float32)
    shifted = np.zeros_like(blur)
    dx, dy = 7 * S, 16 * S
    shifted[dy:, dx:] = blur[:h - dy, :w - dx]
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow.putalpha(Image.fromarray(np.clip(shifted * 0.5, 0, 255).astype(np.uint8)))
    final = Image.alpha_composite(shadow, instrument)
    pil = final.resize((LOCAL_W, LOCAL_H), Image.Resampling.LANCZOS)
    return np.array(pil, dtype=np.uint8)


_BOW_CACHE = None


def _make_bow_sprites():
    """Pre-rendered bow at each tilt angle, composited per frame."""
    global _BOW_CACHE
    if _BOW_CACHE is not None:
        return _BOW_CACHE
    S = 2
    w, h = 300 * S, 760 * S
    sprites = []
    for ang in range(-BOW_ANGLE_RANGE, BOW_ANGLE_RANGE + 1):
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        cx, cy = w // 2, h // 2
        half = 350 * S
        dx = int(math.tan(math.radians(ang)) * half)
        x0, y0 = cx - dx, cy - half
        x1, y1 = cx + dx, cy + half
        off = 26 * S
        # Hair sits on the string side of the stick.
        d.line([(x0 - off, y0 + 10 * S), (x1 - off, y1 - 10 * S)],
               fill=(246, 242, 234, 235), width=8 * S)
        d.line([(x0 - off - 1 * S, y0 + 16 * S), (x1 - off - 1 * S, y1 - 16 * S)],
               fill=(255, 252, 246, 140), width=3 * S)
        # Pernambuco stick with a lit edge, dark frog at the lower end.
        d.line([(x0, y0), (x1, y1)], fill=(112, 74, 46, 255), width=16 * S)
        d.line([(x0 - 3 * S, y0), (x1 - 3 * S, y1)], fill=(58, 36, 24, 255), width=4 * S)
        d.line([(x0 + 4 * S, y0), (x1 + 4 * S, y1)], fill=(196, 146, 90, 255), width=5 * S)
        d.line([(x1 - 1 * S, y1 - 84 * S), (x1 - 1 * S, y1 - 5 * S)],
               fill=(28, 20, 16, 255), width=28 * S)
        d.line([(x0 + 1 * S, y0 + 4 * S), (x0 + 1 * S, y0 + 62 * S)],
               fill=(232, 218, 190, 255), width=7 * S)
        arr = np.asarray(img.resize((300, 760), Image.Resampling.LANCZOS), dtype=np.float32)
        sprites.append((arr[:, :, :3], arr[:, :, 3] / 255.0))
    _BOW_CACHE = sprites
    return sprites


def _string_index(pitch):
    for i in (3, 2, 1, 0):
        if pitch >= STRINGS[i][1]:
            return i
    return 0


def _stop_x(pitch, si):
    """Fretless stop position: one octave up is exactly half the string."""
    semis = max(0, pitch - STRINGS[si][1])
    return BRIDGE_X - (BRIDGE_X - NUT_X) * (2.0 ** (-semis / 12.0))


def _clip_add(roi, add):
    roi[:] = np.minimum(roi.astype(np.uint16) + add.astype(np.uint16), 255).astype(np.uint8)


def _glow_sprite(size):
    """Tighter falloff than the scene sprites: a soft halo, no flat core."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c
    return np.clip(1.0 - r, 0, 1) ** 2.6


class ViolinStripGenerator:
    def __init__(self, section, notes, seed=0):
        pal = palette(section["palette"])
        self.base = make_violin_base(section["palette"])
        self.keys = {k["pitch"]: k for k in key_table()}
        self.key_color = np.array(pal["key_glow"], dtype=np.float32)
        self.accent = np.array(pal["accent"], dtype=np.float32)
        self.spark_color = np.array(pal["accent"], dtype=np.float32)
        self.finger_glow = _glow_sprite(64)
        self.contact_glow = _glow_sprite(44)
        self.spark = make_spark_sprite(14)
        self.bow_sprites = _make_bow_sprites()
        glow_img = Image.fromarray((_glow_sprite(96) * 255).astype(np.uint8), "L")
        self.body_glow = (np.asarray(glow_img.resize((760, 260), Image.Resampling.BICUBIC),
                                     dtype=np.float32) / 255.0)
        self.M = cv2.getRotationMatrix2D(ANCHOR_LOCAL, -TILT_DEG, TILT_SCALE)
        self.M[0, 2] += ANCHOR_SCREEN[0] - ANCHOR_LOCAL[0]
        self.M[1, 2] += ANCHOR_SCREEN[1] - ANCHOR_LOCAL[1]

        self.string_labels = self._make_string_labels()

        self.badge_sprites = {}
        self.badge_pitch = None
        self.badge_si = 0
        self.badge_alpha = 0.0
        self.badge_since = -10.0
        self.badge_hold = 0.4

        self.section = section
        self.rng = random.Random(seed * 7919 + section["index"] * 104729)

        sec_start, sec_end = section["start"], section["end"]
        self.notes = [n for n in notes if n["start"] <= sec_end and n["end"] >= sec_start - 0.4]
        self.violin_notes = [n for n in self.notes if n["hand"] == "R"]
        self.active = []
        self.next_idx = 0
        self.particles = []

        self.finger_x = [None, None, None, None]
        self.finger_target = [None, None, None, None]
        self.finger_level = [0.0, 0.0, 0.0, 0.0]
        self.vib = [0.0, 0.0, 0.0, 0.0]

        self.bow_dir = 1.0
        self.bow_tilt = 0.0
        self.bow_y = 0.0
        self.bow_x = float(BOW_X)
        self.bow_target_x = float(BOW_X)
        self.stroke_t0 = -10.0
        self.stroke_dur = 0.0

        self.frame_events = {}
        for n in self.violin_notes:
            f = int(round((n["start"] - sec_start) * FPS))
            if 0 <= f < section["frames"]:
                self.frame_events.setdefault(f, []).append(n)

    def frames(self):
        sec = self.section
        frames = sec["frames"]
        dt = 1.0 / FPS
        for i in range(frames):
            t = sec["start"] + i * dt
            layer = self.base.copy()
            violin, piano = self._pressed(t)
            self._on_onsets(i, t)
            self._update_bow(t, violin)
            self._draw_fingers(layer, t, violin)
            self._draw_bow(layer, t, violin)
            self._draw_body(layer, piano)
            self._update_particles(layer, dt)
            frame = cv2.warpAffine(layer, self.M, (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))
            for sprite, x, y in self.string_labels:
                self._blend(frame, x, y, sprite[0], sprite[1], 0.58)
            self._update_badge(t, violin)
            self._draw_badge(frame)
            yield frame.tobytes()

    def _project(self, x, y):
        return (self.M[0, 0] * x + self.M[0, 1] * y + self.M[0, 2],
                self.M[1, 0] * x + self.M[1, 1] * y + self.M[1, 2])

    def _make_string_labels(self):
        out = []
        font = load_font("sans", 10)
        for name, row in zip((s[0] for s in STRINGS), STRING_Y):
            img = Image.new("RGBA", (22, 14), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.text((6, 0), name, font=font, fill=(225, 233, 248, 185))
            arr = np.asarray(img, dtype=np.float32)
            sprite = (arr[:, :, :3], arr[:, :, 3] / 255.0)
            sx, sy = self._project(NUT_X + 46, row - 24)
            out.append((sprite, int(round(sx - 11)), int(round(sy - 7))))
        return out

    def _pressed(self, t):
        while self.next_idx < len(self.notes) and self.notes[self.next_idx]["start"] <= t:
            self.active.append(self.notes[self.next_idx])
            self.next_idx += 1
        self.active = [n for n in self.active if n["end"] + 0.25 >= t]
        violin = {}
        piano = 0.0
        for n in self.active:
            if t < n["start"]:
                continue
            rel = t - n["start"]
            attack = min(1.0, rel / 0.012)
            sustain = 0.5 + 0.5 * math.exp(-rel / 0.55)
            value = attack * sustain * (n["vel"] / 127.0)
            if t > n["end"]:
                value *= math.exp(-(t - n["end"]) / 0.13)
            if n["hand"] == "R":
                if value > violin.get(n["pitch"], 0.0):
                    violin[n["pitch"]] = value
            else:
                piano = max(piano, value)
        return violin, piano

    def _on_onsets(self, frame_index, t):
        for n in self.frame_events.get(frame_index, ()):
            self.bow_dir = -self.bow_dir
            self.stroke_t0 = t
            self.stroke_dur = max(0.30, min(1.6, n["end"] - n["start"]))
            si = _string_index(n["pitch"])
            self.vib[si] = min(1.25, self.vib[si] + 0.55 + n["vel"] / 220.0)
            self._spawn(n)

    def _update_bow(self, t, violin):
        energy = max(violin.values(), default=0.0)
        if violin:
            # The bow tracks the melody: low stops draw it nearer the board,
            # high stops nearer the bridge, as a player shifts contact point.
            pitch = max(violin, key=violin.get)
            key = self.keys.get(pitch)
            if key is not None:
                target = 2120.0 + (key["center"] - 880.0) * 0.62
                self.bow_target_x = max(BOW_X_MIN, min(BOW_X_MAX, target))
        self.bow_x += (self.bow_target_x - self.bow_x) * 0.08
        if self.stroke_dur > 0:
            phase = min(1.0, max(0.0, (t - self.stroke_t0) / self.stroke_dur))
            target = self.bow_dir * (phase * 2.0 - 1.0) * BOW_EXCURSION
        else:
            target = 0.0
        self.bow_y += (target - self.bow_y) * 0.35
        sway = 5.0 * math.sin(t * 2.1) + 2.5 * math.sin(t * 5.3)
        self.bow_tilt = (-8.0 * self.bow_dir * (0.35 + 0.65 * min(1.0, energy))
                         + sway * 0.45)

    def _draw_fingers(self, strip, t, violin):
        string_val = [0.0, 0.0, 0.0, 0.0]
        string_pitch = [None, None, None, None]
        for pitch, val in violin.items():
            si = _string_index(pitch)
            if val > string_val[si]:
                string_val[si] = val
                string_pitch[si] = pitch

        for si in range(4):
            if string_pitch[si] is not None:
                self.finger_target[si] = _stop_x(string_pitch[si], si)
            self.finger_level[si] = max(string_val[si], self.finger_level[si] * 0.90)
            if self.finger_target[si] is not None:
                if self.finger_x[si] is None:
                    self.finger_x[si] = self.finger_target[si]
                else:
                    self.finger_x[si] += (self.finger_target[si] - self.finger_x[si]) * 0.35

            fx = self.finger_x[si]
            row = STRING_Y[si]
            lvl = self.finger_level[si]
            if fx is None or lvl <= 0.015:
                continue

            # An open string has no stopped finger; a softer glow at the nut
            # marks it without lighting the pegbox like a lamp.
            open_str = (string_pitch[si] is not None
                        and string_pitch[si] <= STRINGS[si][1])
            glow = 0.55 if open_str else 1.0

            self._add_column(strip, int(round(fx)), NECK_Y0 + 2, row - 8,
                             self.key_color, (0.07 + 0.32 * lvl) * glow)
            self._stamp(strip, fx, row, self.finger_glow, self.key_color, 0.55 * lvl * glow)
            self._stamp(strip, fx, row, self.spark, self.accent, 0.85 * lvl * glow)

            x0 = max(0, int(fx) - 4)
            x1 = min(LOCAL_W, int(fx) + 4)
            band = np.broadcast_to((self.accent * lvl * 0.55 * glow).astype(np.uint16),
                                   (3, x1 - x0, 3))
            _clip_add(strip[row - 1:row + 2, x0:x1, :3], band)

            self.vib[si] = max(string_val[si], self.vib[si] * 0.93)
            if self.vib[si] > 0.02 and BRIDGE_X - fx > 40:
                self._draw_wave(strip, si, fx, BRIDGE_X, t, self.vib[si])

            # Light where the bow hair meets the string.
            if string_val[si] > 0.02:
                self._stamp(strip, int(round(self.bow_x)) - 24, row, self.contact_glow,
                            self.key_color, 0.18 * string_val[si])

    def _draw_wave(self, strip, si, xa, xb, t, energy):
        y_base = STRING_Y[si]
        xs = np.arange(max(0, int(xa)), min(LOCAL_W, int(xb)))
        if xs.size < 6:
            return
        seg = (xs - xa) / max(1.0, xb - xa)
        env = np.sin(np.pi * seg)
        amp = 2.8 * energy
        ys = y_base + amp * env * np.sin(2 * np.pi * (xs / 22.0 - t * 3.2))
        yi = np.clip(np.round(ys).astype(int), 0, LOCAL_H - 1)
        col = np.clip(self.key_color * (0.55 + 0.45 * energy), 0, 255).astype(np.uint16)
        cur = strip[yi, xs, :3].astype(np.uint16)
        strip[yi, xs, :3] = np.minimum(cur + col[None, :], 255).astype(np.uint8)
        for dy in (-1, 1):
            yn = np.clip(yi + dy, 0, LOCAL_H - 1)
            cur = strip[yn, xs, :3].astype(np.uint16)
            strip[yn, xs, :3] = np.minimum(
                cur + (col * 0.4).astype(np.uint16)[None, :], 255).astype(np.uint8)

    def _draw_bow(self, strip, t, violin):
        energy = min(1.0, max(violin.values(), default=0.0))
        idx = max(0, min(2 * BOW_ANGLE_RANGE,
                         int(round(self.bow_tilt)) + BOW_ANGLE_RANGE))
        rgb, alpha = self.bow_sprites[idx]
        a_mul = 0.14 + 0.86 * energy
        sh, sw = rgb.shape[0], rgb.shape[1]
        ox = int(round(self.bow_x - sw * 0.5 + 6 * math.sin(t * 1.1)))
        oy = int(round(MID - sh * 0.5 + self.bow_y))
        cx0, cy0 = max(0, ox), max(0, oy)
        cx1, cy1 = min(LOCAL_W, ox + sw), min(LOCAL_H, oy + sh)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sub_rgb = rgb[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox]
        sub_a = (alpha[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox] * a_mul)[:, :, None]
        roi = strip[cy0:cy1, cx0:cx1]
        roi[:, :, :3] = np.clip(roi[:, :, :3] * (1.0 - sub_a) + sub_rgb * sub_a,
                                0, 255).astype(np.uint8)
        roi[:, :, 3] = np.maximum(roi[:, :, 3], (sub_a[:, :, 0] * 255).astype(np.uint8))

    def _draw_body(self, strip, piano):
        if piano <= 0.01:
            return
        sh, sw = self.body_glow.shape
        cx, cy = 2120, int(MID + 170)
        x0, y0 = cx - sw // 2, cy - sh // 2
        x1, y1 = x0 + sw, y0 + sh
        cy0, cy1 = max(0, y0), min(LOCAL_H, y1)
        cx0, cx1 = max(0, x0), min(LOCAL_W, x1)
        if cy1 <= cy0 or cx1 <= cx0:
            return
        sub = self.body_glow[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0][:, :, None]
        add = (sub * self.accent[None, None, :] * (0.16 * piano)).astype(np.uint16)
        _clip_add(strip[cy0:cy1, cx0:cx1, :3], add)

    def _add_column(self, strip, cx, y0, y1, color, intensity):
        h = y1 - y0
        if h <= 2:
            return
        w = 34
        x0 = max(0, cx - w // 2)
        x1 = min(LOCAL_W, x0 + w)
        if x1 <= x0:
            return
        xx = np.arange(x1 - x0, dtype=np.float32) + (x0 - (cx - w // 2))
        horiz = np.clip(1.0 - np.abs(xx - w / 2) / (w / 2), 0, 1)[None, :, None] ** 1.4
        vert = np.linspace(0.35, 1.0, h, dtype=np.float32)[:, None, None]
        add = color[None, None, :] * (horiz * vert) * intensity
        _clip_add(strip[y0:y1, x0:x1, :3], np.clip(add, 0, 255).astype(np.uint16))

    def _stamp(self, strip, cx, cy, sprite, color, intensity):
        sh = sprite.shape[0]
        x0, y0 = int(round(cx - sh / 2)), int(round(cy - sh / 2))
        x1, y1 = x0 + sh, y0 + sh
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(LOCAL_W, x1), min(LOCAL_H, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sub = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0][:, :, None]
        add = (sub * color[None, None, :] * (255.0 * min(1.0, intensity))).astype(np.uint16)
        _clip_add(strip[cy0:cy1, cx0:cx1, :3], add)

    def _blend(self, frame, x, y, rgb, alpha, mul=1.0):
        sh, sw = alpha.shape
        cx0, cy0 = max(0, x), max(0, y)
        cx1, cy1 = min(W, x + sw), min(H, y + sh)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        a = (alpha[cy0 - y:cy1 - y, cx0 - x:cx1 - x] * mul)[:, :, None]
        roi = frame[cy0:cy1, cx0:cx1]
        roi[:, :, :3] = np.clip(roi[:, :, :3] * (1.0 - a)
                                + rgb[cy0 - y:cy1 - y, cx0 - x:cx1 - x] * a,
                                0, 255).astype(np.uint8)
        roi[:, :, 3] = np.maximum(roi[:, :, 3], (a[:, :, 0] * 255).astype(np.uint8))

    def _badge_sprite(self, pitch, si):
        key = (pitch, si)
        sprite = self.badge_sprites.get(key)
        if sprite is not None:
            return sprite
        note = f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"
        semis = pitch - STRINGS[si][1]
        if semis <= 0:
            detail = "open string"
        else:
            # Whole-step hand frame: each finger sits two semitones above the
            # previous one, e.g. on the D string E=1st, F/F#=2nd, G/G#=3rd.
            finger = min(4, max(1, (semis + 1) // 2))
            detail = (f"{FINGER_NAMES[finger - 1]} finger · "
                      f"{STRINGS[si][0]} string")
        f_note = load_font("sans", 14)
        f_detail = load_font("sans", 11)
        probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        nw = int(probe.textlength(note, font=f_note))
        dw = int(probe.textlength(detail, font=f_detail))
        pad_x, gap, h = 10, 8, 29
        w = pad_x * 2 + nw + gap + dw
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=8, fill=(7, 10, 17, 214),
                            outline=tuple(int(c) for c in self.accent) + (105,), width=1)
        d.text((pad_x, 6), note, font=f_note, fill=(236, 243, 255, 240))
        d.text((pad_x + nw + gap, 9), detail, font=f_detail,
               fill=(204, 214, 232, 225))
        arr = np.asarray(img, dtype=np.float32)
        sprite = (arr[:, :, :3], arr[:, :, 3] / 255.0, w, h)
        self.badge_sprites[key] = sprite
        return sprite

    def _update_badge(self, t, violin):
        if violin:
            pitch = max(violin, key=violin.get)
            if pitch != self.badge_pitch and t - self.badge_since >= self.badge_hold:
                self.badge_pitch = pitch
                self.badge_si = _string_index(pitch)
                self.badge_since = t
            target = 1.0
        else:
            target = 0.0
        self.badge_alpha += (target - self.badge_alpha) * 0.22

    def _draw_badge(self, frame):
        if self.badge_pitch is None or self.badge_alpha < 0.03:
            return
        rgb, alpha, w, h = self._badge_sprite(self.badge_pitch, self.badge_si)
        fx = self.finger_x[self.badge_si]
        if fx is None:
            fx = _stop_x(self.badge_pitch, self.badge_si)
        sx, sy = self._project(fx, STRING_Y[self.badge_si])
        x = int(round(sx + 18))
        y = int(round(sy + 16))
        # The score panel is composited after this layer, so slide the badge
        # to the other side of the finger rather than let it be cut off.
        if (x + w > PANEL_X - 6 and x < PANEL_X + PANEL_W + 6
                and y + h > PANEL_Y - 6 and y < PANEL_Y + PANEL_H + 6):
            x = int(round(sx - w - 18))
        x = max(16, min(W - w - 16, x))
        y = max(16, min(H - h - 16, y))
        self._blend(frame, x, y, rgb, alpha, min(1.0, self.badge_alpha))

    def _spawn(self, note):
        si = _string_index(note["pitch"])
        count = 2 + int(note["vel"] / 60)
        for _ in range(count):
            self.particles.append({
                "x": _stop_x(note["pitch"], si) + self.rng.uniform(-8, 8),
                "y": STRING_Y[si] - self.rng.uniform(4, 16),
                "vx": self.rng.uniform(-20, 20),
                "vy": -self.rng.uniform(45, 120),
                "life": self.rng.uniform(0.4, 0.85),
                "age": 0.0,
            })

    def _update_particles(self, strip, dt):
        alive = []
        sprite = self.spark
        sh = sprite.shape[0]
        for p in self.particles:
            p["age"] += dt
            if p["age"] >= p["life"]:
                continue
            p["x"] += p["vx"] * dt
            p["vy"] += 30 * dt
            p["y"] += p["vy"] * dt
            life = 1.0 - p["age"] / p["life"]
            x, y = int(p["x"]), int(p["y"])
            x0, y0 = x - sh // 2, y - sh // 2
            x1, y1 = x0 + sh, y0 + sh
            if x1 <= 0 or y1 <= 0 or x0 >= LOCAL_W or y0 >= LOCAL_H:
                continue
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(LOCAL_W, x1), min(LOCAL_H, y1)
            sub = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            add = (sub[:, :, None] * self.spark_color[None, None, :]
                   * (life ** 1.4) * 190).astype(np.uint16)
            _clip_add(strip[cy0:cy1, cx0:cx1, :3], add)
            alive.append(p)
        self.particles = alive


class PipeWriter:
    """Writes frames to a subprocess stdin on a background thread."""

    def __init__(self, stream, maxsize=4):
        self.stream = stream
        self.q = queue.Queue(maxsize=maxsize)
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def write(self, data):
        while not self.error:
            try:
                self.q.put(data, timeout=0.5)
                return
            except queue.Full:
                continue

    def close(self):
        while True:
            try:
                self.q.put(None, timeout=0.5)
                break
            except queue.Full:
                continue
        self.thread.join()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            if self.error is not None:
                continue
            try:
                self.stream.write(item)
            except (BrokenPipeError, ValueError, OSError) as exc:
                self.error = exc


def stream_section(section, notes, stream, seed=0):
    gen = ViolinStripGenerator(section, notes, seed)
    writer = PipeWriter(stream)
    try:
        for frame in gen.frames():
            writer.write(frame)
    finally:
        writer.close()
        try:
            stream.close()
        except Exception:
            pass
