"""High-CTR 1280x720 thumbnail built from a real rendered frame.

The frame already carries the photoreal violin, the animated strings/bow and
the sheet-music panel, so this module only layers branding, kicker, dominant
title and a gold frame on top. Everything uses bundled OFL fonts, which makes
it deterministic on Linux CI as well as Windows.
"""
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from config import FFMPEG, FINAL_EXPORT_DIR, OUTPUT_DIR
from render.scene import load_font, palette

TW, TH = 1280, 720
TEXT_X = 792
TEXT_W = TW - TEXT_X - 54


def _load_meta(video_path):
    stem = os.path.splitext(video_path)[0]
    candidates = [
        stem + "_meta.json",
        os.path.join(FINAL_EXPORT_DIR, os.path.basename(stem) + "_meta.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f), p
            except Exception:
                pass
    return {}, None


def _hero_section(meta):
    sections = meta.get("sections") or []
    for s in sections:
        if s.get("role") == "climax":
            return s
    return sections[min(2, len(sections) - 1)] if sections else None


def _extract_hero(video_path, meta):
    stem = os.path.splitext(video_path)[0]
    for cand in (stem + "_hero.jpg",
                 os.path.join(FINAL_EXPORT_DIR, os.path.basename(stem) + "_hero.jpg")):
        if os.path.exists(cand):
            return cand, False

    section = _hero_section(meta)
    duration = float(meta.get("duration", 0) or 0)
    if not duration:
        duration = _probe_duration(video_path)
    if section:
        t = section["start"] + 25
    else:
        t = duration * 0.5 if duration < 30 else duration * 0.38
    t = max(0.5, min(t, max(0.5, duration - 0.5)))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "_thumb_raw.jpg")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", video_path,
                    "-frames:v", "1", "-q:v", "2", out], check=False)
    if not os.path.exists(out) or os.path.getsize(out) < 1024:
        subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", "0.5", "-i", video_path,
                        "-frames:v", "1", "-q:v", "2", out], check=False)
    return out, True


def _probe_duration(video_path):
    try:
        from config import FFPROBE
        res = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", video_path],
                             capture_output=True, text=True, timeout=30)
        return float(res.stdout.strip() or 0)
    except Exception:
        return 0.0


def _split_title(title):
    words = title.split()
    if len(words) < 3:
        return title.upper(), ""
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        score = abs(len(a) - len(b))
        if best is None or score < best[0]:
            best = (score, a, b)
    return best[1].upper(), best[2].upper()


def _build_hook(meta, minutes):
    title = meta.get("title") or "Relaxing Violin & Piano Suite"
    t1, t2 = _split_title(title)
    kicker = f"{minutes} MIN VIOLIN & PIANO" if minutes else "VIOLIN & PIANO SUITE"
    return {
        "kicker": kicker,
        "title1": t1,
        "title2": t2,
        "sub": "ORIGINAL COMPOSITION · DEEP FOCUS · SLEEP",
        "badge": "CINEMATIC VIOLIN",
    }


def _fit_text(draw, text, max_w, start, minimum, kind="sans"):
    size = start
    while size > minimum:
        font = load_font(kind, size)
        if draw.textlength(text, font=font) <= max_w:
            return font, size
        size -= 1
    return load_font(kind, minimum), minimum


def _fit_title(draw, lines, max_w, start=64, minimum=36):
    size = start
    while size > minimum:
        font = load_font("sans", size)
        if all(draw.textlength(line, font=font) <= max_w for line in lines if line):
            return font, size
        size -= 2
    return load_font("sans", minimum), minimum


def create_violinspheric_thumbnail(video_path, output_thumb_path, hook_dict=None,
                                   minutes=None, episode=None):
    if not os.path.isabs(output_thumb_path):
        output_thumb_path = os.path.abspath(output_thumb_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_thumb_path)), exist_ok=True)

    meta, _ = _load_meta(video_path)
    hero_path, temp_hero = _extract_hero(video_path, meta)
    if not os.path.exists(hero_path):
        raise FileNotFoundError(f"Could not extract a hero frame from {video_path}")
    print(f"[thumbnail] Hero frame: {os.path.basename(hero_path)}")

    if minutes is None:
        minutes = round(meta.get("duration", 0) / 60) or None

    pal_name = "ember"
    section = _hero_section(meta)
    if section and section.get("palette"):
        pal_name = section["palette"]
    pal = palette(pal_name)

    full = Image.open(hero_path).convert("RGB")
    scale = max(TW / full.width, TH / full.height)
    img = full.resize((int(full.width * scale + 0.5), int(full.height * scale + 0.5)),
                      Image.Resampling.LANCZOS)
    left = (img.width - TW) // 2
    top = (img.height - TH) // 2
    img = img.crop((left, top, left + TW, top + TH))
    img = ImageEnhance.Color(img).enhance(1.26)
    img = ImageEnhance.Contrast(img).enhance(1.12)
    img = ImageEnhance.Brightness(img).enhance(1.04)

    overlay = Image.new("RGBA", (TW, TH), (0, 0, 0, 0))

    # Right-hand scrim: the violin stays fully visible on the left while the
    # title column gets a clean, dark stage. The top boost tames the sheet
    # panel that the render bakes into the upper right.
    yy, xx = np.mgrid[0:TH, 0:TW].astype(np.float32)
    right = np.clip((xx - 500.0) / 780.0, 0, 1) ** 1.2
    top_boost = (np.clip(1.0 - yy / 370.0, 0, 1) ** 1.3
                 * np.clip((xx - 620.0) / 660.0, 0, 1) ** 1.1)
    bottom = np.clip((yy - 560.0) / 160.0, 0, 1) ** 1.5
    alpha = np.clip(242 * right + 112 * top_boost + 46 * bottom, 0, 250)
    scrim = Image.new("RGBA", (TW, TH), (5, 7, 12, 255))
    scrim.putalpha(Image.fromarray(alpha.astype(np.uint8)))
    overlay.alpha_composite(scrim)

    # The render bakes a sheet-music panel into the upper right. Fade most of
    # it out so the corner mark never competes with the branding, then bring
    # the score itself back as a crisp card (visible sheet music sells the
    # "this is real music" feeling).
    sheet_mask = Image.new("L", (TW, TH), 0)
    ImageDraw.Draw(sheet_mask).rounded_rectangle((620, -110, 1360, 345), radius=90, fill=252)
    sheet_mask = sheet_mask.filter(ImageFilter.GaussianBlur(62))
    sheet = Image.new("RGBA", (TW, TH), (4, 6, 11, 255))
    sheet.putalpha(sheet_mask)
    overlay.alpha_composite(sheet)

    # The panel's header sits in empty stage above the violin; mask that top
    # strip fully so no baked title text survives the scrim.
    strip = np.zeros((TH, TW, 4), dtype=np.uint8)
    sx = np.clip((xx - 520.0) / 80.0, 0, 1)
    sy = np.clip((135.0 - yy) / 60.0, 0, 1)
    strip[..., 0:3] = (5, 7, 12)
    strip[..., 3] = (250 * sx * sy).astype(np.uint8)
    overlay.alpha_composite(Image.fromarray(strip, "RGBA"))
    draw = ImageDraw.Draw(overlay)

    # Hide the channel mark baked into the hero frame behind our brand row.
    patch = np.zeros((124, 600, 4), dtype=np.uint8)
    ax = np.clip((600.0 - np.arange(600, dtype=np.float32)) / 80.0, 0, 1)
    ay = np.clip((124.0 - np.arange(124, dtype=np.float32)) / 80.0, 0, 1)
    patch[..., 0:3] = (5, 7, 12)
    patch[..., 3] = (250 * ay[:, None] * ax[None, :]).astype(np.uint8)
    overlay.alpha_composite(Image.fromarray(patch, "RGBA"), (0, 0))

    # Original score card: crop the engraved staves straight from the
    # full-resolution frame, lift the contrast, and present them as a framed
    # card above the title column.
    CARD_X, CARD_Y, CARD_W = 358, 92, 810
    src = full.crop((int(full.width * 0.515), int(full.height * 0.088),
                     int(full.width * 0.96), int(full.height * 0.28)))
    card_h = int(round(src.height * CARD_W / src.width))
    src = src.resize((CARD_W, card_h), Image.Resampling.LANCZOS)
    src = ImageEnhance.Brightness(src).enhance(1.38)
    src = ImageEnhance.Contrast(src).enhance(1.3)

    card_mask = Image.new("L", (CARD_W, card_h), 0)
    ImageDraw.Draw(card_mask).rounded_rectangle((0, 0, CARD_W - 1, card_h - 1),
                                                radius=20, fill=255)
    shadow = Image.new("RGBA", (CARD_W + 60, card_h + 60), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((30, 36, CARD_W + 30, card_h + 36),
                                             radius=22, fill=(0, 0, 0, 165))
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    overlay.alpha_composite(shadow, (CARD_X - 30, CARD_Y - 30))

    card = Image.new("RGBA", (CARD_W, card_h), (0, 0, 0, 0))
    card.paste(src, (0, 0), mask=card_mask)
    ImageDraw.Draw(card).rounded_rectangle((0, 0, CARD_W - 1, card_h - 1), radius=20,
                                           outline=tuple(pal["accent"]) + (150,), width=2)
    overlay.alpha_composite(card, (CARD_X, CARD_Y))

    score_chip = "ORIGINAL SCORE"
    f_chip = load_font("sans", 14)
    chip_w = int(draw.textlength(score_chip, font=f_chip)) + 22
    cx0 = CARD_X + CARD_W - chip_w - 14
    cy0 = CARD_Y + card_h - 38
    draw.rounded_rectangle((cx0, cy0, cx0 + chip_w, cy0 + 26), radius=7,
                           fill=(8, 10, 16, 225),
                           outline=tuple(pal["accent"]) + (150,), width=1)
    draw.text((cx0 + 11, cy0 + 6), score_chip, font=f_chip,
              fill=tuple(pal["accent"]) + (255,))

    hook = dict(_build_hook(meta, minutes))
    if hook_dict:
        hook.update(hook_dict)

    # Brand row --------------------------------------------------------------
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
    logo_size = 42
    lx, ly = 46, 34
    if os.path.exists(logo_path):
        logo = Image.open(logo_path).convert("RGBA").resize(
            (logo_size, logo_size), Image.Resampling.LANCZOS)
        mask = Image.new("L", (logo_size, logo_size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, logo_size, logo_size), fill=255)
        circ = Image.new("RGBA", (logo_size, logo_size), (0, 0, 0, 0))
        circ.paste(logo, (0, 0), mask=mask)
        overlay.alpha_composite(circ, (lx, ly))
        draw.ellipse((lx - 2, ly - 2, lx + logo_size + 2, ly + logo_size + 2),
                     outline=tuple(pal["accent"]) + (90,), width=1)

    f_brand = load_font("sans", 22)
    f_badge = load_font("sans", 13)
    brand_text = "ViolinSpheric"
    tx = lx + logo_size + 14
    b_bbox = draw.textbbox((0, 0), brand_text, font=f_brand)
    b_th = b_bbox[3] - b_bbox[1]
    ty = ly + (logo_size - b_th) // 2 - b_bbox[1]
    draw.text((tx + 2, ty + 2), brand_text, font=f_brand, fill=(0, 0, 0, 200))
    draw.text((tx, ty), brand_text, font=f_brand, fill=(255, 255, 255, 255))

    vx = tx + int(draw.textlength(brand_text, font=f_brand)) + 10
    vy = ly + (logo_size - 16) // 2
    draw.ellipse((vx, vy, vx + 16, vy + 16), fill=(29, 155, 240, 255))
    draw.line((vx + 4, vy + 8, vx + 7, vy + 11), fill=(255, 255, 255, 255), width=2)
    draw.line((vx + 7, vy + 11, vx + 11, vy + 5), fill=(255, 255, 255, 255), width=2)

    badge_text = hook.get("badge", "CINEMATIC VIOLIN")
    bw = int(draw.textlength(badge_text, font=f_badge)) + 24
    bx = vx + 28
    draw.rounded_rectangle((bx, ly + 7, bx + bw, ly + 33), radius=13,
                           fill=(8, 10, 16, 210), outline=tuple(pal["accent"]) + (170,), width=1)
    draw.text((bx + 12, ly + 13), badge_text, font=f_badge,
              fill=tuple(pal["accent"]) + (255,))

    # Kicker -----------------------------------------------------------------
    f_kicker = load_font("sans", 17)
    kicker = hook.get("kicker", "VIOLIN & PIANO SUITE")
    ky = CARD_Y + card_h + 16
    kw = int(draw.textlength(kicker, font=f_kicker)) + 30
    draw.rounded_rectangle((TEXT_X, ky, TEXT_X + kw, ky + 34), radius=8,
                           fill=tuple(pal["accent"]) + (238,))
    draw.text((TEXT_X + 15, ky + 8), kicker, font=f_kicker, fill=(8, 10, 16, 255))

    # Title ------------------------------------------------------------------
    lines = [hook.get("title1", ""), hook.get("title2", "")]
    f_main, size = _fit_title(draw, lines, max_w=TEXT_W)
    t_y = ky + 54
    for i, line in enumerate(lines):
        if not line:
            continue
        y = t_y + i * int(size * 1.02)
        for ox, oy in ((-3, -3), (3, -3), (-3, 3), (3, 3), (0, 6), (0, 9)):
            draw.text((TEXT_X + ox, y + oy), line, font=f_main, fill=(0, 0, 0, 235))
        fill = (255, 255, 255, 255) if i == 0 else tuple(pal["rh"]) + (255,)
        draw.text((TEXT_X, y), line, font=f_main, fill=fill)

    n_lines = sum(1 for line in lines if line) or 1
    rule_y = t_y + n_lines * int(size * 1.02) + 16
    draw.line((TEXT_X, rule_y, TEXT_X + 92, rule_y),
              fill=tuple(pal["accent"]) + (130,), width=2)

    # Subtitle ---------------------------------------------------------------
    sub = hook.get("sub", "")
    sub_y = rule_y + 22
    if sub:
        f_sub, _ = _fit_text(draw, sub, max_w=TEXT_W, start=25, minimum=15, kind="serif")
        draw.text((TEXT_X + 1, sub_y + 2), sub, font=f_sub, fill=(0, 0, 0, 220))
        draw.text((TEXT_X, sub_y), sub, font=f_sub, fill=(226, 232, 242, 255))

    # Episode chip -----------------------------------------------------------
    if episode:
        f_ep = load_font("sans", 15)
        ep_text = f"EP. {int(episode):02d}"
        ep_w = int(draw.textlength(ep_text, font=f_ep)) + 26
        ex = TW - 54 - ep_w
        ey = TH - 64
        draw.rounded_rectangle((ex, ey, ex + ep_w, ey + 32), radius=8,
                               fill=(8, 10, 16, 205), outline=tuple(pal["accent"]) + (120,), width=1)
        draw.text((ex + 13, ey + 8), ep_text, font=f_ep,
                  fill=tuple(pal["accent"]) + (225,))

    # Frame ------------------------------------------------------------------
    draw.rectangle([(0, 0), (TW - 1, TH - 1)],
                   outline=tuple(pal["accent"]) + (120,), width=3)

    final = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    final.save(output_thumb_path, "JPEG", quality=92)

    if temp_hero and os.path.exists(hero_path):
        os.remove(hero_path)

    print(f"[thumbnail] Saved: {output_thumb_path}")
    return output_thumb_path


if __name__ == "__main__":
    import sys
    create_violinspheric_thumbnail(sys.argv[1], sys.argv[2])
