"""Renders one section: background + violin strip + title + score panel."""
import collections
import os
import subprocess
import threading

from config import FFPROBE, FFMPEG, FPS, H, W, WORK_DIR
from render.violin_strip import stream_section
from render.scene import make_background, make_title_card
from render.score_cursor import build_cursor_video
from render.score_layer import (CROP_EXPR, FRAME_PAD, PANEL_X, PANEL_Y, WIN_H,
                                WIN_W, WIN_X, WIN_Y, build_score_assets)

ENC_ARGS = [
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
    "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
    "-x264-params", "aq-mode=3:keyint=60:min-keyint=60:scenecut=0",
    "-threads", "2", "-filter_complex_threads", "2",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    "-color_range", "tv", "-an",
]


def section_paths(index, work_dir):
    stem = os.path.join(work_dir, f"s{index:02d}")
    return {
        "stem": stem,
        "bg": stem + "_bg.png",
        "title": stem + "_title.png",
        "out": stem + ".mp4",
    }


def render_section(task):
    section, notes, work_dir, seed, resume = task
    paths = section_paths(section["index"], work_dir)
    if resume and _valid_section(paths["out"], section["frames"]):
        return paths["out"], True

    bg = make_background(section["palette"], seed + section["index"])
    bg.save(paths["bg"])
    title = make_title_card(section["name"], section["palette"], center_x=460)
    title.save(paths["title"])
    score = build_score_assets(section, notes, work_dir)
    cursor = build_cursor_video(section, score[2], work_dir) if score else None
    _encode(section, notes, paths, seed, score, cursor)
    if not _valid_section(paths["out"], section["frames"]):
        raise RuntimeError(f"section {section['index']} produced an invalid file")
    return paths["out"], False


def _encode(section, notes, paths, seed, score=None, cursor=None):
    frames = section["frames"]
    dur = frames / FPS
    out_st = max(3.0, dur - 1.3)
    score_chain = ""
    final_label = "v2"
    if score:
        parts = [
            "[3:v]format=rgba,loop=loop=-1:size=1:start=0,"
            f"crop={WIN_W}:{WIN_H}:0:{CROP_EXPR},"
            f"fade=t=out:st={out_st:.2f}:d=1.3:alpha=1[score]; ",
            "[4:v]format=rgba,loop=loop=-1:size=1:start=0,"
            f"fade=t=out:st={out_st:.2f}:d=1.3:alpha=1[scoref]; ",
            f"[v2][score]overlay={WIN_X}:{WIN_Y}:eof_action=repeat[v3]; ",
        ]
        under = "v3"
        if cursor:
            parts.append(
                "[5:v]format=rgba,"
                f"fade=t=out:st={out_st:.2f}:d=1.3:alpha=1[cursor]; "
                f"[v3][cursor]overlay={WIN_X}:{WIN_Y}:eof_action=repeat[v3c]; ")
            under = "v3c"
        parts.append(
            f"[{under}][scoref]overlay={PANEL_X - FRAME_PAD}:"
            f"{PANEL_Y - FRAME_PAD}:eof_action=repeat[v4]; ")
        score_chain = "".join(parts)
        final_label = "v4"
    filter_graph = (
        "[0:v]format=gbrp,loop=loop=-1:size=1:start=0[bg]; "
        "[bg][2:v]overlay=0:0:eof_action=repeat[v1]; "
        "[1:v]format=rgba,loop=loop=-1:size=1:start=0,"
        "fade=t=in:st=0:d=0.45:alpha=1,fade=t=out:st=1.55:d=0.55:alpha=1[t]; "
        "[v1][t]overlay=0:0:enable='lte(t,2.2)'[v2]; "
        f"{score_chain}"
        f"[{final_label}]eq=contrast=1.04:saturation=1.12:gamma=1.05,format=yuv420p[v]"
    )
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "warning", "-nostats",
        "-framerate", str(FPS), "-i", paths["bg"],
        "-framerate", str(FPS), "-i", paths["title"],
        "-f", "rawvideo", "-pixel_format", "rgba",
        "-video_size", f"{W}x{H}", "-framerate", str(FPS), "-i", "pipe:0",
    ]
    if score:
        cmd += ["-framerate", str(FPS), "-i", score[0],
                "-framerate", str(FPS), "-i", score[1]]
    if cursor:
        cmd += ["-i", cursor]
    cmd += [
        "-filter_complex", filter_graph,
        "-map", "[v]", "-frames:v", str(frames), "-r", str(FPS),
        *ENC_ARGS, paths["out"],
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    tail = collections.deque(maxlen=60)

    def drain():
        for line in proc.stderr:
            tail.append(line.decode("utf-8", "replace"))

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        stream_section(section, notes, proc.stdin, seed)
    except BrokenPipeError:
        pass
    proc.wait()
    reader.join(timeout=5)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg section {section['index']} failed:\n" + "".join(tail)[-3000:])


def _valid_section(path, frames):
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return False
    try:
        res = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=nb_frames", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=30)
        return int(res.stdout.strip() or 0) == frames
    except Exception:
        return False
