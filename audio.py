"""Two-stem FluidSynth rendering (violin + piano) plus mastering.

The violin stem uses a General MIDI soundfont; the piano stem prefers the same
Salamander grand used by the piano channel, so the accompaniment keeps that
close-miked, natural character. Stems are mixed, then run through one shared
hall reverb and loudness normaliser so both instruments sit in the same room.
"""
import glob
import json
import os
import re
import struct
import subprocess

import numpy as np

from config import (FFMPEG, PROJECT_DIR, SAMPLE_RATE, SOUNDFONT_DIR, WORK_DIR,
                    find_fluidsynth)


def resolve_violin_soundfont():
    preferred = os.getenv("VIOLIN_SF2", "").strip()
    if preferred and os.path.exists(preferred):
        return preferred
    for pattern in ("MuseScore_General*.sf3", "MuseScore_General*.sf2",
                    "FluidR3*.sf2", "GeneralUser*.sf2"):
        for hit in sorted(glob.glob(os.path.join(SOUNDFONT_DIR, pattern))):
            if os.path.getsize(hit) > 10 * 1024 * 1024:
                return hit
    fallback = os.path.join(SOUNDFONT_DIR, "TimGM6mb.sf2")
    if os.path.exists(fallback):
        print("[audio] WARNING: using the 6 MB TimGM6mb fallback for strings.")
        return fallback
    raise RuntimeError("No violin soundfont found in soundfonts/")


def resolve_piano_soundfont():
    preferred = os.getenv("PIANO_SF2", "").strip()
    if preferred and os.path.exists(preferred):
        return preferred
    sibling = os.path.join(os.path.dirname(PROJECT_DIR), "piano_longform_bot",
                           "soundfonts")
    candidates = []
    for folder in (SOUNDFONT_DIR, sibling):
        candidates += sorted(glob.glob(os.path.join(folder, "Salamander*.sf2")))
        candidates.append(os.path.join(folder, "Yamaha_Grand.sf2"))
    for hit in candidates:
        if os.path.exists(hit) and os.path.getsize(hit) > 10 * 1024 * 1024:
            return hit
    return resolve_violin_soundfont()


def _split_stems(midi_path, work_dir):
    """Seperate the violin and piano tracks into their own MIDI files."""
    from mido import MidiFile

    mid = MidiFile(midi_path)
    tempo = mid.tracks[0]
    picked = {}
    for track in mid.tracks[1:]:
        name = ""
        for msg in track:
            if msg.type == "track_name":
                name = msg.name.lower()
                break
        key = "violin" if "violin" in name else ("piano" if "piano" in name else None)
        if key and key not in picked:
            picked[key] = track
    stems = {}
    for key, track in picked.items():
        sub = MidiFile(type=1, ticks_per_beat=mid.ticks_per_beat)
        sub.tracks.append(tempo)
        sub.tracks.append(track)
        path = os.path.join(work_dir,
                            os.path.basename(midi_path).replace(".mid", f"_{key}.mid"))
        sub.save(path)
        stems[key] = path
    if "violin" not in stems or "piano" not in stems:
        raise RuntimeError("MIDI is missing the Violin or Piano track")
    return stems


def _write_wav_f32(path, mono, sample_rate):
    data = np.asarray(mono, dtype="<f4").tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, 1, sample_rate,
                            sample_rate * 4, 4, 32))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def ensure_hall_ir(path, seconds=2.0):
    """Build a diffuse stereo-stage hall IR.

    No discrete early reflections: a handful of spikes at fixed delays is
    what makes a synthetic reverb sound like a small tiled room. Energy is
    normalised to 1 so afir's wet gain maps directly to reverb level.
    """
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(11)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE

    noise = rng.standard_normal(n)
    # Split into bands whose tails decay at different rates, so the high
    # end dies away first the way air absorption works in a real hall.
    k = max(8, int(0.018 * SAMPLE_RATE))
    low = np.convolve(noise, np.ones(k) / k, mode="same")
    high = noise - low
    ir = (low * np.exp(-t * 2.6) * 0.95
          + high * np.exp(-t * 7.0) * 0.28)

    # Soft onset: no dry-sounding slap, just a gently swelling diffuse field.
    attack = int(0.035 * SAMPLE_RATE)
    ir[:attack] *= np.linspace(0.0, 1.0, attack) ** 1.5

    fade_len = int(0.4 * SAMPLE_RATE)
    ir[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
    _write_wav_f32(path, ir, SAMPLE_RATE)
    return path


def _render_stem(fluid, sf2, midi_path, wav_path):
    cmd = [
        fluid, "-ni", "-q",
        "-F", wav_path, "-T", "wav", "-O", "float",
        "-r", str(SAMPLE_RATE), "-g", "0.8",
        "-o", "synth.polyphony=256",
        "-o", "synth.cpu-cores=2",
        "-o", "synth.chorus.active=0",
        # FluidSynth's built-in room is a cheap metallic Schroeder reverb;
        # the convolution hall below replaces it entirely.
        "-o", "synth.reverb.active=0",
        sf2, midi_path,
    ]
    print(f"[audio] FluidSynth: {os.path.basename(sf2)} -> {os.path.basename(wav_path)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1024:
        raise RuntimeError(f"FluidSynth failed:\n{res.stderr[-2000:]}")


def render_audio(midi_path, output_wav, work_dir=None):
    fluid = find_fluidsynth()
    if not fluid:
        raise RuntimeError("fluidsynth not found (install it or keep bin/fluidsynth)")
    work_dir = work_dir or os.path.dirname(output_wav)
    os.makedirs(work_dir, exist_ok=True)
    stems = _split_stems(midi_path, work_dir)
    raw_wav = os.path.join(work_dir, os.path.basename(output_wav).replace(".wav", "_raw.wav"))
    vln_wav = os.path.join(work_dir, os.path.basename(output_wav).replace(".wav", "_violin_raw.wav"))
    pno_wav = os.path.join(work_dir, os.path.basename(output_wav).replace(".wav", "_piano_raw.wav"))

    _render_stem(fluid, resolve_violin_soundfont(), stems["violin"], vln_wav)
    _render_stem(fluid, resolve_piano_soundfont(), stems["piano"], pno_wav)

    mix_chain = (
        "[0:a]aformat=sample_fmts=fltp:channel_layouts=stereo,volume=1.0[v];"
        "[1:a]aformat=sample_fmts=fltp:channel_layouts=stereo,volume=2.2[p];"
        "[v][p]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[pre]"
    )
    cmd = [FFMPEG, "-y", "-v", "error", "-i", vln_wav, "-i", pno_wav,
           "-filter_complex", mix_chain, "-map", "[pre]", "-c:a", "pcm_f32le", raw_wav]
    _run(cmd, "violin/piano stem mix")

    master_audio(raw_wav, output_wav, work_dir)
    return output_wav


def master_audio(raw_wav, output_wav, work_dir):
    ir = ensure_hall_ir(os.path.join(WORK_DIR, "hall_strings_2.6s.wav"), 2.6)
    processed = os.path.join(work_dir, os.path.basename(output_wav).replace(".wav", "_proc.wav"))

    chain = (
        # Strings carry most of their energy in the 150-800 Hz body, so the
        # low shelf stays gentle and presence sits around 3 kHz.
        "[0:a]highpass=f=30,lowpass=f=18000,"
        "equalizer=f=240:t=q:w=1.0:g=-1.0,"
        "equalizer=f=800:t=q:w=1.2:g=-0.6,"
        "equalizer=f=3200:t=q:w=1.2:g=0.8,"
        "equalizer=f=9000:t=q:w=1.0:g=0.5,"
        "acompressor=threshold=0.25:ratio=1.4:attack=40:release=500:knee=8:makeup=1,"
        "aformat=sample_fmts=fltp:channel_layouts=stereo,asplit[dry][conv];"
        "[1:a]aformat=sample_fmts=fltp:channel_layouts=mono[ir];"
        # Some ffmpeg builds drop afir's dry path entirely (dry behaves like
        # a multiplier), so the dry/wet mix is done explicitly with amix.
        # The IR is energy-normalised: wet=0.24 is simply "24% of the direct
        # sound", not a room-sized wash.
        "[conv][ir]afir=dry=1:wet=1:gtype=none:irnorm=-1:irfmt=mono:irlink=1:maxir=3,"
        "volume=0.24[wv];"
        "[dry][wv]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
        "stereowiden=delay=11:feedback=0.12:crossfeed=0.18:drymix=0.85,"
        "alimiter=limit=0.95:attack=5:release=80:level=0,"
        "aresample=48000:resampler=soxr:precision=28"
    )
    cmd = [FFMPEG, "-y", "-v", "error", "-i", raw_wav, "-i", ir,
           "-filter_complex", chain, "-c:a", "pcm_f32le", processed]
    _run(cmd, "tone/reverb mastering")

    measured = _measure_loudness(processed)
    apply_chain = (
        "loudnorm=I=-16:TP=-1.5:LRA=11"
        f":measured_I={measured['input_i']}:measured_LRA={measured['input_lra']}"
        f":measured_TP={measured['input_tp']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true,"
        "aresample=48000:resampler=soxr:precision=28"
    )
    cmd = [FFMPEG, "-y", "-v", "error", "-i", processed,
           "-af", apply_chain, "-c:a", "pcm_s24le", output_wav]
    _run(cmd, "loudness normalisation")
    print(f"[audio] Mastered {os.path.basename(output_wav)} "
          f"({measured['input_i']} LUFS -> -16 LUFS)")
    return output_wav


def _measure_loudness(path):
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-i", path,
           "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", res.stderr, re.S)
    if not match:
        raise RuntimeError("Could not measure loudness:\n" + res.stderr[-1500:])
    return json.loads(match.group(0))


def _run(cmd, label):
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed during {label}:\n{res.stderr[-2000:]}")


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "output/preview_42.mid"
    render_audio(src, src.replace(".mid", "_audio.wav"))
