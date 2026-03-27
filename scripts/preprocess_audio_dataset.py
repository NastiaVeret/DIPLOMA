#!/usr/bin/env python3
"""
End-to-end audio preprocessing for spectrogram classification datasets.

Pipeline per clip:
  load → resample/mono → optional denoise → optional speech-band attenuation → peak normalize
  → fixed-length 4 s segments → optional augmentation (train split only)
  → mel spectrogram PNG

Speech-band STFT masking (300–3400 Hz) is off by default: it wipes the mid-frequency mel bins and
makes spectrograms look like a dark horizontal “gap” (bad for rockets/drones and for CNN input).
Use --attenuate-voice-band only if you explicitly want crude voice suppression.

Train / val / test split is done at the *source file* level so clips from the same
recording do not appear in multiple splits.

Example:
  python scripts/preprocess_audio_dataset.py \\
    --source-root "/Users/averet/Downloads/АУДІО" \\
    --output-root "dataset/preprocessed_from_audio" \\
    --train-ratio 0.7 --val-ratio 0.15
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Iterator, List, Optional, Sequence, Tuple

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from audioread.exceptions import NoBackendError  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

try:
    import noisereduce as nr  # type: ignore

    _HAS_NOISE_REDUCE = True
except ImportError:
    _HAS_NOISE_REDUCE = False

AUDIO_EXTENSIONS = {".wav", ".ogg", ".mp3", ".flac", ".m4a", ".webm"}


@dataclass(frozen=True)
class PreprocessConfig:
    sample_rate: int = 16_000
    clip_seconds: float = 4.0
    clip_hop_seconds: Optional[float] = None
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    spectrogram_size_inches: float = 4.0
    spectrogram_dpi: int = 128
    voice_band_hz: Tuple[float, float] = (300.0, 3_400.0)
    attenuate_voice_band: bool = False
    use_noise_reduce: bool = True
    augment_train: bool = True
    aug_pitch_steps: Tuple[float, ...] = (-1.0, 1.0)
    random_seed: int = 42


def _slug_class_name(name: str) -> str:
    """Filesystem-safe folder name from dataset subdirectory."""
    s = name.strip().replace(" ", "_")
    s = re.sub(r"[^\w\-_.]", "_", s, flags=re.UNICODE)
    return s.strip("_") or "class"


def iter_audio_files(class_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(class_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            out.append(p)
    return out


def _load_with_ffmpeg(path: Path, sr: int) -> np.ndarray:
    """Decode any format ffmpeg supports → mono float32 at ``sr`` (e.g. OGG when audioread has no backend)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "Cannot decode this audio format: no ffmpeg on PATH and no other decoder. "
            "Install ffmpeg (macOS: brew install ffmpeg) and retry."
        )
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for {path}: {err or proc.returncode}")
    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return y


def load_resampled_mono(path: Path, sr: int) -> np.ndarray:
    # Prefer soundfile (WAV/FLAC; sometimes OGG if libsndfile was built with Vorbis).
    try:
        y, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        return np.asarray(y, dtype=np.float32)
    except (OSError, RuntimeError, getattr(sf, "LibsndfileError", OSError)):
        pass

    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        return np.asarray(y, dtype=np.float32)
    except NoBackendError:
        y = _load_with_ffmpeg(path, sr)
        return np.asarray(y, dtype=np.float32)


def attenuate_speech_band(y: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Attenuate STFT bins in [low_hz, high_hz] (rough voice band), keep phase."""
    if y.size == 0:
        return y
    d = librosa.stft(y)
    mag, phase = np.abs(d), np.angle(d)
    freqs = librosa.fft_frequencies(sr=sr)
    mask = (freqs < low_hz) | (freqs > high_hz)
    mag = mag * mask.astype(mag.dtype)[:, np.newaxis]
    d_out = mag * np.exp(1j * phase)
    y_out = librosa.istft(d_out, length=len(y))
    return y_out.astype(np.float32)


def denoise(y: np.ndarray, sr: int, use_nr: bool) -> np.ndarray:
    if y.size == 0:
        return y
    if use_nr and _HAS_NOISE_REDUCE:
        reduced = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.85)
        return np.asarray(reduced, dtype=np.float32)
    if use_nr and not _HAS_NOISE_REDUCE:
        print(
            "Warning: noisereduce not installed; skipping spectral noise reduction. "
            "pip install noisereduce",
            file=sys.stderr,
        )
    return y


def peak_normalize(y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    peak = float(np.max(np.abs(y))) + eps
    return (y / peak).astype(np.float32)


def clip_windows(
    y: np.ndarray, sr: int, clip_samples: int, hop_samples: int
) -> Iterator[np.ndarray]:
    n = len(y)
    if n == 0:
        return
    start = 0
    while start < n:
        chunk = y[start : start + clip_samples]
        if chunk.size < clip_samples:
            chunk = np.pad(chunk, (0, clip_samples - chunk.size))
        yield chunk.astype(np.float32)
        if start + clip_samples >= n:
            break
        start += hop_samples


def augment_variants(y: np.ndarray, sr: int, cfg: PreprocessConfig, rng: np.random.Generator) -> List[np.ndarray]:
    if not cfg.augment_train:
        return [y]
    variants: List[np.ndarray] = [y]
    for steps in cfg.aug_pitch_steps:
        variants.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=float(steps)).astype(np.float32))
    for rate in (1.05, 0.95):
        stretched = librosa.effects.time_stretch(y, rate=rate)
        variants.append(librosa.util.fix_length(stretched, size=len(y)).astype(np.float32))
    # tiny gaussian noise on one copy
    noise = rng.normal(0.0, 0.002, size=y.shape).astype(np.float32)
    variants.append(np.clip(y + noise, -1.0, 1.0))
    return variants


def process_waveform(y: np.ndarray, sr: int, cfg: PreprocessConfig) -> np.ndarray:
    y = denoise(y, sr, cfg.use_noise_reduce)
    if cfg.attenuate_voice_band:
        y = attenuate_speech_band(y, sr, cfg.voice_band_hz[0], cfg.voice_band_hz[1])
    y = peak_normalize(y)
    return y


def save_mel_png(y_clip: np.ndarray, sr: int, out_path: Path, cfg: PreprocessConfig) -> None:
    mel = librosa.feature.melspectrogram(
        y=y_clip,
        sr=sr,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    fig = plt.figure(figsize=(cfg.spectrogram_size_inches, cfg.spectrogram_size_inches))
    ax = fig.add_subplot(111)
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=cfg.hop_length,
        x_axis=None,
        y_axis=None,
        ax=ax,
        cmap="magma",
    )
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=cfg.spectrogram_dpi)
    plt.close(fig)


def discover_classes(source_root: Path) -> List[Path]:
    subs = [p for p in sorted(source_root.iterdir()) if p.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No class subfolders under {source_root}")
    return subs


def collect_file_labels(class_dirs: Sequence[Path]) -> Tuple[List[Path], List[str]]:
    paths: List[Path] = []
    labels: List[str] = []
    for d in class_dirs:
        label = _slug_class_name(d.name)
        for f in iter_audio_files(d):
            paths.append(f)
            labels.append(label)
    if not paths:
        raise FileNotFoundError("No audio files found in class folders.")
    return paths, labels


def _stratify_ok(label_list: Sequence[str]) -> bool:
    if len(set(label_list)) <= 1:
        return False
    return min(Counter(label_list).values()) >= 2


def split_paths(
    paths: List[Path],
    labels: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Need 0 < train_ratio < 1, val_ratio >= 0, train_ratio + val_ratio < 1")

    test_ratio = 1.0 - train_ratio - val_ratio
    idx = list(range(len(paths)))
    strat = labels if _stratify_ok(labels) else None
    train_idx, temp_idx = train_test_split(
        idx, test_size=(1.0 - train_ratio), random_state=seed, stratify=strat
    )
    rel_val = val_ratio / (val_ratio + test_ratio)
    temp_labels = [labels[i] for i in temp_idx]
    strat_temp = temp_labels if _stratify_ok(temp_labels) else None
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1.0 - rel_val), random_state=seed + 1, stratify=strat_temp
    )
    return train_idx, val_idx, test_idx


def spectrogram_stem(split: str, class_label: str, source_stem: str, tag: str) -> str:
    safe = f"{split}_{class_label}_{source_stem}_{tag}".replace(os.sep, "_")
    return safe


def run_pipeline(args: argparse.Namespace) -> None:
    cfg = PreprocessConfig(
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        clip_hop_seconds=args.clip_hop_seconds,
        n_mels=args.n_mels,
        attenuate_voice_band=args.attenuate_voice_band,
        use_noise_reduce=not args.no_noise_reduce,
        augment_train=not args.no_augment,
        random_seed=args.seed,
    )
    rng = np.random.default_rng(cfg.random_seed)

    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    work_root = output_root / "work"
    image_root = output_root / "images"

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    class_dirs = discover_classes(source_root)
    paths, labels = collect_file_labels(class_dirs)

    train_idx, val_idx, test_idx = split_paths(
        paths, labels, args.train_ratio, args.val_ratio, args.seed
    )

    split_map = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    clip_samples = int(cfg.clip_seconds * cfg.sample_rate)
    hop_samples = (
        int(cfg.clip_hop_seconds * cfg.sample_rate)
        if cfg.clip_hop_seconds is not None
        else clip_samples
    )

    counters: dict[str, int] = {}

    def handle_file(file_idx: int, split_name: str) -> None:
        path = paths[file_idx]
        class_label = labels[file_idx]
        y = load_resampled_mono(path, cfg.sample_rate)
        y = process_waveform(y, cfg.sample_rate, cfg)

        base_stem = path.stem
        augment = cfg.augment_train and split_name == "train"
        wave_variants = augment_variants(y, cfg.sample_rate, cfg, rng) if augment else [y]

        for v_i, y_v in enumerate(wave_variants):
            v_tag = f"v{v_i}" if augment else "base"
            for w_i, window in enumerate(clip_windows(y_v, cfg.sample_rate, clip_samples, hop_samples)):
                tag = f"{v_tag}_w{w_i + 1}"
                stem = spectrogram_stem(split_name, class_label, base_stem, tag)
                png_name = f"{stem}.png"
                out_dir = image_root / split_name / class_label
                png_path = out_dir / png_name

                clip_wav = work_root / split_name / class_label / f"{stem}.wav"
                clip_wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(clip_wav), window, cfg.sample_rate)

                save_mel_png(window, cfg.sample_rate, png_path, cfg)
                counters[split_name] = counters.get(split_name, 0) + 1

    for split_name, indices in split_map.items():
        for i in indices:
            handle_file(i, split_name)

    print("Done.")
    print(f"Source: {source_root}")
    print(f"Output images (ImageFolder): {image_root}")
    print(f"Intermediate WAV clips: {work_root}")
    for s in ("train", "val", "test"):
        print(f"  {s}: {counters.get(s, 0)} spectrograms")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--source-root",
        type=str,
        required=True,
        help="Folder with one subdirectory per class (e.g. АУДІО/Ракети, АУДІО/Шахеди).",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="dataset/preprocessed_from_audio",
        help="Where to write work/ and images/ trees.",
    )
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-rate", type=int, default=16_000)
    p.add_argument("--clip-seconds", type=float, default=4.0)
    p.add_argument(
        "--clip-hop-seconds",
        type=float,
        default=None,
        help="If set (e.g. 2.0), use overlapping 4 s windows with this hop.",
    )
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument(
        "--attenuate-voice-band",
        action="store_true",
        help="Crude STFT notch ~300–3400 Hz (often ruins mel spectrograms for non-speech targets).",
    )
    p.add_argument(
        "--no-noise-reduce",
        action="store_true",
        help="Disable noisereduce spectral denoising.",
    )
    p.add_argument("--no-augment", action="store_true", help="Disable train-only augmentation.")
    p.add_argument(
        "--clean",
        action="store_true",
        help="Delete output-root before running.",
    )
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
