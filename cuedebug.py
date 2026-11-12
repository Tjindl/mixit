"""--cue-debug: see each song's sections and cue points, to check them by eye and ear.

For every song in the set, this writes two files to a cue_debug folder next to the
mix:

  NN title.txt  an Audacity label file. In Audacity: open the song's original file,
                then File > Import > Labels... and pick this file. You get a label
                track with the sections (intro, build, drop 1...), every phrase
                boundary, the cue points automix considered ("exit?", "entry?"),
                the ones it chose ("OUT ->", "IN <-") and the part that plays.
  NN title.png  the same as a picture: the song's loudness over time, its sections
                in colour, phrase lines, and the chosen cues.

Times are in seconds from the start of the original file, so they line up exactly
with the file in Audacity.
"""

import re
import sys
from pathlib import Path

LABEL_COLOURS = {"intro": "#9ecae1", "build": "#fdae6b", "drop": "#e6550d", "breakdown": "#a1d99b",
                 "groove": "#bcbddc", "outro": "#9ecae1"}


def write(folder: Path, songs, cues, chosen, plan, ranges, file_time, sr: int) -> None:
    """Write the label file and the PNG for every song (see the module docstring)."""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.iterdir():  # last run's files (the order, and so the numbering, may differ)
        if re.fullmatch(r"\d\d .*\.(txt|png)", old.name):
            old.unlink()
    try:
        import matplotlib

        matplotlib.use("Agg")  # draw to files, no window
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("  note: matplotlib isn't installed, so --cue-debug writes labels but no PNGs "
              "(pip install matplotlib)", file=sys.stderr)
    for i, (song, cue) in enumerate(zip(songs, cues)):
        marks = []  # (time, text, colour) for the chosen cue points
        if i > 0:
            option = chosen[i - 1]
            marks.append((file_time(song, plan.entries[i - 1] / sr),
                          f"IN <- {songs[i - 1].title}: {option.entry.label} ({option.kind})", "green"))
        if i < len(chosen):
            option = chosen[i]
            marks.append((file_time(song, plan.exits[i] / sr),
                          f"OUT -> {songs[i + 1].title}: {option.exit.label} ({option.kind})", "red"))
        stem = folder / f"{i + 1:02d} {_safe(song.title)}"
        labels = audacity_labels(song, cue, marks, ranges[i], file_time)
        stem.with_suffix(".txt").write_text(labels)
        if plt is not None and song.structure_features is not None:
            _timeline(plt, song, marks, ranges[i], stem.with_suffix(".png"))
    print(f"\nCue debug files written to {folder}")


def audacity_labels(song, cue, marks, played, file_time) -> str:
    """Audacity's label format: one label per line, "start<TAB>end<TAB>text" (start = end for a point)."""
    rows = []
    if song.structure is not None:
        rows += [(s.start, s.end, s.name) for s in song.structure.sections]
        rows += [(p, p, f"phrase {k + 1}") for k, p in enumerate(song.structure.phrases)]
    rows += [(file_time(song, c.time), file_time(song, c.time), f"exit? {c.label}") for c in cue.exits]
    rows += [(file_time(song, c.time), file_time(song, c.time), f"entry? {c.label}") for c in cue.entries]
    rows += [(t, t, text) for t, text, _ in marks]
    rows.append((played[0], played[1], "plays"))
    rows.sort(key=lambda row: (row[0], row[1]))
    return "".join(f"{start:.6f}\t{end:.6f}\t{text}\n" for start, end, text in rows)


def _timeline(plt, song, marks, played, path: Path) -> None:
    features, structure = song.structure_features, song.structure
    fig, ax = plt.subplots(figsize=(14, 3.2))
    for s in structure.sections:
        ax.axvspan(s.start, s.end, color=LABEL_COLOURS.get(s.label, "#dddddd"), alpha=0.5, linewidth=0)
        ax.text((s.start + s.end) / 2, 0.97, s.name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8)
    for p in structure.phrases:
        ax.axvline(p, color="grey", linewidth=0.5, linestyle=":")
    ax.plot(features.beat_times, features.energy_db, color="black", linewidth=0.8, label="loudness (dB)")
    ax.plot(features.beat_times, features.bass_db, color="#3182bd", linewidth=0.6, label="bass (dB)")
    ax.axvspan(played[0], played[1], ymin=0, ymax=0.04, color="black", label="plays")
    for t, text, colour in marks:
        ax.axvline(t, color=colour, linewidth=1.8)
        ax.text(t, 0.05, f" {text}", transform=ax.get_xaxis_transform(), color=colour, fontsize=8,
                rotation=90, va="bottom")
    ax.set_xlabel("seconds into the original file")
    ax.set_title(song.title, fontsize=10)
    ax.set_xlim(0, max(features.beat_times[-1], played[1]) + 1)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _safe(name: str) -> str:
    """A file name without characters that file systems object to."""
    return re.sub(r'[\\/:*?"<>|]', "_", name)[:80]
