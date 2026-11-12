"""Ramp mode (--tempo-mode ramp): songs keep their own tempo, and change it only around transitions.

The audio tests run on click tracks, through the `stretcher` fixture: once with a
resampling stand-in that follows the time map exactly, and once more with Rubber
Band when it's installed.
"""

import re

import numpy as np
import pytest
import soundfile as sf

import automix
import tempo
from analysis import detect_beats
from audio_io import SAMPLE_RATE as SR
from mixer import _fit_plain_crossfades, mix_planned, plan_ramped
from tempo import SETTLE, STITCH, TimeMapBuilder, apply_time_map, transition_tempo

from conftest import find_hit_centroids, make_click_track, resample_time_map


def make_set(bpms, seconds=40.0):
    """Click-track 'songs' at these tempos: [(audio, true beat times in seconds)], and their analysis."""
    songs = [make_click_track(bpm, seconds=seconds, seed=int(bpm)) for bpm in bpms]
    return songs, [detect_beats(audio) for audio, _ in songs]


def plan_for(songs, found, meet="middle", max_stretch=0.08, crossfade_beats=16, ramp_beats=16):
    return plan_ramped([len(audio) for audio, _ in songs], [f.beats * SR for f in found],
                       [f.bpm for f in found], crossfade_beats=crossfade_beats, ramp_beats=ramp_beats,
                       meet=meet, max_stretch=max_stretch, fallback_crossfade=8 * SR)


def render(songs, plan, stretcher):
    tracks = [apply_time_map(audio, s.time_map, s.out_start, s.out_end, stretcher=stretcher)
              for (audio, _), s in zip(songs, plan.songs)]
    mix_plan = plan.mix_plan([len(t.audio) for t in tracks])
    return tracks, mix_plan, mix_planned([t.audio for t in tracks], mix_plan)


def hits_in_track(track, true_beats):
    """Where each song's real drum hits are in its rendered track (centre of energy, in samples)."""
    expected = track.positions(true_beats * SR)
    inside = (expected > 0.05 * SR) & (expected < len(track.audio) - 0.1 * SR)
    return find_hit_centroids(track.audio, expected[inside]), expected[inside]


@pytest.fixture(scope="module")
def three_songs():
    """120, 126 and 123 BPM, so the middle song has a ramp after its intro AND before its outro."""
    return make_set((120, 126, 123))


# The time map itself


def test_a_ramp_glides_smoothly_from_one_rate_to_the_other():
    builder = TimeMapBuilder()
    builder.hold(SR, 1.0)
    builder.ramp(SR + 8 * SR, 1.0, 1.05, n_anchors=64)  # 8 s of the original
    builder.hold(20 * SR, 1.05)
    time_map = builder.build()
    assert time_map.rate_at(SR // 2) == 1.0
    assert time_map.rate_at(SR + 10) == pytest.approx(1.0, abs=1e-4)  # starts at the old rate...
    assert time_map.rate_at(9 * SR - 10) == pytest.approx(1.05, abs=1e-4)  # ...ends at the new one
    rates = [time_map.rate_at(SR + k * SR // 8) for k in range(64)]
    assert np.all(np.diff(rates) >= 0) and max(np.diff(rates)) < 0.002  # in tiny, steady steps
    # The ramp lasts its input length divided by the average rate: 2L / (r0 + r1).
    assert time_map(9 * SR) - time_map(SR) == pytest.approx(2 * 8 * SR / 2.05, abs=1)


def test_transition_tempo_meets_in_the_middle_or_at_the_incoming_song():
    incoming = transition_tempo(120, 126, "incoming")
    assert incoming.bpm == 126 and incoming.rate_out == pytest.approx(1.05) and incoming.rate_in == 1.0
    middle = transition_tempo(120, 126, "middle")
    assert middle.bpm == pytest.approx(np.sqrt(120 * 126))
    assert middle.rate_out == pytest.approx(1 / middle.rate_in)  # the same ratio, opposite directions


def test_a_half_tempo_song_is_matched_two_beats_to_one():
    t = transition_tempo(170, 86, "middle")
    assert t.factor == 2.0  # 86 is matched as 172...
    assert t.stretch < 0.01  # ...so it barely needs to change


# Part A requirements


def test_the_middle_of_each_song_is_sample_identical_to_the_original(three_songs, stretcher):
    songs, found = three_songs
    plan = plan_for(songs, found)
    tracks, _, _ = render(songs, plan, stretcher)
    for (audio, _), track, song in zip(songs, tracks, plan.songs):
        assert len(track.untouched) == 1
        start, end, position = track.untouched[0]
        np.testing.assert_array_equal(track.audio[position : position + end - start], audio[start:end])
        # ...and the untouched part is everything outside the ramps and overlaps, apart
        # from the ~0.1 s beside them where the stretched audio is stitched back on.
        ramp_in = [r for r in song.ramps if r.rate_to == 1.0]
        ramp_out = [r for r in song.ramps if r.rate_from == 1.0]
        if ramp_in:
            assert start <= ramp_in[0].end + SETTLE + STITCH
        if ramp_out:
            assert end >= ramp_out[0].start - SETTLE - STITCH
    assert all(len(song.ramps) == n for song, n in zip(plan.songs, (1, 2, 1)))


def test_the_tempo_at_each_end_of_a_ramp_is_the_expected_bpm(three_songs, stretcher):
    songs, found = three_songs
    plan = plan_for(songs, found)
    tracks, _, _ = render(songs, plan, stretcher)
    for (_, true_beats), track, song, f in zip(songs, tracks, plan.songs, found):
        for ramp in song.ramps:
            in_ramp = true_beats[(true_beats * SR >= ramp.start) & (true_beats * SR <= ramp.end)]
            for beats, rate in ((in_ramp[:3], ramp.rate_from), (in_ramp[-3:], ramp.rate_to)):
                centres, _ = hits_in_track(track, beats)
                period = np.polyfit(np.arange(len(centres)), centres, 1)[0]  # samples per beat, heard
                assert 60 * SR / period == pytest.approx(f.bpm * rate, abs=0.25)


@pytest.mark.parametrize("meet", ["middle", "incoming"])
def test_beats_in_the_overlap_line_up_within_10ms(three_songs, stretcher, meet):
    songs, found = three_songs
    plan = plan_for(songs, found, meet=meet)
    tracks, mix_plan, mix = render(songs, plan, stretcher)
    assert plan.mix_plan().beat_aligned == [True, True]
    for i in range(2):
        a_hits, _ = hits_in_track(tracks[i], songs[i][1])
        b_hits, b_expected = hits_in_track(tracks[i + 1], songs[i + 1][1])
        overlap = b_expected < mix_plan.crossfades[i] - 0.1 * SR  # B's hits during the crossfade
        assert overlap.sum() >= 15
        a_in_mix = mix.starts[i] + a_hits
        gaps_ms = [np.min(np.abs(a_in_mix - (mix.starts[i + 1] + b))) / SR * 1000 for b in b_hits[overlap]]
        assert max(gaps_ms) < 10


def test_meeting_in_the_middle_stretches_each_song_about_half_as_much():
    songs, found = make_set((120, 126), seconds=30)
    middle = plan_for(songs, found, meet="middle").transitions[0].tempo
    incoming = plan_for(songs, found, meet="incoming").transitions[0].tempo
    assert incoming.rate_in == 1.0  # with "incoming", B starts at its own tempo
    assert abs(middle.rate_out - 1) == pytest.approx(abs(incoming.rate_out - 1) / 2, rel=0.05)
    assert abs(middle.rate_in - 1) == pytest.approx(abs(incoming.rate_out - 1) / 2, rel=0.05)
    assert middle.stretch == pytest.approx(incoming.stretch / 2, rel=0.05)


def test_a_transition_beyond_max_stretch_falls_back_to_a_plain_crossfade():
    songs, found = make_set((120, 122, 150), seconds=30)
    plan = plan_for(songs, found, max_stretch=0.08)
    assert [t.aligned for t in plan.transitions] == [True, False]
    assert plan.transitions[1].reason == "stretch"
    assert plan.transitions[1].tempo.stretch > 0.08  # 122 and 150 would need ~11% each
    assert plan.mix_plan().crossfades[1] == 8 * SR  # the plain fallback crossfade
    # Only that transition falls back: 122 still ramps into its first transition, but
    # it plays its outro at its own tempo, and 150 is never stretched at all.
    assert [len(s.ramps) for s in plan.songs] == [1, 1, 0]
    assert plan.songs[2].time_map.stretched_regions(0, len(songs[2][0])) == []


def test_ramps_shrink_to_fit_a_short_song(stretcher):
    # The middle song is 12 s long: its intro and outro overlaps take 16 beats each,
    # which leaves too little room for two full 16-beat ramps.
    songs = [make_click_track(120, 30, seed=1), make_click_track(126, 12, seed=2),
             make_click_track(123, 30, seed=3)]
    found = [detect_beats(audio) for audio, _ in songs]
    plan = plan_for(songs, found, crossfade_beats=8)
    middle = plan.songs[1]
    ramp_in, ramp_out = middle.ramps
    full = 16 * 60 / found[1].bpm * SR
    assert ramp_in.end - ramp_in.start < full and ramp_out.end - ramp_out.start < full
    assert ramp_in.end <= ramp_out.start  # shortened so they don't overlap
    render(songs, plan, stretcher)  # and it renders without trouble


def test_plain_crossfades_fit_around_beat_matched_ones():
    # Track 1 (10,000 samples) already gives 7,000 to a beat-matched fade-in, so a plain
    # fade-out can only have the remaining 3,000 (not half the track).
    assert _fit_plain_crossfades([20_000, 10_000, 20_000], [7_000, None], fallback=8_000) == [7_000, 3_000]
    assert _fit_plain_crossfades([20_000, 10_000], [None], fallback=8_000) == [5_000]  # half rule


# Through the command line


def write_songs(folder, bpms, seconds=24):
    folder.mkdir()
    for bpm in bpms:
        sf.write(folder / f"s{bpm}.wav", make_click_track(bpm, seconds=seconds, seed=bpm)[0], SR)
    return folder


@pytest.fixture
def ramp_ready(monkeypatch):
    """Ramp mode, whether or not Rubber Band is installed (the resampling stand-in fills in)."""
    if not tempo.ramp_stretch_available():
        monkeypatch.setattr(automix, "ramp_stretch_available", lambda: True)
        monkeypatch.setattr(tempo, "stretch_time_map", resample_time_map)


def test_ramp_mode_end_to_end(tmp_path, ramp_ready, capsys):
    folder = write_songs(tmp_path / "songs", (120, 126))
    out = tmp_path / "mix.wav"
    assert automix.main([str(folder), "-o", str(out), "--order", "given"]) == 0
    stdout = capsys.readouterr().out
    assert "ramp mode" in stdout
    assert "s120 -> s126: beat-matched at 123.0 BPM, 16 beats" in stdout
    assert "s120 +2.5%, s126 -2.4%" in stdout
    audio, sr = sf.read(out)
    assert sr == SR and np.abs(audio).max() <= 1.0
    assert 35 < len(audio) / sr < 45  # two ~23 s songs overlapping by ~8 s


def test_ramp_mode_warns_and_falls_back_to_a_plain_crossfade_beyond_max_stretch(tmp_path, ramp_ready, capsys):
    folder = write_songs(tmp_path / "songs", (120, 150))
    assert automix.main([str(folder), "-o", str(tmp_path / "mix.wav"), "--order", "given"]) == 0
    captured = capsys.readouterr()
    assert "s120 -> s150: plain 8.0 s crossfade (needs a 11.8% tempo change; limit 8%)" in captured.out
    assert re.search(r"warning: s120 .* would need a 11\.8% tempo change \(limit 8%\)", captured.err)


def test_without_rubber_band_ramp_mode_falls_back_to_global(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(automix, "ramp_stretch_available", lambda: False)
    folder = write_songs(tmp_path / "songs", (120, 126))
    assert automix.main([str(folder), "-o", str(tmp_path / "mix.wav")]) == 0
    captured = capsys.readouterr()
    assert "using --tempo-mode global instead" in captured.err
    assert "mixing at 123.0 BPM" in captured.out


def test_clicks_follow_the_ramps(tmp_path, ramp_ready):
    """--clicks in ramp mode: one click per played beat, on the beats as they are after stretching."""
    folder = write_songs(tmp_path / "songs", (120, 126))
    plain, clicked = tmp_path / "plain.wav", tmp_path / "clicked.wav"
    assert automix.main([str(folder), "-o", str(plain), "--order", "given"]) == 0
    assert automix.main([str(folder), "-o", str(clicked), "--order", "given", "--clicks"]) == 0
    p, c = sf.read(plain)[0][:, 0], sf.read(clicked)[0][:, 0]
    clicks = c - p * (np.dot(c, p) / np.dot(p, p))  # take away the music (the mixes differ only in gain)
    loud = np.flatnonzero(np.abs(clicks) > 0.3 * np.abs(clicks).max())
    starts = loud[np.concatenate([[True], np.diff(loud) > 0.1 * SR])]
    gaps = np.diff(starts) / SR
    assert len(starts) > 70  # ~47 + ~45 beats, with the 16 in the overlap clicking together
    assert np.all((gaps > 0.45) & (gaps < 0.52))  # a click every beat (120 to 126 BPM), none missed


def test_bpm_means_global_mode(tmp_path):
    args = automix.parse_args([str(tmp_path), "--bpm", "124"])
    assert args.tempo_mode == "global"
    assert automix.parse_args([str(tmp_path)]).tempo_mode == "ramp"
    with pytest.raises(SystemExit):
        automix.parse_args([str(tmp_path), "--bpm", "124", "--tempo-mode", "ramp"])
