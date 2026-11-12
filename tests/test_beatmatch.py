"""Beat-aligned transitions: planning, alignment accuracy, fallbacks and clicks."""

from pathlib import Path

import numpy as np
import pytest

from analysis import detect_beats
from audio_io import SAMPLE_RATE as SR
from automix import Song, match_tempos, stretch_songs
from mixer import mix_planned, plan_beatmatched
from processing import TARGET_RMS_DB, normalize_loudness, rms_db
from tempo import rubberband_available, stretch_beat_times, stretch_rate, target_bpm, time_stretch

PERIOD = 22_050.0  # samples per beat at 120 BPM


def grid(n_beats, period=PERIOD, offset=0.0):
    """Perfectly regular beat positions, in samples."""
    return offset + period * np.arange(n_beats)


def mix_starts(plan):
    """Where each track's played part starts in the mix (the same layout render_mix uses)."""
    starts = [0]
    for (start, end), overlap in zip(plan.cuts, plan.crossfades):
        starts.append(starts[-1] + (end - start) - overlap)
    return starts


def to_mix(plan, i, positions):
    """Sample positions in track i -> sample positions in the mix."""
    return mix_starts(plan)[i] + np.asarray(positions) - plan.cuts[i][0]


def run_stage2(songs):
    """Detect -> stretch to the median tempo -> plan, the way automix.py does.

    `songs` holds (audio, true_beat_times) pairs. Returns the plan, the stretched
    tracks, where each track's TRUE beats are after stretching (in samples), and the
    beat period.
    """
    found = [detect_beats(audio) for audio, _ in songs]
    target = target_bpm([f.bpm for f in found])
    tracks, beats, hits = [], [], []
    for (audio, true_times), f in zip(songs, found):
        rate = stretch_rate(f.bpm, target)
        tracks.append(time_stretch(audio, rate))
        beats.append(stretch_beat_times(f.beats, rate) * SR)  # detected beats, moved by math
        hits.append(true_times / rate * SR)  # where the real hits now are
    period = 60.0 / target * SR
    plan = plan_beatmatched([len(t) for t in tracks], beats, [True] * len(songs), period,
                            crossfade_beats=16, fallback_crossfade=8 * SR)
    return plan, tracks, hits, period


def in_overlap(plan, period, positions_b):
    """Which of track B's positions fall inside the A -> B crossfade (a True/False mask)."""
    start = mix_starts(plan)[1]
    b = to_mix(plan, 1, positions_b)
    return (b >= start - period / 2) & (b < start + plan.crossfades[0] - period / 2)


def worst_gap_ms(a_positions, b_positions):
    """For each B position, the distance to the nearest A position; return the worst, in ms."""
    return max(np.min(np.abs(a_positions - b)) for b in b_positions) / SR * 1000


def assert_beats_in_sync(plan, hits, period):
    assert plan.beat_aligned == [True]
    b_hits = to_mix(plan, 1, hits[1][in_overlap(plan, period, hits[1])])
    assert len(b_hits) == 16  # a 16-beat crossfade...
    assert worst_gap_ms(to_mix(plan, 0, hits[0]), b_hits) < 10  # ...with every beat in sync


@pytest.mark.parametrize("bpm_a, bpm_b", [(120, 126), (124, 128), (128, 122)])
def test_beats_line_up_within_10ms_through_the_overlap(click_track, bpm_a, bpm_b):
    plan, _, hits, period = run_stage2([click_track(bpm, 30, seed=bpm) for bpm in (bpm_a, bpm_b)])
    assert_beats_in_sync(plan, hits, period)


def test_beats_line_up_even_where_the_tracker_slipped_onto_offbeats(syncopated_track):
    # The tracker puts the LAST 5 beats of the first song, and the FIRST 8 beats of
    # the second, a 16th note off: exactly where the mix points are read.
    plan, _, hits, period = run_stage2([syncopated_track(127.5, seed=3),
                                        syncopated_track(128.2, seed=4)])
    assert_beats_in_sync(plan, hits, period)


def test_drum_hits_in_the_stretched_audio_line_up_too(click_track, hit_centroids):
    """The same check on the stretched audio itself: where each hit's energy lands.

    This adds the time-stretcher's own effect on drum hits (see tempo.py). Measured
    over six tempo pairs, Rubber Band kept every hit within 2 ms of its partner, but
    librosa's phase vocoder smears hits: typically a few ms apart, occasionally up to
    18 ms. So the bar depends on which stretcher is installed.
    """
    plan, tracks, hits, period = run_stage2([click_track(bpm, 30, seed=bpm) for bpm in (120, 126)])
    a_centres = to_mix(plan, 0, hit_centroids(tracks[0], hits[0]))
    b_centres = to_mix(plan, 1, hit_centroids(tracks[1], hits[1][in_overlap(plan, period, hits[1])]))
    assert len(b_centres) == 16
    gaps_ms = [np.min(np.abs(a_centres - b)) / SR * 1000 for b in b_centres]
    if rubberband_available():
        assert max(gaps_ms) < 5
    else:
        assert np.median(gaps_ms) < 10 and max(gaps_ms) < 20


def test_plan_cuts_each_track_at_its_mix_points():
    a, b = grid(60, offset=1000), grid(60, offset=500)
    lengths = [int(a[-1] + 3 * PERIOD), int(b[-1] + 3 * PERIOD)]  # 3 beats of audio after the last beat
    plan = plan_beatmatched(lengths, [a, b], [True, True], PERIOD, 16, 8 * SR)
    assert plan.beat_aligned == [True]
    assert plan.crossfades == [round(16 * PERIOD)]
    assert plan.cuts[0] == (0, round(a[-1]))  # A plays up to its last beat...
    assert plan.cuts[1] == (500, lengths[1])  # ...and B starts right on its first beat


def test_mix_points_ignore_beats_the_tracker_put_on_an_offbeat():
    a, b = grid(60), grid(60, offset=1000)
    a_seen, b_seen = a.copy(), b.copy()
    a_seen[-5:] += PERIOD / 4  # the tracker put A's last 5 beats a 16th note late
    b_seen[:8] += PERIOD / 4  # and B's first 8
    lengths = [int(a[-1] + 3 * PERIOD), int(b[-1] + 3 * PERIOD)]
    plan = plan_beatmatched(lengths, [a_seen, b_seen], [True, True], PERIOD, 16, 8 * SR)
    assert plan.cuts[0][1] == round(a[-1])  # the crossfade still ends on A's true last beat
    assert plan.cuts[1][0] == round(b[0])  # and starts on B's true first beat


def test_stretched_songs_still_enter_the_mix_at_the_target_loudness():
    # The phase vocoder loses ~3 dB on noisy sounds, so stretch_songs measures again.
    noise = 0.1 * np.random.default_rng(0).standard_normal((10 * SR, 2))
    song = Song(Path("noisy.wav"), audio=normalize_loudness(noise.astype(np.float32)),
                bpm=120.0, beats=grid(20) / SR, rate=1.05, matched=True)
    stretch_songs([song])
    assert rms_db(song.audio) == pytest.approx(TARGET_RMS_DB, abs=0.1)


def test_crossfade_shrinks_in_beats_for_a_short_song():
    beats = [grid(60), grid(12), grid(60)]  # the middle song has only 12 beats
    lengths = [int(b[-1] + PERIOD) for b in beats]
    plan = plan_beatmatched(lengths, beats, [True] * 3, PERIOD, 16, 8 * SR)
    assert plan.beat_aligned == [True, True]
    assert plan.crossfades == [round(5 * PERIOD)] * 2  # (12 - 2) // 2 = 5 beats each
    # render_mix refuses fades that overlap inside a track, so this would raise if they did:
    mix_planned([np.zeros((n, 2), np.float32) for n in lengths], plan)


def test_a_fade_out_that_would_overlap_the_fade_in_falls_back_instead_of_crashing(monkeypatch):
    # Force the worst case: the middle song's mix-out lands inside its own fade-in.
    import mixer

    beats = [grid(60), grid(40), grid(60)]
    lengths = [int(b[-1] + PERIOD) for b in beats]
    real_mix_out = mixer._mix_out_beat
    monkeypatch.setattr(mixer, "_mix_out_beat",
                        lambda b, n, p: 0.0 if len(b) == 40 else real_mix_out(b, n, p))
    plan = plan_beatmatched(lengths, beats, [True] * 3, PERIOD, 16, 8 * SR)
    assert plan.beat_aligned == [True, False]  # the second transition fell back to plain
    mix_planned([np.zeros((n, 2), np.float32) for n in lengths], plan)  # and renders fine


def test_too_few_beats_means_a_plain_crossfade():
    beats = [grid(60), grid(3)]
    lengths = [int(b[-1] + PERIOD) for b in beats]
    plan = plan_beatmatched(lengths, beats, [True, True], PERIOD, 16, SR)
    assert plan.beat_aligned == [False]
    assert plan.crossfades == [min(SR, lengths[1] // 2)]


def test_a_song_past_max_stretch_keeps_its_tempo_and_gets_plain_crossfades():
    songs = [Song(Path(f"{bpm}.wav"), bpm=bpm, beats=grid(60) / SR) for bpm in (120.0, 121.0, 140.0)]
    target = match_tempos(songs, target=None, max_stretch=0.08)
    assert target == 121.0  # the median
    assert [s.matched for s in songs] == [True, True, False]  # 140 would need -13.6%
    assert songs[2].rate == 1.0  # so it isn't stretched at all

    beats = [s.beats * SR for s in songs]
    lengths = [int(b[-1] + PERIOD) for b in beats]
    plan = plan_beatmatched(lengths, beats, [s.matched for s in songs], PERIOD, 16, 8 * SR)
    assert plan.beat_aligned == [True, False]
    assert plan.crossfades[1] == 8 * SR  # the Stage 1 crossfade, 8 seconds


def test_songs_already_at_the_mix_tempo_are_left_unstretched():
    songs = [Song(Path("a.wav"), bpm=124.0, beats=grid(40) / SR),
             Song(Path("b.wav"), bpm=124.01, beats=grid(40) / SR)]  # 0.008% apart
    match_tempos(songs, target=None, max_stretch=0.08)
    assert all(s.matched for s in songs)
    assert [s.rate for s in songs] == [1.0, 1.0]  # not worth a stretch


def test_the_max_stretch_limit_decides_who_gets_stretched():
    def songs():  # 100 BPM and 107 BPM; reaching 100 takes the second one -6.5%
        return [Song(Path("a.wav"), bpm=100.0, beats=grid(40) / SR),
                Song(Path("b.wav"), bpm=107.0, beats=grid(40) / SR)]

    within = songs()
    match_tempos(within, target=100.0, max_stretch=0.08)
    assert [s.matched for s in within] == [True, True]
    assert within[1].rate == pytest.approx(100 / 107)

    beyond = songs()
    match_tempos(beyond, target=100.0, max_stretch=0.05)
    assert [s.matched for s in beyond] == [True, False]
    assert beyond[1].rate == 1.0

    match_tempos(within, target=100.0, max_stretch=0.05)  # re-deciding resets old choices
    assert [s.matched for s in within] == [True, False] and within[1].rate == 1.0


def test_clicks_land_on_every_played_beat():
    beats = [grid(40, offset=1000), grid(40, offset=1000)]
    lengths = [int(b[-1] + PERIOD) for b in beats]
    plan = plan_beatmatched(lengths, beats, [True, True], PERIOD, 16, 8 * SR)
    silence = [np.zeros((n, 2), np.float32) for n in lengths]
    mix = mix_planned(silence, plan, clicks=beats)

    # Silence plus clicks: a click starts wherever sound follows a stretch of silence.
    loud = np.flatnonzero(np.abs(mix.audio[:, 0]) > 0.01)
    click_starts = loud[np.concatenate([[True], np.diff(loud) > 100])]
    played = [beats[i][(beats[i] >= a) & (beats[i] < b)] for i, (a, b) in enumerate(plan.cuts)]
    expected = np.unique(np.round(np.concatenate([to_mix(plan, i, p) for i, p in enumerate(played)])))
    # During the crossfade both songs click on the very same samples, so 16 clicks are shared.
    assert len(expected) == len(played[0]) + len(played[1]) - 16
    assert len(click_starts) == len(expected)
    assert np.max(np.abs(click_starts - expected)) <= 2  # within 2 samples of every beat
