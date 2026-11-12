"""Beat detection, BPM refinement, half/double-tempo correction and caching."""

import os

import numpy as np
import pytest

from analysis import AnalysisCache, BeatAnalysis, apply_octave, detect_beats, fit_beat_grid, octave_factors


@pytest.mark.parametrize("bpm", [120, 126])
def test_detects_bpm_within_1_bpm(click_track, bpm):
    audio, _ = click_track(bpm, seed=bpm)
    assert detect_beats(audio).bpm == pytest.approx(bpm, abs=1.0)


def test_bpm_is_refined_beyond_librosas_frame_grid(click_track):
    # librosa alone reads this 128 BPM track as 129.2: it picks the tempo from whole
    # 11.6 ms frames. The straight-line fit through all the beats gets it right.
    audio, _ = click_track(128, seed=128)
    assert detect_beats(audio).bpm == pytest.approx(128, abs=0.05)


@pytest.mark.parametrize("bpm, seed", [(125.4, 0), (127.5, 3)])
def test_loud_offbeat_hits_dont_pull_the_tempo_off(syncopated_track, bpm, seed):
    # Modelled on a real song that first measured 126.6 BPM instead of 125.4: loud
    # sounds between the beats lured a loosely held beat tracker off the grid
    # (the first case), or librosa's coarse tempo led it astray (the second case
    # reads as 126.05 without the sharpening in step 2).
    audio, _ = syncopated_track(bpm, seed=seed)
    assert detect_beats(audio).bpm == pytest.approx(bpm, abs=0.1)


def test_beat_times_land_on_the_hits(click_track):
    audio, true_beats = click_track(120, seed=120)
    found = detect_beats(audio).beats
    errors = np.array([np.min(np.abs(true_beats - t)) for t in found])
    assert len(found) >= len(true_beats) - 3  # the tracker may drop the very first/last beat
    assert np.median(errors) < 0.005  # 5 ms, although each frame is 11.6 ms long
    assert np.percentile(errors, 90) < 0.010


@pytest.mark.parametrize("bpm", [100, 126, 174])
def test_fitted_beat_grid_lands_within_5ms_of_every_hit(click_track, bpm):
    # Within 5 ms per song keeps two songs within 10 ms of each other. This relies on
    # snapping each beat to its onset peak (126 BPM is ~7 ms off without it).
    audio, true_beats = click_track(bpm, seconds=24, seed=bpm)
    first, period, numbers = fit_beat_grid(detect_beats(audio).beats)
    grid = first + numbers * period
    assert max(np.min(np.abs(true_beats - t)) for t in grid) < 0.005


def test_silence_steady_tones_and_noise_have_no_tempo(sine):
    noise = 0.1 * np.random.default_rng(0).standard_normal((20 * 44_100, 2))
    assert detect_beats(np.zeros((5 * 44_100, 2), dtype=np.float32)).bpm is None
    assert detect_beats(sine(440, 20.0)).bpm is None  # librosa alone "finds" ~48 beats here
    assert detect_beats(noise.astype(np.float32)).bpm is None


def test_beat_grid_fit_survives_a_missed_beat_and_a_misplaced_one():
    beats = 1.0 + 0.5 * np.arange(40)  # a beat every 0.5 s (120 BPM), starting at 1 s
    beats = np.delete(beats, 17)  # the tracker missed one...
    beats[-1] -= 0.15  # ...and put the last one 150 ms early
    first, period, numbers = fit_beat_grid(beats)
    assert period == pytest.approx(0.5, abs=1e-6)
    assert first == pytest.approx(1.0, abs=1e-6)
    assert numbers[17] == 18  # the gap was counted as two beats


def test_a_half_tempo_reading_next_to_128_bpm_songs_is_doubled():
    assert octave_factors([128.0, 127.5, 64.2]) == [1.0, 1.0, 2.0]


def test_a_double_tempo_reading_is_halved():
    assert octave_factors([70.0, 71.0, 141.0]) == [1.0, 1.0, 0.5]


def test_a_one_against_one_tie_is_settled_toward_120_bpm():
    assert octave_factors([64.0, 128.0]) == [2.0, 1.0]


def test_an_unrelated_song_does_not_hijack_the_consensus():
    # 90 BPM relates to none of the others, so it "needs no corrections" as a
    # reference; the consensus must still be the tempo most songs agree with.
    assert octave_factors([120.0, 126.0, 64.0, 90.0]) == [1.0, 1.0, 2.0, 1.0]


def test_genuinely_different_tempos_are_left_alone():
    assert octave_factors([128.0, 90.0]) == [1.0, 1.0]  # 90 isn't half of anything here
    assert octave_factors([128.0, None, 126.0]) == [1.0, 1.0, 1.0]  # no tempo, no change


def test_octave_correction_against_a_given_tempo():
    assert octave_factors([140.0, 72.0], reference=70.0) == [0.5, 1.0]


def test_doubling_adds_the_missing_beats_halfway_and_halving_drops_every_other():
    doubled = apply_octave(BeatAnalysis(64.0, np.array([0.0, 1.0, 2.0])), 2.0)
    assert doubled.bpm == 128.0
    np.testing.assert_allclose(doubled.beats, [0.0, 0.5, 1.0, 1.5, 2.0])
    halved = apply_octave(BeatAnalysis(256.0, np.arange(6.0)), 0.5)
    assert halved.bpm == 128.0 and halved.beats.tolist() == [0.0, 2.0, 4.0]


def test_a_64_bpm_song_among_128_bpm_songs_end_to_end(click_track):
    readings = [detect_beats(click_track(bpm, seed=seed)[0])
                for bpm, seed in ((128, 1), (128, 2), (64, 3))]
    assert readings[2].bpm == pytest.approx(64, abs=1)  # it genuinely reads as 64...
    factors = octave_factors([r.bpm for r in readings])
    assert factors == [1.0, 1.0, 2.0]
    fixed = apply_octave(readings[2], factors[2])
    assert fixed.bpm == pytest.approx(128, abs=0.5)  # ...and is corrected to 128
    assert len(fixed.beats) == 2 * len(readings[2].beats) - 1


def test_cache_saves_reloads_and_notices_edited_files(tmp_path):
    song = tmp_path / "song.wav"
    song.write_bytes(b"pretend audio")
    cache_file = tmp_path / "cache.json"
    AnalysisCache(cache_file).put(song, BeatAnalysis(127.98, np.array([0.5, 0.97, 1.44])))

    reloaded = AnalysisCache(cache_file).get(song)  # a fresh object reads the JSON file
    assert reloaded.bpm == 127.98
    np.testing.assert_allclose(reloaded.beats, [0.5, 0.97, 1.44])

    stat = song.stat()  # "edit" the song: its modification time changes
    os.utime(song, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert AnalysisCache(cache_file).get(song) is None


def test_a_damaged_cache_file_is_ignored_and_replaced(tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("{this is not json")
    song = tmp_path / "song.wav"
    song.write_bytes(b"pretend audio")
    cache = AnalysisCache(cache_file)
    assert cache.get(song) is None
    cache.put(song, BeatAnalysis(None, np.zeros(0)))
    assert AnalysisCache(cache_file).get(song).bpm is None


def _onset_envelope(bpm, tresillo=0.0, seconds=120):
    """A made-up onset curve: a hit on every beat, plus tresillo hits (3/4 and 3/2 of a beat into every 2)."""
    from analysis import HOP_LENGTH

    fps = 44_100 / HOP_LENGTH
    beat, env = fps * 60 / bpm, np.zeros(int(seconds * fps))
    for k in range(int(len(env) / beat)):
        env[int(round(k * beat))] += 1.0
    for cycle in range(int(len(env) / (2 * beat))):
        for offset in (0.75, 1.5):
            i = int(round((2 * cycle + offset) * beat))
            if i < len(env):
                env[i] += tresillo
    return np.convolve(env, np.hanning(5), mode="same")


@pytest.mark.parametrize("tresillo", [0.0, 0.5, 0.8])
def test_a_tresillo_song_read_at_4_3_of_its_tempo_is_corrected(tresillo):
    # A real 95 BPM Punjabi pop song first read as 126.5: its 3+3+2 rhythm puts hits
    # 3/4 of a beat apart. Over the whole song the real beat repeats more strongly.
    from analysis import _check_three_quarters

    assert _check_three_quarters(_onset_envelope(95, tresillo), 44_100, 95 * 4 / 3) == pytest.approx(95, abs=0.2)


def test_the_3_4_check_leaves_a_steady_beat_and_an_ambiguous_one_alone():
    from analysis import _check_three_quarters

    assert _check_three_quarters(_onset_envelope(126.67), 44_100, 126.67) == 126.67  # nothing at 3/4
    assert _check_three_quarters(_onset_envelope(95, 1.3), 44_100, 126.67) == 126.67  # no clear evidence: keep it
