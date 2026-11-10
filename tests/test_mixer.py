import numpy as np
import pytest

from audio_io import SAMPLE_RATE as SR
from mixer import mix_tracks, plan_crossfades, render_mix
from transitions import equal_power_gains


def test_output_length_is_sum_of_lengths_minus_crossfades(sine):
    tracks = [sine(220, 5.0), sine(330, 7.5), sine(440, 6.0)]
    mix = mix_tracks(tracks, crossfade=2 * SR)
    assert mix.crossfades == [2 * SR, 2 * SR]
    assert len(mix.audio) == sum(len(t) for t in tracks) - 2 * (2 * SR)
    assert mix.audio.shape[1] == 2


def test_each_track_starts_one_crossfade_before_the_previous_one_ends(sine):
    tracks = [sine(220, 5.0), sine(330, 7.5), sine(440, 6.0)]
    mix = mix_tracks(tracks, crossfade=2 * SR)
    # B starts 2 s before A (5 s) ends -> 3 s. C starts 2 s before B ends -> 3 + 7.5 - 2 = 8.5 s.
    assert mix.starts == [0, 3 * SR, round(8.5 * SR)]


def test_no_samples_above_1(sine):
    # Two loud tones at the same pitch line up in phase during the crossfade, so
    # their amplitudes add: 0.9 * (cos + sin) peaks at 0.9 * 1.41 = 1.27.
    tracks = [sine(440, 3.0, amplitude=0.9), sine(440, 3.0, amplitude=0.9)]
    raw, _ = render_mix(tracks, plan_crossfades([len(t) for t in tracks], SR))
    assert np.abs(raw).max() > 1.0  # without protection this mix would clip...

    mix = mix_tracks(tracks, crossfade=SR)
    assert np.abs(mix.audio).max() <= 1.0  # ...but the finished mix never does
    assert mix.safety_gain_db < 0  # and the mix reports that it was turned down


def test_crossfade_shrinks_instead_of_crashing_when_a_track_is_too_short(sine):
    tracks = [sine(220, 10.0), sine(330, 3.0), sine(440, 10.0)]
    mix = mix_tracks(tracks, crossfade=8 * SR)  # two 8 s fades won't fit in a 3 s song
    half = len(tracks[1]) // 2
    assert mix.crossfades == [half, half]  # each capped at half the short song: 1.5 s
    assert len(mix.audio) == sum(len(t) for t in tracks) - 2 * half


def test_tiny_tracks_do_not_crash(sine):
    tracks = [sine(220, 1.0), sine(330, 0.0001), sine(440, 1.0)]  # the middle one is 4 samples
    mix = mix_tracks(tracks, crossfade=8 * SR)
    assert mix.crossfades == [2, 2]
    assert len(mix.audio) == sum(len(t) for t in tracks) - 4


def test_single_track_comes_out_unchanged(sine):
    track = sine(440, 2.0)
    mix = mix_tracks([track], crossfade=8 * SR)
    assert mix.starts == [0] and mix.crossfades == [] and mix.safety_gain_db == 0.0
    np.testing.assert_array_equal(mix.audio, track)


def test_zero_crossfade_just_joins_the_tracks(sine):
    a, b = sine(220, 1.0), sine(330, 1.0)
    mix = mix_tracks([a, b], crossfade=0)
    np.testing.assert_array_equal(mix.audio, np.concatenate([a, b]))


def test_overlap_is_the_equal_power_blend_and_solo_parts_are_untouched():
    a = np.full((10_000, 2), 0.5, dtype=np.float32)
    b = np.full((10_000, 2), 0.25, dtype=np.float32)
    mix = mix_tracks([a, b], crossfade=4_000)  # B starts at 6,000; mix is 16,000 long

    fade_out, fade_in = equal_power_gains(4_000)
    np.testing.assert_allclose(mix.audio[6_000:10_000, 0], 0.5 * fade_out + 0.25 * fade_in, rtol=1e-6)
    assert np.all(mix.audio[:6_000] == 0.5)
    assert np.all(mix.audio[10_000:] == 0.25)


def test_mixing_nothing_is_an_error():
    with pytest.raises(ValueError):
        mix_tracks([], crossfade=SR)
