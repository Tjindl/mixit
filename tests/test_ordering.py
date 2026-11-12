"""Transition costs and finding the best order (ordering.py). No audio needed: just analysis data."""

import itertools

import numpy as np
import pytest

from analysis import NOTE_NAMES
from ordering import (
    KEY_COSTS,
    UNMIXABLE_COST,
    TempoSettings,
    Track,
    brute_force_order,
    cost_matrices,
    energy_target,
    heuristic_order,
    key_relation,
    order_songs,
    route_cost,
    transition_cost,
)

RAMP = TempoSettings("ramp", "middle", 0.08)


def song(name="s", bpm=124.0, key="A minor", confidence=1.0, energy=0.5):
    return Track(name, bpm, key, confidence, energy)


# Costs


def test_key_relations_on_the_camelot_wheel():
    assert key_relation("8A", "8A") == "same"
    assert key_relation("8A", "9A") == key_relation("8A", "7A") == "neighbour"
    assert key_relation("12B", "1B") == "neighbour"  # the wheel wraps round
    assert key_relation("8A", "8B") == "relative"
    assert key_relation("8A", "10A") == "boost"
    assert key_relation("10A", "8A") == "clash"  # the boost only works going up
    assert key_relation("8A", "3B") == "clash"


def test_same_key_costs_less_than_a_neighbour_which_costs_less_than_a_distant_key():
    a = song(key="A minor")  # 8A
    same = transition_cost(a, song(key="A minor"), RAMP).total
    neighbour = transition_cost(a, song(key="E minor"), RAMP).total  # 9A
    relative = transition_cost(a, song(key="C major"), RAMP).total  # 8B
    boost = transition_cost(a, song(key="B minor"), RAMP).total  # 10A
    distant = transition_cost(a, song(key="Eb minor"), RAMP).total  # 2A, the far side of the wheel
    assert same < neighbour < boost < distant
    assert relative == neighbour  # both share almost every note
    assert same == 0.0


def test_unsure_keys_count_for_less():
    clash_sure = transition_cost(song(key="A minor"), song(key="Eb minor"), RAMP).key
    clash_unsure = transition_cost(song(key="A minor", confidence=0.2), song(key="Eb minor"), RAMP).key
    assert clash_sure == KEY_COSTS["clash"]
    assert clash_unsure == pytest.approx(0.2 * clash_sure)
    assert transition_cost(song(key=None), song(key="Eb minor"), RAMP).key == 0.0  # no key: no key cost


def test_a_smaller_tempo_gap_costs_less():
    a = song(bpm=124.0)
    costs = [transition_cost(a, song(bpm=bpm), RAMP).tempo for bpm in (124.0, 125.0, 127.0, 130.0)]
    assert costs[0] == 0.0
    assert costs == sorted(costs) and len(set(costs)) == 4


def test_meeting_in_the_middle_costs_less_tempo_than_matching_the_incoming_song():
    a, b = song(bpm=120.0), song(bpm=126.0)
    middle = transition_cost(a, b, TempoSettings("ramp", "middle", 0.08))
    incoming = transition_cost(a, b, TempoSettings("ramp", "incoming", 0.08))
    assert middle.tempo < incoming.tempo
    assert middle.stretch == pytest.approx(incoming.stretch / 2, rel=0.05)


def test_global_mode_tempo_cost_is_the_stretch_to_the_mix_tempo():
    settings = TempoSettings("global", max_stretch=0.08, target=125.0)
    cost = transition_cost(song(bpm=120.0), song(bpm=126.0), settings)
    assert cost.stretch == pytest.approx(125 / 120 - 1)  # the bigger of the two songs' stretches


def test_a_transition_beyond_max_stretch_costs_a_lot():
    cost = transition_cost(song(bpm=120.0), song(bpm=150.0), RAMP)
    assert not cost.beat_matched and cost.tempo == UNMIXABLE_COST
    assert cost.label == "risky"


def test_half_and_double_tempo_count_as_close():
    # An 86 BPM song mixes into a 170 BPM one two beats to one: a ~1% change, not 50%.
    cost = transition_cost(song(bpm=170.0), song(bpm=86.0), RAMP)
    assert cost.beat_matched and cost.stretch < 0.01


def test_big_energy_jumps_cost_much_more_than_small_steps():
    small = transition_cost(song(energy=0.5), song(energy=0.6), RAMP).energy
    big = transition_cost(song(energy=0.5), song(energy=0.9), RAMP).energy
    assert big == pytest.approx(16 * small)  # 4x the jump, 16x the cost
    drop = transition_cost(song(energy=0.9), song(energy=0.5), RAMP).energy
    assert drop == big  # sudden drops are penalized like spikes


def test_labels():
    assert transition_cost(song(), song(), RAMP).label == "perfect"
    assert transition_cost(song(key="A minor"), song(key="Eb minor", bpm=127.0), RAMP).label == "risky"


# Ordering


def assert_monotonic(values):
    assert values == sorted(values) or values == sorted(values, reverse=True)


def test_brute_force_chains_tempos_in_order():
    tempos = [126.0, 120.0, 130.0, 124.0, 122.0, 128.0]
    tracks = [song(f"s{bpm:g}", bpm=bpm) for bpm in tempos]
    result = order_songs(tracks, RAMP)
    assert result.method == "brute force"
    assert_monotonic([tracks[i].bpm for i in result.order])  # no doubling back


def test_brute_force_walks_round_the_camelot_wheel():
    keys = ["B minor", "A minor", "F# minor", "E minor"]  # 10A, 8A, 11A, 9A
    tracks = [song(key, key=key) for key in keys]
    order = [tracks[i].key for i in order_songs(tracks, RAMP).order]
    assert order in (["A minor", "E minor", "B minor", "F# minor"],  # 8A 9A 10A 11A
                     ["F# minor", "B minor", "E minor", "A minor"])  # or the other way round


def test_start_forces_the_opening_song():
    tracks = [song(f"s{bpm:g}", bpm=bpm, energy=e) for bpm, e in ((120.0, 0.2), (122.0, 0.4), (124.0, 0.5),
                                                                  (126.0, 0.7), (128.0, 0.9))]
    for heuristic in (False, True):
        costs, positions = cost_matrices(tracks, RAMP)
        result = (heuristic_order if heuristic else brute_force_order)(costs, positions, start=2)
        assert result.order[0] == 2
        best = min(route_cost((2, *rest), costs, positions) for rest in itertools.permutations([0, 1, 3, 4]))
        assert result.cost == pytest.approx(best)  # the best of the orders that open with song 2


@pytest.mark.parametrize("shape", ["rise", "peak"])
def test_energy_shapes(shape):
    energies = [0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8]
    tracks = [song(f"e{e}", energy=e) for e in energies]
    order = [tracks[i].energy for i in order_songs(tracks, RAMP, energy_shape=shape).order]
    if shape == "rise":
        assert order == sorted(energies)
    else:
        peak = order.index(max(order))
        assert peak == 4  # 2/3 of the way through 7 songs (positions 0..6)
        assert order[:peak + 1] == sorted(order[:peak + 1])  # building up...
        assert order[peak:] == sorted(order[peak:], reverse=True)  # ...then cooling down


def test_energy_target_curves():
    assert [energy_target(k, 5, "rise") for k in range(5)] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert energy_target(4, 7, "peak") == pytest.approx(1.0)
    assert energy_target(0, 7, "peak") == 0.0 and energy_target(6, 7, "peak") == pytest.approx(0.3)
    assert energy_target(3, 7, "any") is None


def test_brute_force_really_finds_the_cheapest_order():
    rng = np.random.default_rng(5)
    costs, positions = rng.uniform(0, 1, (6, 6)), rng.uniform(0, 0.5, (6, 6))
    best = min(route_cost(p, costs, positions) for p in itertools.permutations(range(6)))
    assert brute_force_order(costs, positions).cost == pytest.approx(best)


def random_set(rng, n):
    return [Track(f"s{i}", bpm=float(rng.uniform(118, 132)),
                  key=f"{NOTE_NAMES[rng.integers(12)]} {rng.choice(['major', 'minor'])}",
                  key_confidence=float(rng.uniform(0.3, 1.0)), energy=float(rng.uniform(0, 1))) for i in range(n)]


@pytest.mark.parametrize("shape", ["any", "peak"])
def test_the_heuristic_stays_close_to_brute_force(shape):
    rng = np.random.default_rng(11)
    ratios = []
    for _ in range(25):
        costs, positions = cost_matrices(random_set(rng, int(rng.integers(7, 9))), RAMP, shape)
        ratios.append(heuristic_order(costs, positions).cost / brute_force_order(costs, positions).cost)
    assert min(ratios) >= 1 - 1e-9  # it can't beat the true best...
    assert max(ratios) < 1.10  # ...but it's never far off
    assert np.mean(ratios) < 1.02  # and usually it finds it


def test_big_sets_use_the_heuristic_and_respect_start():
    rng = np.random.default_rng(3)
    tracks = random_set(rng, 20)
    result = order_songs(tracks, RAMP, energy_shape="rise", start=7)
    assert result.method == "greedy + 2-opt"
    assert result.order[0] == 7 and sorted(result.order) == list(range(20))
