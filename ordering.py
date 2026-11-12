"""Smart ordering: which song should follow which (Stage 3).

transition_cost() scores how well song A flows into song B (lower is better), from
three things a DJ listens for:

    tempo    how much the transition has to stretch the songs (tempo.py)
    key      how well their keys go together (the Camelot wheel, analysis.py)
    energy   whether the intensity jumps or drops suddenly

order_songs() then finds the order of the whole set with the lowest total cost,
optionally following an energy shape (building up, or peaking two-thirds through).

All the weights and costs are in the block just below: that's the place to change
how much each one matters.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np

from analysis import camelot
from tempo import transition_tempo

# Costs and weights
# =================


@dataclass(frozen=True)
class Weights:
    """How much each part counts in a transition's cost (and the energy shape in a set's)."""

    tempo: float = 1.0
    key: float = 1.0
    energy: float = 1.0
    shape: float = 1.0  # --energy rise/peak: distance from the target energy curve


WEIGHTS = Weights()

# Key: cost of each relationship on the Camelot wheel, before weighting.
KEY_COSTS = {
    "same": 0.0,  # the same key
    "neighbour": 0.15,  # +-1 on the wheel: a fifth apart, 6 of 7 notes shared
    "relative": 0.15,  # same number, A <-> B: relative major/minor, all 7 notes shared
    "boost": 0.4,  # +2 on the wheel: up a whole tone, the "energy boost" mix
    "clash": 1.0,  # anything else
}

# Tempo: a stretch of TEMPO_FULL_COST_PERCENT costs 1.0, as much as a key clash.
TEMPO_FULL_COST_PERCENT = 8.0
UNMIXABLE_COST = 10.0  # beyond --max-stretch: it would fall back to a plain crossfade
NO_BEAT_COST = 0.5  # a song with no steady beat can only ever get a plain crossfade

# Energy: a jump of ENERGY_FULL_COST_JUMP costs 1.0. The cost grows with the SQUARE of
# the jump, so small steps are nearly free and big drops or spikes are expensive.
# Energies span 0..1 across the set (analysis.energy_scores), so 0.75 means a jump
# across three-quarters of the set's range; a 0.25 step costs only 0.11.
ENERGY_FULL_COST_JUMP = 0.75

# --energy rise/peak: being SHAPE_FULL_COST_DISTANCE away from the target energy costs 1.0.
SHAPE_FULL_COST_DISTANCE = 0.35
PEAK_AT = 2 / 3  # --energy peak: where in the set the energy peaks
PEAK_END = 0.3  # --energy peak: the target at the very end, as a fraction of the set's range

# Labels for the tracklist: a transition costing less than these is "perfect"/"good",
# otherwise "risky" (a sure key clash alone costs 1.0, a stretch past the limit 10).
LABELS = ((0.3, "perfect"), (0.8, "good"))

BRUTE_FORCE_MAX = 8  # up to this many songs, try every order (8! = 40,320)
TWO_OPT_ROUTES = 10  # beyond that: improve this many of the best greedy routes with 2-opt


@dataclass(frozen=True)
class Track:
    """What ordering needs to know about a song (no audio)."""

    name: str
    bpm: float | None = None
    key: str | None = None  # e.g. "A minor"
    key_confidence: float = 0.0
    energy: float | None = None  # 0..1, relative to the set (analysis.energy_scores)


@dataclass(frozen=True)
class TempoSettings:
    """How the mix will change tempo, which decides what a transition's stretch is."""

    mode: str = "ramp"  # "ramp", "global", or "none" (--no-beatmatch: plain crossfades only)
    meet: str = "middle"  # ramp mode's --meet
    max_stretch: float = 0.08
    target: float | None = None  # global mode's mix tempo


@dataclass(frozen=True)
class TransitionCost:
    total: float
    tempo: float
    key: float
    energy: float
    stretch: float | None  # the bigger of the two songs' tempo changes (0.03 = 3%); None if no beat
    beat_matched: bool  # False: it will be a plain crossfade
    key_relation: str | None  # "same", "neighbour", ..., or None if a key is unknown

    @property
    def label(self) -> str:
        return label_for(self.total)


def label_for(cost: float) -> str:
    return next((label for limit, label in LABELS if cost < limit), "risky")


def key_relation(camelot_a: str, camelot_b: str) -> str:
    """How two Camelot keys relate: "same", "neighbour", "relative", "boost" or "clash"."""
    number_a, letter_a = int(camelot_a[:-1]), camelot_a[-1]
    number_b, letter_b = int(camelot_b[:-1]), camelot_b[-1]
    step = (number_b - number_a) % 12  # steps clockwise round the wheel, from A to B
    if letter_a != letter_b:
        return "relative" if step == 0 else "clash"
    return {0: "same", 1: "neighbour", 11: "neighbour", 2: "boost"}.get(step, "clash")


def transition_stretch(a: Track, b: Track, settings: TempoSettings) -> tuple[float | None, bool]:
    """(stretch, beat_matched) for A -> B under these settings.

    The stretch is the bigger of the two songs' tempo changes as a fraction, worked
    out exactly as the mix will do it: ramp mode meets at tempo.transition_tempo
    (which also tries matching at half or double tempo); global mode stretches both
    songs to the mix tempo. None means a song has no steady beat.
    """
    if settings.mode == "none":
        return 0.0, False
    if a.bpm is None or b.bpm is None:
        return None, False
    if settings.mode == "ramp":
        stretch = transition_tempo(a.bpm, b.bpm, settings.meet).stretch
    else:
        if settings.target is None:
            raise ValueError("global mode needs the mix tempo (TempoSettings.target)")
        stretch = max(abs(settings.target / a.bpm - 1), abs(settings.target / b.bpm - 1))
    return stretch, stretch <= settings.max_stretch


def transition_cost(a: Track, b: Track, settings: TempoSettings, weights: Weights = WEIGHTS) -> TransitionCost:
    """How well A flows into B: a weighted sum of tempo, key and energy costs (lower = better)."""
    stretch, matched = transition_stretch(a, b, settings)
    if settings.mode == "none":
        tempo = 0.0  # plain crossfades everywhere: no stretching, whatever the order
    elif stretch is None:
        tempo = NO_BEAT_COST
    elif not matched:
        tempo = UNMIXABLE_COST
    else:
        tempo = stretch * 100 / TEMPO_FULL_COST_PERCENT

    relation, key = None, 0.0
    camelot_a, camelot_b = camelot(a.key), camelot(b.key)
    if camelot_a and camelot_b:
        relation = key_relation(camelot_a, camelot_b)
        # An unsure key detection makes the key cost unreliable, so it counts for less.
        key = KEY_COSTS[relation] * min(a.key_confidence, b.key_confidence)

    energy = 0.0
    if a.energy is not None and b.energy is not None:
        energy = ((b.energy - a.energy) / ENERGY_FULL_COST_JUMP) ** 2

    total = weights.tempo * tempo + weights.key * key + weights.energy * energy
    return TransitionCost(total, tempo, key, energy, stretch, matched, relation)


def energy_target(position: int, count: int, shape: str) -> float | None:
    """The target energy (0 = the set's lowest, 1 = its highest) at `position` of `count`.

    rise: a straight climb from lowest to highest.
    peak: climb to the highest at PEAK_AT (2/3) of the way through, then cool down
          to PEAK_END. None for "any": no target.
    """
    if shape == "any":
        return None
    x = position / (count - 1) if count > 1 else 1.0
    if shape == "rise":
        return x
    if shape == "peak":
        return x / PEAK_AT if x <= PEAK_AT else 1 - (1 - PEAK_END) * (x - PEAK_AT) / (1 - PEAK_AT)
    raise ValueError(f"unknown energy shape {shape!r}")


# Finding the best order
# ======================
#
# Why it's hard
#   Think of the songs as the towns on a map and each transition cost as the
#   distance from one town to the next: the best set is the shortest route that
#   visits every town once. That's the Traveling Salesman Problem. There are n!
#   orders of n songs (n x (n-1) x ... x 1): 8 songs have 40,320, which a computer
#   tries in a blink, but 15 have over a trillion, and nobody knows a fast way that
#   is guaranteed to find the best one. So beyond BRUTE_FORCE_MAX songs we use a
#   heuristic: a method that finds a very good order quickly, without a guarantee.
#
# Greedy nearest neighbour
#   Start somewhere, and always go next to the song that's cheapest to move to.
#   It's fast, but short-sighted: an early cheap choice can force an expensive
#   one later. We try every starting song.
#
# 2-opt
#   Then improve it: take any stretch of the order and play it backwards. On a map,
#   that removes two "roads" and reconnects the route the other way round, which
#   untangles a route that crosses itself:
#
#       ... A -> [B -> C -> D] -> E ...   becomes   ... A -> [D -> C -> B] -> E ...
#
#   Keep any reversal that lowers the total cost, and repeat until none does. The
#   result is "2-optimal": no single reversal can improve it. (Our costs aren't
#   quite symmetric, since an energy boost only works going up, so we recompute
#   the whole route's cost for each candidate rather than just the two new joins.)
#
#   The best greedy route doesn't always improve into the best 2-optimal one, so we
#   2-opt the TWO_OPT_ROUTES cheapest greedy routes and keep the winner. On random
#   sets of 7-8 songs, compared with brute force: improving only the best greedy
#   route came out up to 17% worse than the true best; improving the 10 best, at
#   most 2.6% worse (and usually exactly the best). 60 songs still take ~0.3 s.
#
# The energy shape adds a cost for each song's distance from the target curve AT ITS
# POSITION in the set, so it's part of the same total that both methods minimize.


@dataclass(frozen=True)
class Ordering:
    order: list[int]  # indices into the tracks, in playing order
    cost: float  # total: transitions plus energy shape
    method: str  # "brute force" or "greedy + 2-opt"


def order_songs(
    tracks: list[Track],
    settings: TempoSettings,
    energy_shape: str = "any",
    start: int | None = None,
    weights: Weights = WEIGHTS,
    pair_costs: np.ndarray | None = None,
) -> Ordering:
    """The lowest-cost order of `tracks`, opening with `start` if it's given.

    `pair_costs[i, j]`, if given, replaces transition_cost() as the cost of song i
    into song j (Stage 4 passes the cost of each pair's best cue points).
    """
    costs, positions = cost_matrices(tracks, settings, energy_shape, weights, pair_costs)
    if len(tracks) <= BRUTE_FORCE_MAX:
        return brute_force_order(costs, positions, start)
    return heuristic_order(costs, positions, start)


def cost_matrices(tracks: list[Track], settings: TempoSettings, energy_shape: str = "any",
                  weights: Weights = WEIGHTS, pair_costs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """costs[i, j]: song i into song j. positions[i, k]: song i placed k-th (energy shape)."""
    n = len(tracks)
    if pair_costs is not None:
        costs = np.array(pair_costs, dtype=float)
        np.fill_diagonal(costs, 0.0)
    else:
        costs = np.zeros((n, n))
        for i, j in itertools.permutations(range(n), 2):
            costs[i, j] = transition_cost(tracks[i], tracks[j], settings, weights).total
    positions = np.zeros((n, n))
    energies = [t.energy for t in tracks if t.energy is not None]
    if energy_shape != "any" and energies:
        low, high = min(energies), max(energies)
        for k in range(n):
            target = low + (high - low) * energy_target(k, n, energy_shape)
            for i, track in enumerate(tracks):
                if track.energy is not None:
                    positions[i, k] = weights.shape * ((track.energy - target) / SHAPE_FULL_COST_DISTANCE) ** 2
    return costs, positions


def route_cost(order, costs: np.ndarray, positions: np.ndarray) -> float:
    order = np.asarray(order)
    return float(costs[order[:-1], order[1:]].sum() + positions[order, np.arange(len(order))].sum())


def brute_force_order(costs: np.ndarray, positions: np.ndarray, start: int | None = None) -> Ordering:
    """Try every order (with `start` first, if given) and keep the cheapest."""
    n = len(costs)
    if n == 0:
        return Ordering([], 0.0, "brute force")
    c, p = costs.tolist(), positions.tolist()  # plain lists: much faster than numpy for tiny lookups
    best, best_cost = list(range(n)), math.inf
    firsts = [start] if start is not None else range(n)
    for first in firsts:
        rest = [i for i in range(n) if i != first]
        for tail in itertools.permutations(rest):
            order = (first, *tail)
            total = p[first][0]
            for k in range(1, n):
                total += c[order[k - 1]][order[k]] + p[order[k]][k]
                if total >= best_cost:
                    break  # already worse than the best: no need to finish adding it up
            else:
                best, best_cost = list(order), total
    return Ordering(best, best_cost, "brute force")


def heuristic_order(costs: np.ndarray, positions: np.ndarray, start: int | None = None) -> Ordering:
    """Greedy nearest neighbour from every starting song, then 2-opt on the best few."""
    n = len(costs)
    if n == 0:
        return Ordering([], 0.0, "greedy + 2-opt")
    firsts = [start] if start is not None else range(n)
    routes = sorted((nearest_neighbour(costs, positions, first) for first in firsts),
                    key=lambda route: route_cost(route, costs, positions))
    improved = [two_opt(route, costs, positions, keep_first=start is not None) for route in routes[:TWO_OPT_ROUTES]]
    best = min(improved, key=lambda route: route_cost(route, costs, positions))
    return Ordering(best, route_cost(best, costs, positions), "greedy + 2-opt")


def nearest_neighbour(costs: np.ndarray, positions: np.ndarray, first: int) -> list[int]:
    """Start at `first`, then keep going to whichever song is cheapest to play next."""
    order, left = [first], set(range(len(costs))) - {first}
    while left:
        k = len(order)
        following = min(left, key=lambda s: (costs[order[-1], s] + positions[s, k], s))
        order.append(following)
        left.remove(following)
    return order


def two_opt(order: list[int], costs: np.ndarray, positions: np.ndarray, keep_first: bool = False) -> list[int]:
    """Reverse stretches of the order while that lowers its cost (see "2-opt" above)."""
    order = np.array(order)
    best = route_cost(order, costs, positions)
    n, improved = len(order), True
    while improved:
        improved = False
        for i in range(1 if keep_first else 0, n - 1):
            for j in range(i + 1, n):
                candidate = np.concatenate([order[:i], order[i : j + 1][::-1], order[j + 1 :]])
                cost = route_cost(candidate, costs, positions)
                if cost < best - 1e-9:
                    order, best, improved = candidate, cost, True
    return order.tolist()
