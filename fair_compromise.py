"""
Fair Compromise Algorithm for Survey / Preference Data
======================================================

Finds a balanced, reciprocal compromise between two or more groups.
NOT a simple average: core priorities of each group receive extra weight,
while differences are compressed to prevent any single group from dominating.

Design goals
------------
- Both groups can recognise their key priorities in the outcome.
- Each group concedes on lower-priority issues, not on its core concerns.
- Asymmetric but reciprocal: the compromise leans toward each group's
  spearpoints rather than forcing a single uniform midpoint.

Usage
-----
    from fair_compromise import FairCompromiseAlgorithm, Group, Topic

    topics = [Topic("Housing", scale=10), Topic("Climate", scale=10)]
    groups = [
        Group("Left",  importances=[4, 9], positions=[8, 9]),
        Group("Right", importances=[8, 3], positions=[3, 4]),
    ]
    algo = FairCompromiseAlgorithm(topics, groups)
    result = algo.run()
    result.print_report()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Topic:
    """One survey topic / policy area."""
    name: str
    scale: float = 10.0          # maximum value on the response scale


@dataclass
class Group:
    """One respondent group with their importance and (optionally) position scores."""
    name: str
    importances: list[float]     # how much does the group care? (0 … scale)
    positions: Optional[list[float]] = None   # preferred outcome per topic (0 … scale)
                                              # if None, importances ARE the positions


# ---------------------------------------------------------------------------
# Internal result containers
# ---------------------------------------------------------------------------

@dataclass
class TopicResult:
    topic: Topic
    raw_importances: dict[str, float]        # original scores
    norm_importances: dict[str, float]       # normalised within group (sum = 1)
    relative_importances: dict[str, float]   # compared to own group mean (mean = 1)
    compressed_weights: dict[str, float]     # after difference compression
    spearpoint_for: list[str]                # group names for which this is a spearpoint
    positions: dict[str, float]              # preferred positions per group
    compromise_position: float               # final compromise value
    explanation: str


@dataclass
class CompromiseResult:
    topic_results: list[TopicResult]
    groups: list[Group]
    topics: list[Topic]
    spearpoint_threshold: float
    contrast_threshold: float
    spearpoint_bonus: float

    def print_report(self) -> None:
        _print_report(self)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class FairCompromiseAlgorithm:
    """
    Parameters
    ----------
    topics              : list of Topic objects
    groups              : list of Group objects (≥ 2)
    spearpoint_threshold: a topic is a spearpoint for a group when its
                          relative importance (own score / own mean) exceeds
                          this value. Default 1.4 (40 % above average).
    contrast_threshold  : additionally, the ratio of relative importances
                          between groups must exceed this value. Default 1.5.
    spearpoint_bonus    : extra weight multiplier applied to a group's pull
                          on its spearpoint topics. Default 0.6.
    compression_power   : exponent for compressing raw differences.
                          0.5 = square-root (strong compression),
                          1.0 = no compression. Default 0.55.
    """

    def __init__(
        self,
        topics: list[Topic],
        groups: list[Group],
        spearpoint_threshold: float = 1.4,
        contrast_threshold: float = 1.5,
        spearpoint_bonus: float = 0.6,
        compression_power: float = 0.55,
    ):
        if len(groups) < 2:
            raise ValueError("At least two groups are required.")
        if any(len(g.importances) != len(topics) for g in groups):
            raise ValueError("Each group must have one importance score per topic.")

        self.topics = topics
        self.groups = groups
        self.n_topics = len(topics)
        self.spearpoint_threshold = spearpoint_threshold
        self.contrast_threshold = contrast_threshold
        self.spearpoint_bonus = spearpoint_bonus
        self.compression_power = compression_power

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> CompromiseResult:
        norm_imp = self._normalise_importances()
        rel_imp = self._relative_importances(norm_imp)
        positions = self._resolve_positions()

        topic_results = []
        for i, topic in enumerate(self.topics):
            result = self._process_topic(i, topic, norm_imp, rel_imp, positions)
            topic_results.append(result)

        return CompromiseResult(
            topic_results=topic_results,
            groups=self.groups,
            topics=self.topics,
            spearpoint_threshold=self.spearpoint_threshold,
            contrast_threshold=self.contrast_threshold,
            spearpoint_bonus=self.spearpoint_bonus,
        )

    # ------------------------------------------------------------------
    # Step 1 – Normalise importances (each group sums to 1)
    # ------------------------------------------------------------------

    def _normalise_importances(self) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        for g in self.groups:
            total = sum(g.importances)
            if total == 0:
                raise ValueError(f"Group '{g.name}' has all-zero importances.")
            result[g.name] = [v / total for v in g.importances]
        return result

    # ------------------------------------------------------------------
    # Step 2 – Relative importance (each topic vs. own group mean)
    # ------------------------------------------------------------------

    def _relative_importances(
        self, norm_imp: dict[str, list[float]]
    ) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        for g in self.groups:
            mean = 1.0 / self.n_topics          # mean of uniform distribution
            result[g.name] = [v / mean for v in norm_imp[g.name]]
        return result

    # ------------------------------------------------------------------
    # Step 3 – Resolve positions
    # ------------------------------------------------------------------

    def _resolve_positions(self) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        for g in self.groups:
            if g.positions is not None:
                if len(g.positions) != self.n_topics:
                    raise ValueError(
                        f"Group '{g.name}' has {len(g.positions)} positions "
                        f"but {self.n_topics} topics."
                    )
                result[g.name] = list(g.positions)
            else:
                # importance IS the position
                result[g.name] = list(g.importances)
        return result

    # ------------------------------------------------------------------
    # Step 4 – Per-topic compromise
    # ------------------------------------------------------------------

    def _process_topic(
        self,
        i: int,
        topic: Topic,
        norm_imp: dict[str, list[float]],
        rel_imp: dict[str, list[float]],
        positions: dict[str, list[float]],
    ) -> TopicResult:

        raw_imp = {g.name: g.importances[i] for g in self.groups}
        n_imp = {g.name: norm_imp[g.name][i] for g in self.groups}
        r_imp = {g.name: rel_imp[g.name][i] for g in self.groups}
        pos = {g.name: positions[g.name][i] for g in self.groups}

        # --- Identify spearpoints ------------------------------------
        spearpoint_for = self._identify_spearpoints(r_imp)

        # --- Compress raw importances --------------------------------
        # Square-root-like compression prevents extreme scores from
        # completely dominating; we compress the normalised score.
        comp = {
            g.name: math.pow(n_imp[g.name], self.compression_power)
            for g in self.groups
        }

        # --- Spearpoint bonus ----------------------------------------
        # When a topic is a spearpoint for a group, that group's weight
        # is amplified proportionally to how strongly it is a spearpoint.
        weights = dict(comp)  # start from compressed importances
        for g in self.groups:
            if g.name in spearpoint_for:
                excess = r_imp[g.name] - 1.0   # how much above average
                weights[g.name] *= (1.0 + self.spearpoint_bonus * excess)

        # --- Weighted compromise position ----------------------------
        total_weight = sum(weights.values())
        compromise = sum(weights[g.name] * pos[g.name] for g in self.groups) / total_weight

        # --- Build explanation ----------------------------------------
        explanation = self._build_explanation(
            topic, spearpoint_for, r_imp, n_imp, comp, weights, pos, compromise,
            compression_power=self.compression_power,
        )

        return TopicResult(
            topic=topic,
            raw_importances=raw_imp,
            norm_importances=n_imp,
            relative_importances=r_imp,
            compressed_weights=weights,
            spearpoint_for=spearpoint_for,
            positions=pos,
            compromise_position=compromise,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Spearpoint detection
    # ------------------------------------------------------------------

    def _identify_spearpoints(self, r_imp: dict[str, float]) -> list[str]:
        """
        A topic is a spearpoint for group X when:
          1. Its relative importance for X exceeds spearpoint_threshold.
          2. The ratio rel_imp[X] / rel_imp[Y] for every other group Y
             exceeds contrast_threshold.
        """
        spearpoints = []
        group_names = [g.name for g in self.groups]
        for name in group_names:
            if r_imp[name] < self.spearpoint_threshold:
                continue
            others = [n for n in group_names if n != name]
            if all(
                r_imp[name] / max(r_imp[other], 1e-9) >= self.contrast_threshold
                for other in others
            ):
                spearpoints.append(name)
        return spearpoints

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def _build_explanation(
        self,
        topic: Topic,
        spearpoint_for: list[str],
        r_imp: dict[str, float],
        n_imp: dict[str, float],
        comp: dict[str, float],
        weights: dict[str, float],
        pos: dict[str, float],
        compromise: float,
        compression_power: Optional[float] = None,
    ) -> str:
        if compression_power is None:
            compression_power = self.compression_power
        parts = []

        if not spearpoint_for:
            # Check whether groups are close
            rel_values = list(r_imp.values())
            max_ratio = max(rel_values) / max(min(rel_values), 1e-9)
            if max_ratio < self.contrast_threshold:
                parts.append(
                    "Gedeeld belang: beide groepen hechten vergelijkbaar veel waarde "
                    "aan dit onderwerp. De compromispositie ligt dicht bij het midden."
                )
            else:
                parts.append(
                    "Neutraal onderwerp: de groepen verschillen, maar geen enkele groep "
                    "beschouwt dit als kernprioriteit. Compressie vlakt het verschil af."
                )
        else:
            for name in spearpoint_for:
                parts.append(
                    f"Speerpunt van '{name}' (relatief belang {r_imp[name]:.2f}×): "
                    f"deze groep krijgt extra gewicht (w={weights[name]:.3f}) "
                    f"op haar voorkeurspositie ({pos[name]:.1f})."
                )
            non_spear = [g.name for g in self.groups if g.name not in spearpoint_for]
            for name in non_spear:
                parts.append(
                    f"'{name}' heeft hier lager relatief belang ({r_imp[name]:.2f}×) "
                    f"en krijgt minder trekkracht (w={weights[name]:.3f}), "
                    f"maar telt nog altijd mee."
                )

        parts.append(
            f"Compressie (macht {compression_power:.2f}) "
            "voorkomt dat absolute scores één groep volledig laten domineren."
        )
        parts.append(f"Compromispositie: {compromise:.2f} / {topic.scale:.0f}.")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _print_report(result: CompromiseResult) -> None:
    sep = "=" * 72
    thin = "-" * 72

    print(sep)
    print("  FAIR COMPROMISE REPORT")
    print(sep)
    print(f"  Groepen : {', '.join(g.name for g in result.groups)}")
    print(f"  Thema's : {', '.join(t.name for t in result.topics)}")
    print(f"  Speerpunt-drempel     : relatief belang > {result.spearpoint_threshold}")
    print(f"  Contrastverhouding    : ratio > {result.contrast_threshold}")
    print(f"  Speerpunt-bonus       : +{result.spearpoint_bonus * 100:.0f}% gewicht")
    print(sep)

    for tr in result.topic_results:
        print(f"\n  THEMA: {tr.topic.name.upper()}")
        print(thin)

        # Priorities
        print("  Prioriteiten (origineel → genormaliseerd → relatief):")
        for g in result.groups:
            n = g.name
            print(
                f"    {n:15s}  "
                f"raw={tr.raw_importances[n]:5.1f}  "
                f"norm={tr.norm_importances[n]:.3f}  "
                f"rel={tr.relative_importances[n]:.2f}×"
            )

        # Positions
        print("  Voorkeursposities:")
        for g in result.groups:
            n = g.name
            print(f"    {n:15s}  pos={tr.positions[n]:.1f} / {tr.topic.scale:.0f}")

        # Spearpoint status
        if tr.spearpoint_for:
            print(f"  Speerpunt van: {', '.join(tr.spearpoint_for)}")
        else:
            print("  Speerpunt van: (geen – gedeeld of neutraal)")

        # Weights
        print("  Gewichten na compressie + bonussen:")
        total_w = sum(tr.compressed_weights.values())
        for g in result.groups:
            n = g.name
            share = tr.compressed_weights[n] / total_w * 100
            print(f"    {n:15s}  w={tr.compressed_weights[n]:.4f}  ({share:.1f}%)")

        # Outcome
        print(f"  ► Compromispositie: {tr.compromise_position:.2f} / {tr.topic.scale:.0f}")
        print(f"  Toelichting:")
        for sentence in tr.explanation.split(" | "):
            print(f"    • {sentence}")

    print()
    print(sep)
    print("  SAMENVATTEND OVERZICHT")
    print(sep)
    header = f"  {'Thema':<20}" + "".join(
        f"  {g.name:>8}" for g in result.groups
    ) + "  Compromis"
    print(header)
    print(thin)
    for tr in result.topic_results:
        sp_marker = ""
        if tr.spearpoint_for:
            sp_marker = f"  [SP: {'+'.join(tr.spearpoint_for)}]"
        row = f"  {tr.topic.name:<20}"
        for g in result.groups:
            row += f"  {tr.positions[g.name]:>8.1f}"
        row += f"  {tr.compromise_position:>8.2f}{sp_marker}"
        print(row)
    print(sep)
    print()
