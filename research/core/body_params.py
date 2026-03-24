"""
research/core/body_params.py
-----------------------------
Standalone BodyParams implementation for the body-pattern-research repo.

This module is self-contained and doesn't depend on diverse-body-pattern-adjuster package.

Installed together, diverse-body-pattern-adjuster's
BodyParams is the canonical version. This module exists so that:
  - body-pattern-research works as a completely independent research repo
  - Tests run without needing the adjuster package installed
  - Researchers can clone and run this repo in isolation

The two definitions are kept in sync intentionally. If you change the
Level taxonomy or add axes in diverse-body-pattern-adjuster, mirror
the changes here.

Integration path
----------------
If diverse-body-pattern-adjuster IS installed, ScanToParams will
automatically use its BodyParams. This file is only the fallback.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum


class Level(IntEnum):
    """
    Universal grading scale for body parameter axes.

    0 = no deviation from a standard base block
    1 = mild deviation (e.g. ~8–15mm adjustment)
    2 = moderate deviation
    3 = high deviation
    4 = maximum supported deviation

    Using discrete levels (rather than raw centimetres) means:
    - adjustments are deterministic and reproducible
    - no personally identifiable measurements are stored
    - the same level maps to the same semantic meaning
      regardless of brand base block or garment type
    """
    NONE = 0
    MILD = 1
    MOD  = 2
    HIGH = 3
    MAX  = 4


@dataclass
class BodyParams:
    """
    Complete body parameter set for a single garment adjustment.

    Each axis is a discrete Level (0–4). Only axes that deviate from
    the standard base block need to be set — defaults are all NONE (0).

    Axes are grouped by body region:

    General
    -------
    height_level      : overall height deviation below reference (1.68m)
    weight_level      : torso/overall volume grading

    Upper body
    ----------
    shoulder_width_level : bilateral shoulder breadth
    bust_level           : circumference above waist
    upper_arm_level      : bilateral sleeve width

    Lower body
    ----------
    hip_level    : circumference below waist
    inseam_level : trouser length / crotch depth (negative = shorter)
    thigh_level  : circumference at widest thigh point

    Asymmetry (left/right independent)
    -----------------------------------
    left_upper_arm_step  : left arm volume (prosthesis socket side)
    right_upper_arm_step : right arm volume
    left_leg_step        : left leg length/volume
    right_leg_step       : right leg length/volume

    Postural
    --------
    seated_back_rise   : additional back-rise for wheelchair / seated use
    forward_tilt_level : lumbar curve / forward trunk tilt

    Notes field
    -----------
    notes : free-text annotation (not used by adjustment engine)
    """

    # General
    height_level: Level = Level.NONE
    weight_level: Level = Level.NONE

    # Upper body
    shoulder_width_level: Level = Level.NONE
    bust_level:           Level = Level.NONE
    upper_arm_level:      Level = Level.NONE

    # Lower body
    hip_level:    Level = Level.NONE
    inseam_level: Level = Level.NONE
    thigh_level:  Level = Level.NONE

    # Asymmetry
    left_upper_arm_step:  Level = Level.NONE
    right_upper_arm_step: Level = Level.NONE
    left_leg_step:        Level = Level.NONE
    right_leg_step:       Level = Level.NONE

    # Postural
    seated_back_rise:   Level = Level.NONE
    forward_tilt_level: Level = Level.NONE

    notes: str = ""

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to plain dict (int values, not Level enums)."""
        return {k: int(v) if isinstance(v, Level) else v
                for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "BodyParams":
        """Deserialise from a plain dict."""
        filtered = {k: Level(v) if k != "notes" else v
                    for k, v in d.items()
                    if k in cls.__dataclass_fields__}
        return cls(**filtered)

    def active_params(self) -> dict:
        """Return only axes that differ from the default (NONE/0)."""
        return {k: v for k, v in self.to_dict().items()
                if v not in (0, Level.NONE, "")}

    def as_vector(self) -> "list[int]":
        """Return a flat integer vector of all axes (excluding notes)."""
        return [int(getattr(self, k))
                for k in self.__dataclass_fields__
                if k != "notes"]

    def __repr__(self) -> str:
        active = self.active_params()
        if not active:
            return "BodyParams(standard)"
        parts = ", ".join(f"{k}={v}" for k, v in active.items())
        return f"BodyParams({parts})"
