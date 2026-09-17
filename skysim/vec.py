"""Minimal 3D vector math.

World frame is a local ENU tangent frame: x=east, y=north, z=up (metres),
with the origin on the surface. Gravity is central (see physics.py), so the
same frame carries both low-altitude flight and orbital motion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    __rmul__ = __mul__

    def __truediv__(self, s: float) -> "Vec3":
        return Vec3(self.x / s, self.y / s, self.z / s)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, o: "Vec3") -> float:
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o: "Vec3") -> "Vec3":
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    @property
    def mag(self) -> float:
        return math.sqrt(self.dot(self))

    @property
    def mag2(self) -> float:
        return self.dot(self)

    def unit(self) -> "Vec3":
        m = self.mag
        return ZERO if m < 1e-12 else self / m

    def clamp(self, limit: float) -> "Vec3":
        m = self.mag
        return self if m <= limit or m < 1e-12 else self * (limit / m)

    def perp_to(self, axis: "Vec3") -> "Vec3":
        """Component of self perpendicular to `axis`."""
        a = axis.unit()
        return self - a * self.dot(a)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    @staticmethod
    def from_any(v) -> "Vec3":
        if isinstance(v, Vec3):
            return v
        if isinstance(v, dict):
            return Vec3(float(v.get("x", 0.0)), float(v.get("y", 0.0)), float(v.get("z", 0.0)))
        x, y, z = v
        return Vec3(float(x), float(y), float(z))


ZERO = Vec3()
EAST = Vec3(1.0, 0.0, 0.0)
NORTH = Vec3(0.0, 1.0, 0.0)
UP = Vec3(0.0, 0.0, 1.0)
