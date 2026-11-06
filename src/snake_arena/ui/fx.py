"""Visual effects helpers: cached glow sprites, particles, floating text.

Pure pygame, no assets: glow sprites are radial-alpha surfaces rendered once
per (color, quantized radius) and cached; particles and score popups are
tiny dataclass lists updated per frame. Everything degrades gracefully at
small cell sizes and is safe under the dummy video driver (headless tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pygame

# -- glow sprites ---------------------------------------------------------------

_GLOW_CACHE: dict[tuple[tuple[int, int, int], int, int], pygame.Surface] = {}


def glow_sprite(color, radius: int, peak_alpha: int = 70) -> pygame.Surface:
    """A soft radial glow surface for ``color``; cached and cheap to blit.

    ``radius`` should already be quantized by the caller (int) so the cache
    stays small even for pulsing effects.
    """
    key = (tuple(color)[:3], int(radius), int(peak_alpha))
    cached = _GLOW_CACHE.get(key)
    if cached is not None:
        return cached
    radius = int(radius)
    size = radius * 2
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    # Concentric circles from the rim inward, alpha ramping to peak: smooth
    # enough at these radii and an order of magnitude faster than per-pixel.
    steps = max(4, min(10, radius))
    for i in range(steps, 0, -1):
        t = i / steps  # 1 at rim -> ~0 at core
        alpha = int(peak_alpha * (1.0 - t) ** 1.6)
        if alpha <= 0:
            continue
        pygame.draw.circle(surf, (*key[0], alpha), (radius, radius), max(1, int(radius * t + 1)))
    if len(_GLOW_CACHE) > 512:  # paranoia bound; cache keys are quantized
        _GLOW_CACHE.clear()
    _GLOW_CACHE[key] = surf
    return surf


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_pos(p0, p1, t: float) -> tuple[float, float]:
    return (lerp(p0[0], p1[0], t), lerp(p0[1], p1[1], t))


def blend(c0, c1, t: float) -> tuple[int, int, int]:
    """Linear blend of two RGB colors, clamped to valid channel range.

    Callers may pass t outside [0, 1] (e.g. particle life ratios), so the
    result is clamped channel-wise rather than trusting the inputs.
    """
    return tuple(max(0, min(255, int(lerp(a, b, t))))
                 for a, b in zip(tuple(c0)[:3], tuple(c1)[:3], strict=True))


def _rand(seed: int) -> float:
    """Deterministic pseudo-random in [0, 1) - keeps replays visually stable."""
    return ((seed * 2654435761) % 100000) / 100000.0


# -- particles -------------------------------------------------------------------

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    color: tuple[int, int, int]


@dataclass
class FloatingText:
    x: float
    y: float
    content: str
    color: tuple[int, int, int]
    life: float = 0.9
    max_life: float = 0.9

    @property
    def alpha(self) -> int:
        t = self.life / self.max_life
        return int(200 * min(1.0, t * 1.6))


@dataclass
class EffectsLayer:
    """Particles + floating text for one screen; update/draw each frame."""

    particles: list[Particle] = field(default_factory=list)
    texts: list[FloatingText] = field(default_factory=list)

    def burst(
        self,
        pos,
        color,
        *,
        count: int = 14,
        speed: float = 90.0,
        size: float = 3.0,
        life: float = 0.55,
        base_angle: float | None = None,
    ) -> None:
        """Spawn a small particle burst at ``pos`` (surface coordinates)."""
        color = tuple(tuple(color)[:3])
        for i in range(count):
            if base_angle is None:
                a = math.tau * (i + 0.35 * _rand(i + 1)) / count  # even + jitter
            else:
                a = base_angle + (0.9 * (_rand(i + 7) - 0.5))
            v = speed * (0.55 + 0.9 * _rand(i + 13))
            self.particles.append(Particle(
                x=float(pos[0]), y=float(pos[1]),
                vx=math.cos(a) * v, vy=math.sin(a) * v,
                life=life * (0.7 + 0.6 * _rand(i + 29)),
                max_life=life,
                size=size * (0.7 + 0.6 * _rand(i + 41)),
                color=color,
            ))

    def popup(self, pos, content: str, color) -> None:
        self.texts.append(FloatingText(x=float(pos[0]), y=float(pos[1]),
                                       content=content, color=tuple(tuple(color)[:3])))

    def update(self, dt: float) -> None:
        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            drag = 0.90 ** (dt * 60)
            p.vx *= drag
            p.vy *= drag
            p.life -= dt
        self.particles = [p for p in self.particles if p.life > 0.0]
        for t in self.texts:
            t.y -= 30.0 * dt
            t.life -= dt
        self.texts = [t for t in self.texts if t.life > 0.0]

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        for p in self.particles:
            t = max(0.0, p.life / p.max_life)
            r = max(1, int(p.size * (0.35 + 0.65 * t)))
            faded = blend(p.color, (13, 17, 23), 1.0 - t)  # fade toward bg
            pygame.draw.circle(surface, faded, (int(p.x), int(p.y)), r)
        for t in self.texts:
            surf = font.render(t.content, True, t.color)
            if t.alpha < 200:
                surf.set_alpha(t.alpha)
            surface.blit(surf, surf.get_rect(center=(int(t.x), int(t.y))))
