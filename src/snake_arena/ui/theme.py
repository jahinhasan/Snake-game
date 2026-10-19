"""Visual theme: dark professional palette, typography, drawing helpers."""

from __future__ import annotations

from typing import NamedTuple

import pygame


class Palette(NamedTuple):
    bg: str = "#0d1117"
    bg_alt: str = "#10161f"
    panel: str = "#161b22"
    panel_alt: str = "#1c2330"
    border: str = "#30363d"
    border_light: str = "#3d444d"
    text: str = "#e6edf3"
    text_dim: str = "#9aa7b3"
    text_faint: str = "#64707d"
    accent: str = "#58a6ff"
    accent_dim: str = "#2f6cb3"
    green: str = "#7ee787"
    blue: str = "#79c0ff"
    gold: str = "#e3b341"
    purple: str = "#bc8cff"
    red: str = "#f85149"
    orange: str = "#ffa657"
    human: str = "#f0f6fc"
    ai: str = "#58a6ff"
    danger: str = "#f85149"
    success: str = "#56d364"


COLORS = Palette()


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def rgb(value: str) -> tuple[int, int, int]:
    return hex_to_rgb(value)


def with_alpha(value: str, alpha: float) -> tuple[int, int, int, int]:
    r, g, b = hex_to_rgb(value)
    return r, g, b, max(0, min(255, int(alpha * 255)))


# Font families with broad cross-platform coverage (Linux / Windows / macOS).
_UI_FAMILIES = "dejavusans,segoeui,ubuntu,notosans,helvetica,arial"
_MONO_FAMILIES = "dejavusansmono,consolas,menlo,cascadiacode,monospace"

_fonts: dict[tuple[str, int], pygame.font.Font] = {}
_cache: dict[tuple[str, int, tuple[int, int, int]], pygame.Surface] = {}


class Fonts:
    """Resolved font handles at standard sizes; built once per app run."""

    def __init__(self) -> None:
        self.title = get_font("ui", 46)
        self.h1 = get_font("ui", 30)
        self.h2 = get_font("ui", 22)
        self.h3 = get_font("ui", 19)
        self.body = get_font("ui", 17)
        self.small = get_font("ui", 14)
        self.tiny = get_font("ui", 12)
        self.mono = get_font("mono", 16)
        self.mono_small = get_font("mono", 13)
        self.mono_large = get_font("mono", 22)


def get_font(kind: str, size: int) -> pygame.font.Font:
    key = (kind, size)
    font = _fonts.get(key)
    if font is None:
        families = _UI_FAMILIES if kind == "ui" else _MONO_FAMILIES
        try:
            font = pygame.font.SysFont(families, size)
            # SysFont falls back silently to the default font; that's fine.
        except Exception:  # noqa: BLE001 - missing fonts must never crash
            font = pygame.font.Font(None, size)
        _fonts[key] = font
    return font


def text(
    surface: pygame.Surface,
    content: str,
    pos: tuple[int, int],
    *,
    font: pygame.font.Font,
    color: tuple[int, int, int] | str = COLORS.text,
    align: str = "topleft",
    shadow: bool = False,
) -> pygame.Rect:
    """Draw cached text; returns its rect. ``align`` names the anchor."""
    if isinstance(color, str):
        color = rgb(color)
    key = (content, id(font), color)
    rendered = _cache.get(key)
    if rendered is None:
        rendered = font.render(content, True, color)
        if len(_cache) > 3000:
            _cache.clear()
        _cache[key] = rendered
    rect = rendered.get_rect(**{align: pos})
    if shadow:
        sh = font.render(content, True, (0, 0, 0))
        surface.blit(sh, rect.move(1, 1))
    surface.blit(rendered, rect)
    return rect


def wrap_text(content: str, font: pygame.font.Font, max_width: int) -> list[str]:
    """Greedy word wrap for paragraph rendering."""
    words = content.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.size(candidate)[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    *,
    fill: str = COLORS.panel,
    border: str | None = COLORS.border,
    radius: int = 10,
    border_width: int = 1,
) -> pygame.Rect:
    pygame.draw.rect(surface, rgb(fill), rect, border_radius=radius)
    if border:
        pygame.draw.rect(surface, rgb(border), rect, width=border_width, border_radius=radius)
    return rect


def button_face(
    surface: pygame.Surface,
    rect: pygame.Rect,
    *,
    base: str,
    hover: bool,
    pressed: bool,
    radius: int = 9,
    border: str = COLORS.border_light,
) -> None:
    color = rgb(base)
    if hover:
        color = tuple(min(255, c + 26) for c in color)
    if pressed:
        color = tuple(max(0, c - 22) for c in color)
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    pygame.draw.rect(surface, rgb(border), rect, width=1, border_radius=radius)


def hline(surface: pygame.Surface, y: int, x0: int, x1: int, color: str = COLORS.border) -> None:
    pygame.draw.line(surface, rgb(color), (x0, y), (x1, y))


def chip(
    surface: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    *,
    font: pygame.font.Font,
    fg: str = COLORS.text,
    bg: str = COLORS.panel_alt,
) -> pygame.Rect:
    """Small rounded status pill; returns the text rect."""
    panel(surface, rect, fill=bg, border=COLORS.border, radius=999)
    return text(surface, label, rect.center, font=font, color=fg, align="center")


def fade_surface(size: tuple[int, int], alpha: int) -> pygame.Surface:
    surf = pygame.Surface(size, pygame.SRCALPHA)
    surf.fill((5, 8, 12, alpha))
    return surf
