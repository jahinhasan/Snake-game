"""Reusable Pygame widgets: buttons, sliders, dropdowns, charts, tables.

Everything is drawn with Pygame primitives (no extra GUI dependency) and is
kept deliberately small: consistent hit-testing, hover/pressed states, and a
shared visual language from theme.py. Dropdown expansion is drawn in a
second pass (see Screen.draw_widgets) so overlays render above other rows.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import pygame

from . import theme
from . import theme
from .theme import COLORS, Fonts, get_font, rgb, text


class Widget:
    """Base widget: rect-based hit testing and the standard lifecycle."""

    def __init__(self, rect: pygame.Rect | None = None) -> None:
        self.rect = rect or pygame.Rect(0, 0, 10, 10)
        self.visible = True
        self.enabled = True
        self.hovered = False

    def contains(self, pos: tuple[int, int]) -> bool:
        return self.visible and self.enabled and self.rect.collidepoint(pos)

    def handle_event(self, event: pygame.event.Event) -> bool:  # noqa: ARG002
        return False

    def update(self, dt: float) -> None:  # noqa: ARG002
        return None

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        raise NotImplementedError


class Button(Widget):
    def __init__(
        self,
        label: str,
        on_click: Callable[[], None],
        rect: pygame.Rect | None = None,
        *,
        style: str = "primary",  # primary | secondary | danger | ghost
        font: pygame.font.Font | None = None,
    ) -> None:
        super().__init__(rect)
        self.label = label
        self.on_click = on_click
        self.style = style
        self.font = font or get_font("ui", 18)
        self.pressed = False
        self._hover = False

    BASE_COLORS = {
        "primary": "#1f6feb",
        "secondary": "#21262d",
        "danger": "#8e2229",
        "ghost": "#161b22",
    }
    TEXT_COLORS = {
        "primary": "#ffffff",
        "secondary": "#e6edf3",
        "danger": "#ffffff",
        "ghost": "#9aa7b3",
    }

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible or not self.enabled:
            return False
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
            return self._hover
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.pressed = True
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was = self.pressed and self.rect.collidepoint(event.pos)
            self.pressed = False
            if was:
                self.on_click()
                return True
        return False

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if not self.visible:
            return
        theme.button_face(
            surface,
            self.rect,
            base=self.BASE_COLORS[self.style],
            hover=self._hover,
            pressed=self.pressed,
        )
        color = self.TEXT_COLORS[self.style] if self.enabled else "#55606b"
        text(surface, self.label, self.rect.center, font=self.font, color=color, align="center")


class Label(Widget):
    def __init__(
        self,
        content: str,
        pos: tuple[int, int],
        *,
        font: pygame.font.Font | None = None,
        color: str = COLORS.text,
        align: str = "topleft",
    ) -> None:
        super().__init__(pygame.Rect(*pos, 0, 0))
        self.content = content
        self.font = font or get_font("ui", 17)
        self.color = color
        self.align = align

    def set(self, content: str, color: str | None = None) -> None:
        self.content = content
        if color:
            self.color = color

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if self.visible:
            text(surface, self.content, self.rect.topleft, font=self.font, color=self.color, align=self.align)


class Toggle(Widget):
    """Checkbox + label row."""

    def __init__(
        self,
        label: str,
        value: bool,
        on_change: Callable[[bool], None] | None = None,
        pos: tuple[int, int] = (0, 0),
        *,
        font: pygame.font.Font | None = None,
        width: int = 240,
    ) -> None:
        super().__init__(pygame.Rect(*pos, width, 26))
        self.label = label
        self.value = value
        self.on_change = on_change
        self.font = font or get_font("ui", 16)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.contains(event.pos):
            self.value = not self.value
            if self.on_change:
                self.on_change(self.value)
            return True
        return False

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if not self.visible:
            return
        box = pygame.Rect(self.rect.x, self.rect.y + 3, 20, 20)
        pygame.draw.rect(surface, rgb(COLORS.panel_alt), box, border_radius=5)
        if self.value:
            pygame.draw.rect(surface, rgb(COLORS.accent), box.inflate(-6, -6), border_radius=3)
        pygame.draw.rect(surface, rgb(COLORS.border_light), box, width=1, border_radius=5)
        text(surface, self.label, (box.right + 8, box.centery), font=self.font, color=COLORS.text, align="midleft")


class Slider(Widget):
    """Horizontal slider over [min_value, max_value] (ints when step>=1)."""

    def __init__(
        self,
        label: str,
        value: float,
        min_value: float,
        max_value: float,
        on_change: Callable[[float], None] | None = None,
        pos: tuple[int, int] = (0, 0),
        *,
        width: int = 240,
        step: float = 1.0,
        fmt: Callable[[float], str] | None = None,
        font: pygame.font.Font | None = None,
    ) -> None:
        super().__init__(pygame.Rect(*pos, width, 40))
        self.label = label
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.on_change = on_change
        self.fmt = fmt or (lambda v: f"{v:g}")
        self.font = font or get_font("ui", 15)
        self._dragging = False

    def _snap(self, v: float) -> float:
        v = max(self.min_value, min(self.max_value, v))
        if self.step >= 1:
            return float(round(v / self.step) * self.step)
        return round(v, 3)

    def _value_from_x(self, x: int) -> float:
        frac = (x - self.rect.x) / max(1, self.rect.w)
        return self._snap(self.min_value + frac * (self.max_value - self.min_value))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.contains(event.pos):
            self._dragging = True
            new = self._value_from_x(event.pos[0])
            if new != self.value:
                self.value = new
                if self.on_change:
                    self.on_change(self.value)
            return True
        if event.type == pygame.MOUSEMOTION and self._dragging:
            new = self._value_from_x(event.pos[0])
            if new != self.value:
                self.value = new
                if self.on_change:
                    self.on_change(self.value)
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was = self._dragging
            self._dragging = False
            return was
        return False

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if not self.visible:
            return
        label = f"{self.label}: {self.fmt(self.value)}"
        text(surface, label, (self.rect.x, self.rect.y), font=self.font, color=COLORS.text_dim)
        track = pygame.Rect(self.rect.x, self.rect.y + 24, self.rect.w, 6)
        pygame.draw.rect(surface, rgb(COLORS.panel_alt), track, border_radius=3)
        frac = (self.value - self.min_value) / max(1e-9, self.max_value - self.min_value)
        handle_x = int(track.x + frac * track.w)
        pygame.draw.rect(surface, rgb(COLORS.accent), (track.x, track.y, handle_x - track.x, track.h), border_radius=3)
        pygame.draw.circle(surface, rgb(COLORS.text), (handle_x, track.centery), 7)
        pygame.draw.circle(surface, rgb(COLORS.accent_dim), (handle_x, track.centery), 7, width=2)


class Dropdown(Widget):
    """Select one of (value, label) options; expansion drawn above others."""

    def __init__(
        self,
        options: Sequence[tuple[object, str]],
        selected: object,
        on_change: Callable[[object], None] | None = None,
        pos: tuple[int, int] = (0, 0),
        *,
        width: int = 220,
        font: pygame.font.Font | None = None,
    ) -> None:
        super().__init__(pygame.Rect(*pos, width, 30))
        self.options = list(options)
        self.selected = selected
        self.on_change = on_change
        self.font = font or get_font("ui", 15)
        self.expanded = False
        self._hover_index: int = -1

    @property
    def selected_label(self) -> str:
        for value, label in self.options:
            if value == self.selected:
                return label
        return str(self.selected)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) and self.enabled:
                self.expanded = not self.expanded
                return True
            if self.expanded:
                for i in range(len(self.options)):
                    if self._option_rect(i).collidepoint(event.pos):
                        value, _label = self.options[i]
                        self.selected = value
                        self.expanded = False
                        if self.on_change:
                            self.on_change(value)
                        return True
                self.expanded = False
                return True
        if event.type == pygame.MOUSEMOTION and self.expanded:
            self._hover_index = -1
            for i in range(len(self.options)):
                if self._option_rect(i).collidepoint(event.pos):
                    self._hover_index = i
            return True
        return False

    def _option_rect(self, i: int) -> pygame.Rect:
        return pygame.Rect(self.rect.x, self.rect.bottom + 2 + i * 26, self.rect.w, 25)

    @property
    def wants_top_pass(self) -> bool:
        return self.expanded

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if not self.visible:
            return
        theme.button_face(surface, self.rect, base="#21262d", hover=False, pressed=False)
        arrow = "v" if self.expanded else "^"
        text(surface, self.selected_label, (self.rect.x + 8, self.rect.centery),
             font=self.font, color=COLORS.text, align="midleft")
        text(surface, arrow, (self.rect.right - 10, self.rect.centery),
             font=self.font, color=COLORS.text_dim, align="midright")

    def draw_top(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if not (self.visible and self.expanded):
            return
        for i, (_value, label) in enumerate(self.options):
            rect = self._option_rect(i)
            fill = COLORS.panel_alt if i == self._hover_index else COLORS.panel
            pygame.draw.rect(surface, rgb(fill), rect, border_radius=6)
            pygame.draw.rect(surface, rgb(COLORS.border), rect, width=1, border_radius=6)
            color = COLORS.accent if _value == self.selected else COLORS.text
            text(surface, label, (rect.x + 8, rect.centery), font=self.font, color=color, align="midleft")


class TextInput(Widget):
    """Single-line editable text field (click to focus, Esc to blur)."""

    def __init__(
        self,
        value: str,
        on_change: Callable[[str], None] | None = None,
        pos: tuple[int, int] = (0, 0),
        *,
        width: int = 240,
        font: pygame.font.Font | None = None,
        placeholder: str = "",
    ) -> None:
        super().__init__(pygame.Rect(*pos, width, 30))
        self.value = value
        self.on_change = on_change
        self.font = font or get_font("mono", 15)
        self.placeholder = placeholder
        self.focused = False
        self._blink = 0.0

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.focused = self.rect.collidepoint(event.pos)
            return self.focused
        if not self.focused:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.focused = False
                return True
            if event.key == pygame.K_BACKSPACE:
                self.value = self.value[:-1]
                if self.on_change:
                    self.on_change(self.value)
                return True
            if event.key == pygame.K_RETURN:
                self.focused = False
                return True
            if event.unicode and event.unicode.isprintable():
                self.value += event.unicode
                if self.on_change:
                    self.on_change(self.value)
                return True
        return False

    def update(self, dt: float) -> None:
        self._blink += dt

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if not self.visible:
            return
        fill = "#0d1117" if not self.focused else "#101822"
        pygame.draw.rect(surface, rgb(fill), self.rect, border_radius=7)
        pygame.draw.rect(
            surface,
            rgb(COLORS.accent if self.focused else COLORS.border_light),
            self.rect,
            width=2 if self.focused else 1,
            border_radius=7,
        )
        shown = self.value
        if self.font.size(shown)[0] > self.rect.w - 14:
            while shown and self.font.size(shown + "...")[0] > self.rect.w - 14:
                shown = shown[1:]
            shown = "..." + shown
        color = COLORS.text if self.value else COLORS.text_faint
        label = shown or self.placeholder
        text(surface, label, (self.rect.x + 8, self.rect.centery), font=self.font, color=color, align="midleft")
        if self.focused and int(self._blink * 2) % 2 == 0:
            cursor_x = self.rect.x + 8 + self.font.size(shown)[0] + 1
            pygame.draw.line(surface, rgb(COLORS.accent), (cursor_x, self.rect.y + 6), (cursor_x, self.rect.bottom - 6), 2)


class ProgressBar(Widget):
    def __init__(self, pos: tuple[int, int], width: int = 300, height: int = 14) -> None:
        super().__init__(pygame.Rect(*pos, width, height))
        self.fraction = 0.0

    def set(self, fraction: float) -> None:
        self.fraction = max(0.0, min(1.0, fraction))

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if not self.visible:
            return
        pygame.draw.rect(surface, rgb(COLORS.panel_alt), self.rect, border_radius=7)
        inner = self.rect.inflate(-4, -4)
        inner.width = int(inner.w * self.fraction)
        if inner.w > 2:
            pygame.draw.rect(surface, rgb(COLORS.accent), inner, border_radius=5)


class Chart(Widget):
    """Line chart for live dashboard curves (drawn from recorded data)."""

    def __init__(self, title: str, pos: tuple[int, int], size: tuple[int, int]) -> None:
        super().__init__(pygame.Rect(*pos, *size))
        self.title = title
        self.series: list[tuple[str, list[float], str]] = []
        self.y_label = ""

    def set_series(self, *series: tuple[str, list[float], str]) -> None:
        """Each series: (label, values, color)."""
        self.series = list(series)

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if not self.visible:
            return
        theme.panel(surface, self.rect, fill=COLORS.bg_alt)
        text(surface, self.title, (self.rect.x + 10, self.rect.y + 7), font=get_font("ui", 14), color=COLORS.text_dim)
        area = self.rect.inflate(-24, -44)
        area.y += 8
        if area.w < 20 or area.h < 10:
            return
        all_values = [v for _label, values, _c in self.series for v in values]
        pygame.draw.rect(surface, rgb("#0a0e14"), area)
        lo = min(all_values) if all_values else 0.0
        hi = max(all_values) if all_values else 1.0
        if hi - lo < 1e-9:
            hi = lo + 1.0
        for gy in range(5):
            y = area.y + gy * area.h / 4
            pygame.draw.line(surface, rgb("#1b2230"), (area.x, y), (area.right, y))
        for label, values, color in self.series:
            if len(values) < 2:
                continue
            n = len(values)
            points = []
            for i, v in enumerate(values):
                px = area.x + (i / (n - 1)) * area.w
                py = area.bottom - ((v - lo) / (hi - lo)) * area.h
                points.append((px, py))
            pygame.draw.lines(surface, rgb(color), False, points, 2)
        lo_s = f"{lo:.2f}" if abs(lo) < 1000 else f"{lo:.0f}"
        hi_s = f"{hi:.2f}" if abs(hi) < 1000 else f"{hi:.0f}"
        text(surface, hi_s, (area.right + 4, area.y), font=get_font("mono", 11), color=COLORS.text_faint)
        text(surface, lo_s, (area.right + 4, area.bottom), font=get_font("mono", 11), color=COLORS.text_faint, align="topright")
        # legend
        lx = area.x + 4
        for label, _values, color in self.series:
            pygame.draw.rect(surface, rgb(color), (lx, area.bottom - 12, 10, 4), border_radius=2)
            lx = text(surface, label, (lx + 14, area.bottom - 10), font=get_font("ui", 11), color=COLORS.text_dim).right + 10


class BarChart(Widget):
    """Categorical bar chart (benchmark mean scores, etc.)."""

    def __init__(self, title: str, pos: tuple[int, int], size: tuple[int, int]) -> None:
        super().__init__(pygame.Rect(*pos, *size))
        self.title = title
        self.bars: list[tuple[str, float, str]] = []

    def set_bars(self, bars: Sequence[tuple[str, float, str]]) -> None:
        self.bars = list(bars)

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if not self.visible:
            return
        theme.panel(surface, self.rect, fill=COLORS.bg_alt)
        text(surface, self.title, (self.rect.x + 10, self.rect.y + 7), font=get_font("ui", 14), color=COLORS.text_dim)
        area = self.rect.inflate(-24, -50)
        area.y += 8
        if not self.bars or area.w < 20:
            return
        hi = max((v for _l, v, _c in self.bars), default=1.0)
        if hi <= 0:
            hi = 1.0
        n = len(self.bars)
        slot = area.w / n
        bar_w = max(8, int(slot * 0.55))
        font_small = get_font("ui", 12)
        for i, (label, value, color) in enumerate(self.bars):
            cx = area.x + slot * i + slot / 2
            h = int((value / hi) * (area.h - 18))
            rect = pygame.Rect(int(cx - bar_w / 2), area.bottom - h, bar_w, h)
            pygame.draw.rect(surface, rgb(color), rect, border_radius=4)
            text(surface, f"{value:.1f}" if value < 10000 else f"{value:.0f}",
                 (cx, rect.y - 8), font=font_small, color=COLORS.text, align="midbottom")
            text(surface, label, (cx, area.bottom + 4), font=font_small, color=COLORS.text_dim, align="midtop")


class DataTable(Widget):
    """Scrollable text table."""

    def __init__(self, headers: list[str], pos: tuple[int, int], size: tuple[int, int], *,
                 col_ratios: list[float] | None = None) -> None:
        super().__init__(pygame.Rect(*pos, *size))
        self.headers = headers
        self.rows: list[list[str]] = []
        self.col_ratios = col_ratios or [1.0 / max(1, len(headers))] * len(headers)
        self.scroll = 0
        self.font = get_font("mono", 13)
        self.header_font = get_font("ui", 13)

    def set_rows(self, rows: list[list[str]]) -> None:
        self.rows = rows
        self.scroll = max(0, min(self.scroll, max(0, len(rows) - 1)))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible or not self.rect.collidepoint(getattr(event, "pos", (-1, -1))):
            return False
        if event.type == pygame.MOUSEWHEEL:
            self.scroll = max(0, min(len(self.rows) - 1, self.scroll + (1 if event.y < 0 else -1) * 3))
            return True
        return False

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if not self.visible:
            return
        theme.panel(surface, self.rect, fill=COLORS.bg_alt)
        inner = self.rect.inflate(-16, -12)
        x_positions = []
        x = inner.x
        for ratio in self.col_ratios:
            x_positions.append(x)
            x += int(inner.w * ratio)
        hx = {name: i for i, name in enumerate(self.headers)}
        for i, header in enumerate(self.headers):
            text(surface, header, (x_positions[i], inner.y), font=self.header_font, color=COLORS.text_dim)
        theme.hline(surface, inner.y + 18, inner.x, inner.right)
        row_h = 18
        visible_rows = (inner.h - 22) // row_h
        for r, row in enumerate(self.rows[self.scroll : self.scroll + visible_rows]):
            y = inner.y + 24 + r * row_h
            for i, cell in enumerate(row[: len(self.headers)]):
                text(surface, str(cell), (x_positions[i], y), font=self.font,
                     color=COLORS.text if r % 2 == 0 else COLORS.text_dim)


def draw_widgets(surface: pygame.Surface, widgets: list[Widget], fonts: Fonts, dt: float = 0.0) -> None:
    """Standard draw pass; dropdown expansions render after everything else."""
    for widget in widgets:
        widget.update(dt)
        widget.draw(surface, fonts)
    for widget in widgets:
        if isinstance(widget, Dropdown):
            widget.draw_top(surface, fonts)


def send_event(widgets: list[Widget], event: pygame.event.Event) -> bool:
    """Deliver an event; later widgets (drawn on top) get it first."""
    for widget in reversed(widgets):
        if widget.visible and widget.enabled and widget.handle_event(event):
            return True
    return False
