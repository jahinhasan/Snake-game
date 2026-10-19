"""Procedural audio: tiny synthesized cues, no binary assets, fully optional.

Sounds are square/sine blips generated with NumPy at init time. If the mixer
or audio device is missing (headless CI, broken drivers, Windows without
audio), the manager silently no-ops -- the game never depends on audio.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import AudioConfig

log = logging.getLogger("snake_arena.audio")

SAMPLE_RATE = 22050


class AudioManager:
    def __init__(self, config: AudioConfig) -> None:
        self.enabled = False
        self.config = config
        self.sounds: dict[str, object] = {}
        try:
            import pygame

            pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=512)
            self._pygame = pygame
            self.enabled = True
            self._build_sounds()
            self.apply_volume()
        except Exception as exc:  # noqa: BLE001 - any mixer failure disables audio
            log.info("Audio disabled (%s)", exc)

    def _tone(self, freq: float, seconds: float, volume: float = 0.5, decay: bool = True,
              freq_end: float | None = None) -> np.ndarray:
        t = np.linspace(0.0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
        end = freq_end if freq_end is not None else freq
        sweep = np.linspace(freq, end, t.size)
        wave = np.sin(2.0 * np.pi * sweep * t) * 0.7 + np.sign(np.sin(2.0 * np.pi * sweep * t)) * 0.3
        if decay:
            envelope = np.linspace(1.0, 0.0, t.size) ** 1.5
        else:
            envelope = np.ones_like(t)
        data = (wave * envelope * volume * 32767.0).astype(np.int16)
        return data

    def _build_sounds(self) -> None:
        pygame = self._pygame
        specs: dict[str, np.ndarray] = {
            "click": self._tone(660, 0.05, 0.35),
            "eat": self._tone(520, 0.09, 0.45, freq_end=780),
            "golden": np.concatenate([
                self._tone(523, 0.07, 0.4), self._tone(659, 0.07, 0.4), self._tone(784, 0.11, 0.45),
            ]),
            "special": np.concatenate([
                self._tone(523, 0.06, 0.45), self._tone(784, 0.06, 0.45),
                self._tone(1047, 0.09, 0.5), self._tone(1319, 0.12, 0.5),
            ]),
            "trap": self._tone(220, 0.22, 0.5, freq_end=110),
            "death": self._tone(330, 0.5, 0.55, freq_end=70),
            "pause": self._tone(440, 0.08, 0.3),
            "win": np.concatenate([
                self._tone(523, 0.09, 0.4), self._tone(659, 0.09, 0.4),
                self._tone(784, 0.09, 0.45), self._tone(1047, 0.18, 0.5),
            ]),
            "tick_speed": self._tone(880, 0.03, 0.2),
        }
        for name, data in specs.items():
            try:
                self.sounds[name] = pygame.sndarray.make_sound(data)
            except Exception as exc:  # noqa: BLE001
                log.debug("Could not synthesize sound '%s': %s", name, exc)

    def apply_volume(self) -> None:
        if not self.enabled:
            return
        volume = max(0.0, min(1.0, self.config.volume))
        for sound in self.sounds.values():
            sound.set_volume(volume)

    def play(self, name: str) -> None:
        if not self.enabled or not self.config.enabled:
            return
        sound = self.sounds.get(name)
        if sound is not None:
            try:
                sound.play()
            except Exception:  # noqa: BLE001
                pass

    def shutdown(self) -> None:
        if self.enabled:
            try:
                self._pygame.mixer.quit()
            except Exception:  # noqa: BLE001
                pass
            self.enabled = False
