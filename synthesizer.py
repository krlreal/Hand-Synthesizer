import numpy as np
import pygame
import time

def sine_wave(
    freq: int = 440, 
    duration: float = 1, 
    amplitude: float = 0.5, 
    sample_rate: int = 44100
) -> np.ndarray:
    n = int(duration * sample_rate)
    time_points = np.linspace(0, duration, n, endpoint=False)
    sine = np.sin(2 * np.pi * freq * time_points)
    sine *= amplitude

    wave_int = (sine * 32767).astype(np.int16)

    channels = pygame.mixer.get_init()[2]
    if channels == 2:
        wave_int = np.column_stack([wave_int, wave_int])  # duplicate to stereo

    return wave_int

INTERVALS = {
    "m2": 16/15,
    "maj2": 9/8,
    "m3": 6/5,
    "maj3": 5/4,
    "p4": 4/3,
    "tri": 45/32,
    "p5": 3/2,
    "m6": 8/5,
    "maj6": 5/3,
    "m7": 9/5,
    "maj7": 15/8,
    "oct": 2/1,
}

CHORD_SHAPES = {
    "maj": ["root", "maj3", "p5"],
    "min": ["root", "m3", "p5"],
    "maj7": ["root", "maj3", "p5", "maj7"],
    "m7": ["root", "m3", "p5", "m7"],
    "dom7": ["root", "maj3", "p5", "m7"],
    "dim7": ["root", "m3", "tri", "maj6"],
    "min6": ["root", "m3", "p5", "maj6"]
}

NOTES = {
    "A" : 220,
    "Bb" : 234,
    "B" : 247,
    "C" : 264,
    "Db" : 275,
    "D" : 293,
    "Eb" : 309,
    "E" : 330,
    "F" : 352,
    "Gb" : 366,
    "G" : 396,
    "Ab" : 412,
}

def get_interval(root: str, interval: str) -> int:
    note = int(NOTES[root] * INTERVALS[interval])
    if note > 550:
        note //= 2
    return note

def get_chord(root: str, shape: str) -> list[int]:
    return [
        get_interval(root, interval) if interval != "root" else NOTES[root]
        for interval in CHORD_SHAPES[shape]
    ]

class ChordPlayer:
    def __init__(self) -> None:
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=1)
        pygame.mixer.init()

        self.curr_chord = None
        self.active_sounds = []

    def _build_chord_sound(self, root: str, shape: str, beats: int = 1):
        waves = [
            sine_wave(note, beats) for note in get_chord(root, shape)
        ]
        sounds = []
        for wave_int in waves:
            sound = pygame.mixer.Sound(buffer=wave_int)
            sound.set_volume(1 / np.sqrt(len(waves)))
            sounds.append(sound)
        return sounds

    def play_chord(self, root: str, shape: str, beats: int = 1):
        chord = (root, shape)
        self.stop()
        self.active_sounds = self._build_chord_sound(root, shape, beats)
        for s in self.active_sounds:
            s.play(loops=-1)
        self.curr_chord = chord

    def stop(self):
        for s in self.active_sounds:
            s.fadeout(50)
        self.active_sounds = []
        self.curr_chord = None

def main():
    chord_player = ChordPlayer()
    # chord_player.play_chord("D", "maj")
    # time.sleep(3)

if __name__ == "__main__":
    main()
