#!/usr/bin/env python3
"""
Battle City (Tank 1990) — Python/Pygame implementation.
Modules: CSP Map Generator | BFS | Greedy | A* | Minimax+AlphaBeta
"""

import pygame
import sys
import random
import math
from collections import deque
import heapq
import array

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
TILE_SIZE = 24
GRID_W    = 26
GRID_H    = 26
HUD_WIDTH = 220
SCREEN_W  = TILE_SIZE * GRID_W + HUD_WIDTH
SCREEN_H  = TILE_SIZE * GRID_H + 50   # 50 px top bar + grid + bottom strip
FPS       = 60

# Terrain IDs
EMPTY  = 0
BRICK  = 1
STEEL  = 2
WATER  = 3
FOREST = 4
EAGLE  = 5

# Directions
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIR_VEC = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}

# Colours
BLACK      = (0,   0,   0)
WHITE      = (255, 255, 255)
GRAY       = (128, 128, 128)
DARK       = (12,  14,  18)
RED        = (220, 45,  45)
GREEN      = (50,  200, 60)
LIME       = (110, 230, 50)
BLUE       = (40,  90,  220)
YELLOW     = (240, 210, 40)
ORANGE     = (230, 120, 25)
CYAN       = (40,  200, 220)
PURPLE     = (160, 50,  220)
BROWN      = (140, 85,  38)
SILVER     = (185, 195, 210)
GOLD       = (255, 210, 0)
DARK_GREEN = (18,  90,  20)
PINK       = (240, 80,  160)
LIGHT_GRAY = (195, 200, 210)
SAND       = (210, 185, 120)

# Game states
ST_MENU     = 'menu'
ST_PLAYING  = 'playing'
ST_LVLDONE  = 'level_complete'
ST_GAMEOVER = 'game_over'
ST_WIN      = 'win'
ST_BOSSDEAD = 'boss_dead'

# Movement speed: frames between grid steps (60 FPS base)
# Higher = slower.  12 frames → 5 moves/sec feels like classic Tank 1990.
PLAYER_MOVE_INTERVAL = 10   # ~6 moves/sec
BASIC_MOVE_INTERVAL  = 20   # ~3 moves/sec  (slow)
FAST_MOVE_INTERVAL   = 10   # ~6 moves/sec  (fast, 2× basic)
ARMOR_MOVE_INTERVAL  = 15   # ~4 moves/sec  (medium)
POWER_MOVE_INTERVAL  = 12
BOSS_MOVE_INTERVAL   = {1: 20, 2: 14, 3: 9}   # gets faster per phase

BULLET_SPEED = 5   # pixels per frame


# ═══════════════════════════════════════════════════════════════
# PROCEDURAL SOUND ENGINE
# ═══════════════════════════════════════════════════════════════
class SoundEngine:
    """100 % procedural audio — no .wav files needed."""
    RATE  = 22050
    BITS  = -16
    CHANS = 1
    CHUNK = 512

    def __init__(self):
        self.muted   = False
        self.enabled = False
        try:
            pygame.mixer.pre_init(self.RATE, self.BITS, self.CHANS, self.CHUNK)
            pygame.mixer.init()
            self.enabled = True
        except Exception:
            return
        self._cache: dict = {}
        self._build_all()
        self._ch_engine = pygame.mixer.Channel(0)
        self._ch_music  = pygame.mixer.Channel(1)

    def play(self, name: str, volume: float = 1.0):
        if not self.enabled or self.muted:
            return
        snd = self._cache.get(name)
        if snd is None:
            return
        snd.set_volume(max(0.0, min(1.0, volume)))
        ch = pygame.mixer.find_channel()
        if ch:
            try:
                ch.play(snd)
            except Exception:
                pass

    def play_engine(self, volume: float):
        if not self.enabled or self.muted:
            self._ch_engine.stop()
            return
        snd = self._cache.get('engine_tick')
        if snd is None:
            return
        if not self._ch_engine.get_busy():
            snd.set_volume(volume)
            self._ch_engine.play(snd, loops=-1)
        else:
            self._ch_engine.set_volume(volume)

    def stop_engine(self):
        if self.enabled:
            self._ch_engine.stop()

    def play_boss_ambience(self, phase: int):
        if not self.enabled or self.muted:
            self._ch_music.stop()
            return
        snd = self._cache.get(f'boss_amb_{phase}')
        if snd and not self._ch_music.get_busy():
            snd.set_volume(0.18)
            self._ch_music.play(snd, loops=-1)

    def stop_boss_ambience(self):
        if self.enabled:
            self._ch_music.stop()

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            pygame.mixer.pause()
        else:
            pygame.mixer.unpause()
        return self.muted

    # ── PCM helpers ───────────────────────────────────────────
    def _make_sound(self, samples):
        buf = array.array('h', (max(-32767, min(32767, int(s * 32767)))
                                for s in samples))
        return pygame.mixer.Sound(buffer=buf)

    def _silence(self, dur):
        return [0.0] * int(self.RATE * dur)

    def _sine(self, freq, dur, amp=1.0):
        n = int(self.RATE * dur)
        return [amp * math.sin(2 * math.pi * freq * i / self.RATE) for i in range(n)]

    def _noise(self, dur, amp=1.0):
        return [amp * random.uniform(-1, 1) for _ in range(int(self.RATE * dur))]

    def _envelope(self, samples, attack, decay, sustain, release):
        n = len(samples)
        a = int(attack  * self.RATE)
        d = int(decay   * self.RATE)
        r = int(release * self.RATE)
        s_len = max(0, n - a - d - r)
        out = []
        for i, v in enumerate(samples):
            if   i < a:           env = i / max(1, a)
            elif i < a + d:       env = 1.0 - (1.0 - sustain) * (i - a) / max(1, d)
            elif i < a + d + s_len: env = sustain
            else:
                ri  = i - (a + d + s_len)
                env = sustain * (1.0 - ri / max(1, r))
            out.append(v * max(0.0, env))
        return out

    def _mix(self, *tracks):
        maxlen = max(len(t) for t in tracks)
        out = [0.0] * maxlen
        for t in tracks:
            for i, v in enumerate(t):
                out[i] += v
        peak = max(abs(x) for x in out) or 1.0
        if peak > 1.0:
            out = [x / peak for x in out]
        return out

    def _fade_out(self, samples, fade_dur):
        n   = len(samples)
        fn  = int(fade_dur * self.RATE)
        start = max(0, n - fn)
        out = list(samples)
        for i in range(start, n):
            out[i] *= 1.0 - (i - start) / max(1, fn)
        return out

    def _build_all(self):
        b = self._cache
        # shoot
        raw = self._mix(self._sine(880, 0.04, 0.6) + self._sine(440, 0.06, 0.2),
                        self._noise(0.03, 0.9) + self._silence(0.06))
        raw = self._envelope(raw, 0.001, 0.02, 0.1, 0.05)
        b['shoot'] = self._make_sound(self._fade_out(raw, 0.03))
        # boss fire
        raw2 = self._mix(self._sine(440, 0.06, 0.7) + self._silence(0.04),
                         self._sine(110, 0.10, 0.5),
                         self._noise(0.04, 0.8) + self._silence(0.06))
        raw2 = self._envelope(raw2, 0.001, 0.03, 0.15, 0.07)
        b['boss_fire'] = self._make_sound(self._fade_out(raw2, 0.04))
        # hit player
        raw3 = self._mix(self._sine(80, 0.12, 0.9),
                         self._noise(0.08, 0.4) + self._silence(0.04))
        raw3 = self._envelope(raw3, 0.001, 0.04, 0.3, 0.08)
        b['hit_player'] = self._make_sound(self._fade_out(raw3, 0.05))
        # hit steel
        raw4 = self._mix(self._sine(1800, 0.08, 0.5),
                         self._sine(1200, 0.06, 0.3) + self._silence(0.02),
                         self._noise(0.01, 0.7) + self._silence(0.07))
        raw4 = self._envelope(raw4, 0.0005, 0.01, 0.2, 0.07)
        b['hit_steel'] = self._make_sound(self._fade_out(raw4, 0.04))
        # explode small
        raw5 = self._mix(self._noise(0.18, 1.0),
                         self._sine(160, 0.12, 0.4) + self._silence(0.06))
        raw5 = self._envelope(raw5, 0.001, 0.05, 0.2, 0.12)
        b['explode_sm'] = self._make_sound(self._fade_out(raw5, 0.07))
        # explode large
        raw6 = self._mix(self._sine(55, 0.35, 0.9),
                         self._sine(110, 0.25, 0.6) + self._silence(0.1),
                         self._noise(0.30, 0.8))
        raw6 = self._envelope(raw6, 0.001, 0.08, 0.4, 0.22)
        b['explode_lg'] = self._make_sound(self._fade_out(raw6, 0.12))
        # spawn sweep
        sweep = []
        n = int(self.RATE * 0.22)
        for i in range(n):
            f = 300 + 1400 * (i / n) ** 2
            sweep.append(0.5 * math.sin(2 * math.pi * f * i / self.RATE))
        sweep = self._envelope(sweep, 0.01, 0.05, 0.4, 0.10)
        b['spawn'] = self._make_sound(self._fade_out(sweep, 0.05))
        # level win
        parts = []
        for freq in [523, 659, 784, 1047]:
            t = self._sine(freq, 0.14, 0.7)
            t = self._envelope(t, 0.01, 0.04, 0.5, 0.08)
            parts.extend(t)
            parts.extend(self._silence(0.02))
        b['level_win'] = self._make_sound(parts)
        # game over
        parts2 = []
        for freq in [400, 300, 200]:
            t = self._mix(self._sine(freq, 0.22, 0.8),
                          self._sine(freq * 1.5, 0.22, 0.25))
            t = self._envelope(t, 0.005, 0.06, 0.5, 0.14)
            parts2.extend(t)
            parts2.extend(self._silence(0.06))
        b['game_over'] = self._make_sound(parts2)
        # engine tick
        blip = self._mix(self._sine(90, 0.04, 0.18), self._sine(75, 0.04, 0.12))
        blip = self._envelope(blip, 0.005, 0.01, 0.1, 0.02)
        blip += self._silence(0.22)
        b['engine_tick'] = self._make_sound(blip)
        # bullet cancel
        pop = self._noise(0.05, 0.6)
        pop = self._envelope(pop, 0.001, 0.01, 0.2, 0.04)
        b['bullet_cancel'] = self._make_sound(pop)
        # boss ambience phases
        for ph, (f1, f2) in enumerate([(55, 58), (65, 69), (80, 120)], 1):
            d1   = self._sine(f1, 1.0, 0.3)
            d2   = self._sine(f2, 1.0, 0.2)
            nz   = self._noise(1.0, 0.04)
            raw_a = self._mix(d1, d2, nz)
            raw_a = self._envelope(raw_a, 0.1, 0.1, 0.8, 0.1)
            b[f'boss_amb_{ph}'] = self._make_sound(raw_a)


# ═══════════════════════════════════════════════════════════════
# SCREEN SHAKE
# ═══════════════════════════════════════════════════════════════
class ScreenShake:
    def __init__(self):
        self.trauma = 0.0
        self.ox = self.oy = 0

    def add(self, amount):
        self.trauma = min(1.0, self.trauma + amount)

    def update(self):
        if self.trauma > 0:
            self.trauma = max(0.0, self.trauma - 0.04)
            s = self.trauma ** 2
            self.ox = int(random.uniform(-1, 1) * s * 10)
            self.oy = int(random.uniform(-1, 1) * s * 10)
        else:
            self.ox = self.oy = 0

    def offset(self):
        return self.ox, self.oy


# ═══════════════════════════════════════════════════════════════
# FLOATING SCORE POPUP
# ═══════════════════════════════════════════════════════════════
class ScorePopup:
    def __init__(self, gx, gy, value, color=YELLOW):
        self.px    = gx * TILE_SIZE + TILE_SIZE // 2
        self.py    = float(gy * TILE_SIZE)
        self.value = value
        self.color = color
        self.life  = 80
        self.max_life = 80

    def update(self):
        self.py  -= 0.6
        self.life -= 1

    def draw(self, surface, ox, oy, font):
        if self.life <= 0:
            return
        s = font.render(f"+{self.value}", True, self.color)
        s.set_alpha(int(255 * self.life / self.max_life))
        surface.blit(s, (int(self.px) + ox - s.get_width() // 2,
                         int(self.py) + oy))


# ═══════════════════════════════════════════════════════════════
# EXPLOSION
# ═══════════════════════════════════════════════════════════════
class Explosion:
    def __init__(self, gx, gy, size=1):
        self.px  = gx * TILE_SIZE + TILE_SIZE // 2
        self.py  = gy * TILE_SIZE + TILE_SIZE // 2
        self.size = size
        self.life = 0
        self.max_life = 22 + size * 8
        self.rings = []
        for i in range(3 + size):
            self.rings.append({
                'delay':  i * 4,
                'radius': (6 + i * 8) * size,
                'color':  [YELLOW, ORANGE, RED, (80, 20, 0)][min(i, 3)]
            })

    def update(self):
        self.life += 1

    @property
    def alive(self):
        return self.life < self.max_life

    def draw(self, surface, ox, oy):
        cx = int(self.px) + ox
        cy = int(self.py) + oy
        t  = self.life
        if t < 8:
            r = int((1 - t / 8) * 18 * self.size)
            pygame.draw.circle(surface, WHITE if t < 3 else YELLOW, (cx, cy), r)
        for ring in self.rings:
            if t < ring['delay']:
                continue
            rt   = t - ring['delay']
            max_rt = self.max_life - ring['delay']
            if rt > max_rt:
                continue
            rp     = rt / max_rt
            radius = int(ring['radius'] * rp)
            alpha_f = max(0.0, 1 - rp * 1.5)
            if radius > 0 and alpha_f > 0:
                thick = max(1, int(3 * (1 - rp)))
                try:
                    pygame.draw.circle(surface, ring['color'], (cx, cy), radius, thick)
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════
# PARTICLE
# ═══════════════════════════════════════════════════════════════
class Particle:
    def __init__(self, gx, gy, color, size=4, speed=3):
        self.x = gx * TILE_SIZE + TILE_SIZE // 2
        self.y = gy * TILE_SIZE + TILE_SIZE // 2
        angle  = random.uniform(0, math.pi * 2)
        spd    = random.uniform(speed * 0.4, speed)
        self.vx    = math.cos(angle) * spd
        self.vy    = math.sin(angle) * spd - random.uniform(0, speed * 0.5)
        self.color = color
        self.life  = random.randint(18, 38)
        self.mlife = self.life
        self.size  = random.randint(2, max(2, size))

    def update(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vy += 0.3
        self.vx *= 0.96
        self.life -= 1

    def draw(self, surface, ox, oy):
        if self.life <= 0:
            return
        s = max(1, int(self.size * self.life / self.mlife))
        pygame.draw.circle(surface, self.color,
                           (int(self.x) + ox, int(self.y) + oy), s)


# ═══════════════════════════════════════════════════════════════
# TRAIL PARTICLE
# ═══════════════════════════════════════════════════════════════
class TrailParticle:
    def __init__(self, px, py, color):
        self.x, self.y = px, py
        self.color = color
        self.life  = 8
        self.mlife = 8

    def update(self):
        self.life -= 1

    def draw(self, surface, ox, oy):
        if self.life <= 0:
            return
        a = self.life / self.mlife
        s = max(1, int(3 * a))
        pygame.draw.circle(surface, self.color,
                           (int(self.x) + ox, int(self.y) + oy), s)


# ═══════════════════════════════════════════════════════════════
# SPAWN ANIMATION
# ═══════════════════════════════════════════════════════════════
class SpawnAnim:
    def __init__(self, x, y):
        self.gx, self.gy = x, y
        self.life = 45
        self.max_life = 45

    @property
    def alive(self):
        return self.life > 0

    def update(self):
        self.life -= 1

    def draw(self, surface, ox, oy):
        if not self.alive:
            return
        t  = 1 - self.life / self.max_life
        rx = ox + self.gx * TILE_SIZE
        ry = oy + self.gy * TILE_SIZE
        ts = TILE_SIZE
        cx, cy = rx + ts // 2, ry + ts // 2
        scale  = 0.5 + 0.5 * math.sin(t * math.pi * 4)
        size   = int(ts // 2 * scale)
        for angle in range(0, 360, 45):
            rad = math.radians(angle + t * 360)
            ex  = cx + int(math.cos(rad) * size)
            ey  = cy + int(math.sin(rad) * size)
            pygame.draw.line(surface, YELLOW, (cx, cy), (ex, ey), 2)
        pygame.draw.circle(surface, WHITE, (cx, cy), max(2, size // 3))


# ═══════════════════════════════════════════════════════════════
# CSP MAP GENERATOR
# Tile-by-tile backtracking with forward checking.
# ═══════════════════════════════════════════════════════════════
class MapGenerator:
    EAGLE_POS    = (12, 24)
    PLAYER_START = (4,  24)
    SPAWN_POINTS = [(0, 0), (12, 0), (24, 0)]

    # Terrain probability weights per level
    LEVEL_CFG = {
        1: dict(brick=0.26, steel=0.03, water=0.03, forest=0.09),
        2: dict(brick=0.16, steel=0.14, water=0.04, forest=0.06),
        3: dict(brick=0.10, steel=0.08, water=0.03, forest=0.04),
    }

    def __init__(self, level=1):
        self.level = level
        self.cfg   = self.LEVEL_CFG.get(level, self.LEVEL_CFG[1])

    # ── Public entry point ────────────────────────────────────
    def generate(self):
        """
        True CSP backtracking generator.
        Variables  : each grid tile (i,j)
        Domain     : {EMPTY, BRICK, STEEL, WATER, FOREST}  (EAGLE placed first)
        Constraints:
          C1 Base Safety   — Eagle surrounded by ≥1 ring of BRICK/STEEL
          C2 Reachability  — BFS path from every spawn → Eagle must exist
          C3 Fairness      — no spawn within 10 tiles of player start
          C4 Density       — ≤40 % wall tiles (BRICK+STEEL+WATER)
          C5 Water         — water may not be placed on the only path to Eagle
        Forward checking: after each tile assignment, immediately verify local
        reachability cannot be permanently broken.
        """
        grid = self._init_fixed()
        tiles = self._get_variable_tiles(grid)
        result = self._backtrack(grid, tiles, 0)
        if result is not None:
            return result
        return self._fallback()

    # ── Fixed tile placement ──────────────────────────────────
    def _init_fixed(self):
        grid = [[EMPTY] * GRID_W for _ in range(GRID_H)]
        ex, ey = self.EAGLE_POS
        grid[ey][ex] = EAGLE
        # C1: surround Eagle with BRICK (inner ring)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = ex + dx, ey + dy
                if (dx, dy) != (0, 0) and 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    grid[ny][nx] = BRICK
        # Clear spawn zones (C3 fairness pre-condition)
        for sx, sy in self.SPAWN_POINTS:
            for dy in range(3):
                for dx in range(-1, 2):
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                        grid[ny][nx] = EMPTY
        # Clear player start area
        px, py = self.PLAYER_START
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = px + dx, py + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    grid[ny][nx] = EMPTY
        return grid

    def _get_variable_tiles(self, grid):
        """Return list of (x, y) tiles that can be assigned terrain."""
        tiles = []
        for y in range(GRID_H):
            for x in range(GRID_W):
                if grid[y][x] == EMPTY and not self._protected(x, y):
                    tiles.append((x, y))
        return tiles

    def _protected(self, x, y):
        """True if tile must stay EMPTY (spawn zone or player area)."""
        for sx, sy in self.SPAWN_POINTS:
            if abs(x - sx) + abs(y - sy) < 4:
                return True
        px, py = self.PLAYER_START
        if abs(x - px) + abs(y - py) < 4:
            return True
        if y >= GRID_H - 1:   # bottom row always clear
            return True
        return False

    # ── Backtracking CSP with forward checking ────────────────
    def _backtrack(self, grid, tiles, idx):
        """
        Recursively assign terrain to each variable tile.
        Forward check: after each assignment, quickly verify that
        at least one spawn still has a reachable path to Eagle.
        """
        if idx == len(tiles):
            # All tiles assigned — run full constraint validation
            if self._validate(grid):
                return [row[:] for row in grid]
            return None

        x, y = tiles[idx]
        domain = self._ordered_domain(x, y)

        for terrain in domain:
            grid[y][x] = terrain
            # Forward check: only do expensive BFS every 50 tiles
            # to keep generation fast while still catching dead ends.
            if idx % 50 == 0 and not self._quick_reachable(grid):
                grid[y][x] = EMPTY
                continue
            result = self._backtrack(grid, tiles, idx + 1)
            if result is not None:
                return result
            # Backtrack: reset tile
            grid[y][x] = EMPTY
        return None

    def _ordered_domain(self, x, y):
        """
        Return domain values ordered by probability (most-likely first)
        so backtracking rarely needs to backtrack deeply.
        """
        cfg = self.cfg
        r   = random.random()
        cum_brick  = cfg['brick']
        cum_steel  = cum_brick + cfg['steel']
        cum_water  = cum_steel + cfg['water']
        cum_forest = cum_water + cfg['forest']

        if   r < cum_brick:  primary = BRICK
        elif r < cum_steel:  primary = STEEL
        elif r < cum_water:  primary = WATER
        elif r < cum_forest: primary = FOREST
        else:                primary = EMPTY

        # Domain order: most probable first, EMPTY always last fallback
        order = [primary]
        for t in [EMPTY, BRICK, FOREST, STEEL, WATER]:
            if t not in order:
                order.append(t)
        return order

    def _quick_reachable(self, grid):
        """Fast check: can at least one spawn still reach Eagle?"""
        eagle = self.EAGLE_POS
        for sx, sy in self.SPAWN_POINTS:
            if self._bfs_reach(grid, (sx, sy), eagle):
                return True
        return False

    def _validate(self, grid):
        """Full constraint validation (C2, C4, C5)."""
        # C2: every spawn must reach Eagle
        eagle = self.EAGLE_POS
        for sx, sy in self.SPAWN_POINTS:
            if not self._bfs_reach(grid, (sx, sy), eagle):
                return False
        # C4: density ≤ 40 %
        walls = sum(1 for y in range(GRID_H) for x in range(GRID_W)
                    if grid[y][x] in (BRICK, STEEL, WATER))
        if walls / (GRID_W * GRID_H) > 0.40:
            return False
        return True

    def _bfs_reach(self, grid, start, goal):
        visited = {start}
        q = deque([start])
        while q:
            x, y = q.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < GRID_W and 0 <= ny < GRID_H
                        and (nx, ny) not in visited
                        and grid[ny][nx] not in (STEEL, WATER)):
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False

    def _fallback(self):
        """Guaranteed-valid minimal map used only if backtracking fails."""
        grid = [[EMPTY] * GRID_W for _ in range(GRID_H)]
        ex, ey = self.EAGLE_POS
        grid[ey][ex] = EAGLE
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = ex + dx, ey + dy
                if (dx, dy) != (0, 0) and 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    grid[ny][nx] = BRICK
        for y in range(5, 20, 4):
            for x in range(3, 23, 4):
                if grid[y][x] == EMPTY:
                    grid[y][x] = BRICK
        return grid

    @staticmethod
    def boss_arena():
        """Fixed boss arena: 12×12 open area inside steel border."""
        grid = [[STEEL] * GRID_W for _ in range(GRID_H)]
        ax, ay, aw, ah = 7, 7, 12, 12
        for y in range(ay, ay + ah):
            for x in range(ax, ax + aw):
                grid[y][x] = EMPTY
        ex, ey = ax + aw // 2, ay + ah - 2
        grid[ey][ex] = EAGLE
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = ex + dx, ey + dy
                if (dx, dy) != (0, 0):
                    if ay <= ny < ay + ah and ax <= nx < ax + aw:
                        grid[ny][nx] = BRICK
        # Steel pillars
        for px2, py2 in [(ax + 2, ay + 3), (ax + 8, ay + 3),
                         (ax + 2, ay + 7), (ax + 8, ay + 7)]:
            if grid[py2][px2] == EMPTY:
                grid[py2][px2] = STEEL
                if px2 + 1 < ax + aw:
                    grid[py2][px2 + 1] = STEEL
        # Water patch
        grid[ay + 5][ax + 5] = WATER
        grid[ay + 5][ax + 6] = WATER
        return grid, (ex, ey)


# ═══════════════════════════════════════════════════════════════
# SEARCH ALGORITHMS
# ═══════════════════════════════════════════════════════════════

def bfs_path(grid, start, goal):
    """
    BFS: shortest-hop path ignoring terrain cost.
    Treats Empty, Forest, and Eagle (goal) as passable (cost 1).
    Does NOT pass through BRICK or STEEL — takes the open route.
    Returns list of (x,y) steps from start→goal, or [].
    """
    if start == goal:
        return []
    came = {start: None}
    q    = deque([start])
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            path, cur = [], goal
            while cur != start:
                path.append(cur)
                cur = came[cur]
            path.reverse()
            return path
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if (0 <= nx < GRID_W and 0 <= ny < GRID_H
                    and (nx, ny) not in came
                    and grid[ny][nx] not in (STEEL, WATER, BRICK)):
                came[(nx, ny)] = (x, y)
                q.append((nx, ny))
    # Fallback: allow through brick (tank will shoot it)
    if not came.get(goal):
        came2 = {start: None}
        q2 = deque([start])
        while q2:
            x, y = q2.popleft()
            if (x, y) == goal:
                path, cur = [], goal
                while cur != start:
                    path.append(cur)
                    cur = came2[cur]
                path.reverse()
                return path
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < GRID_W and 0 <= ny < GRID_H
                        and (nx, ny) not in came2
                        and grid[ny][nx] not in (STEEL, WATER)):
                    came2[(nx, ny)] = (x, y)
                    q2.append((nx, ny))
    return []


def greedy_step(grid, pos, goal):
    """
    Greedy Best-First: single-step decision using Manhattan heuristic.
    No caching — re-evaluated every tick.
    May get stuck in local minima (intentional per spec).
    """
    sx, sy = pos
    gx, gy = goal
    best, best_h = None, float('inf')
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx, ny = sx + dx, sy + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            if grid[ny][nx] not in (STEEL, WATER):
                h = abs(nx - gx) + abs(ny - gy)
                if h < best_h:
                    best_h = h
                    best   = (nx, ny)
    return best


def astar_path(grid, start, goal):
    """
    A*: cost-optimal path.
    Costs: Empty/Forest=1, Brick=3 (shoot+wait), Steel/Water=∞.
    Eagle tile is the goal — treated as cost 1 (reachable target).
    Finds it cheaper to drill through thin brick than long detours.
    """
    COST = {EMPTY: 1, FOREST: 1, BRICK: 3, EAGLE: 1,
            STEEL: float('inf'), WATER: float('inf')}

    def h(x, y):
        return abs(x - goal[0]) + abs(y - goal[1])

    if start == goal:
        return []
    open_q = [(h(*start), 0, start)]
    came   = {start: None}
    g_sc   = {start: 0}
    while open_q:
        _, g, (x, y) = heapq.heappop(open_q)
        if (x, y) == goal:
            path, cur = [], goal
            while cur != start:
                path.append(cur)
                cur = came[cur]
            path.reverse()
            return path
        if g > g_sc.get((x, y), float('inf')):
            continue
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                c = COST.get(grid[ny][nx], float('inf'))
                if c == float('inf'):
                    continue
                ng = g_sc[(x, y)] + c
                if ng < g_sc.get((nx, ny), float('inf')):
                    g_sc[(nx, ny)] = ng
                    came[(nx, ny)] = (x, y)
                    heapq.heappush(open_q, (ng + h(nx, ny), ng, (nx, ny)))
    return []


# ═══════════════════════════════════════════════════════════════
# BULLET
# ═══════════════════════════════════════════════════════════════
class Bullet:
    """
    Bullets travel at pixel level for smooth visuals.
    Collision is checked against the grid tile the bullet enters.
    Bullets pass through Forest (4); destroyed by Water (3), Brick (1),
    Steel (2), Eagle (5), and tanks.
    """
    def __init__(self, gx, gy, direction, owner='enemy'):
        self.direction = direction
        self.owner     = owner
        self.active    = True
        # Pixel position starts at center of firing tile
        self.px = gx * TILE_SIZE + TILE_SIZE // 2
        self.py = gy * TILE_SIZE + TILE_SIZE // 2
        # Grid position (derived from pixel)
        self.x = gx
        self.y = gy

    def update(self, grid, trail_list):
        """Advance bullet; return (tx, ty, tile_type) on hit, else None."""
        if not self.active:
            return None
        dx, dy = DIR_VEC[self.direction]
        for _ in range(BULLET_SPEED):
            self.px += dx
            self.py += dy
            trail_list.append(TrailParticle(
                self.px, self.py,
                (255, 255, 120) if self.owner == 'player' else (255, 80, 80)))
            tx = self.px // TILE_SIZE
            ty = self.py // TILE_SIZE
            if not (0 <= tx < GRID_W and 0 <= ty < GRID_H):
                self.active = False
                return None
            tile = grid[ty][tx]
            if tile == BRICK:
                grid[ty][tx] = EMPTY
                self.active  = False
                self.x, self.y = tx, ty
                return (tx, ty, BRICK)
            elif tile == STEEL:
                self.active = False
                self.x, self.y = tx, ty
                return (tx, ty, STEEL)
            elif tile == WATER:
                self.active = False
                return None
            elif tile == EAGLE:
                self.active = False
                self.x, self.y = tx, ty
                return (tx, ty, EAGLE)
            # FOREST: pass through
        self.x = self.px // TILE_SIZE
        self.y = self.py // TILE_SIZE
        return None

    def draw(self, surface, ox, oy):
        if not self.active:
            return
        cx  = int(self.px) + ox
        cy  = int(self.py) + oy
        col = YELLOW if self.owner == 'player' else (255, 90, 60)
        glo = (255, 255, 180) if self.owner == 'player' else (255, 180, 120)
        pygame.draw.circle(surface, glo, (cx, cy), 5)
        if self.direction in (UP, DOWN):
            r = pygame.Rect(cx - 2, cy - 6, 5, 12)
        else:
            r = pygame.Rect(cx - 6, cy - 2, 12, 5)
        pygame.draw.rect(surface, col, r, border_radius=2)
        pygame.draw.rect(surface, WHITE, r, 1, border_radius=2)


# ═══════════════════════════════════════════════════════════════
# TANK BASE
# ═══════════════════════════════════════════════════════════════
class Tank:
    color    = WHITE
    firerate = 90   # frames between shots

    def __init__(self, x, y, direction=DOWN):
        self.x, self.y   = x, y
        self.direction   = direction
        self.hp = self.max_hp = 1
        self.active      = True
        self.bullet      = None
        self.fire_cd     = 0
        self.flash       = 0
        # Movement timer: counts down; when 0, tank is allowed to move
        self.move_timer  = 0

    def _tick_cds(self):
        """Decrement cooldowns. Called ONCE per tank per game tick."""
        self.fire_cd   = max(0, self.fire_cd - 1)
        self.move_timer = max(0, self.move_timer - 1)
        if self.flash > 0:
            self.flash -= 1

    def try_move(self, nx, ny, grid, tanks):
        """Attempt to move to (nx,ny). Returns True on success."""
        if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
            return False
        if grid[ny][nx] in (BRICK, STEEL, WATER, EAGLE):
            return False
        for t in tanks:
            if t is not self and t.active and t.x == nx and t.y == ny:
                return False
        self.x, self.y = nx, ny
        return True

    def face(self, dx, dy):
        if   dx > 0: self.direction = RIGHT
        elif dx < 0: self.direction = LEFT
        elif dy > 0: self.direction = DOWN
        else:        self.direction = UP

    def shoot(self, owner='enemy'):
        """Fire a bullet. Returns Bullet or None if on cooldown/already active."""
        if self.fire_cd > 0:
            return None
        if self.bullet and self.bullet.active:
            return None
        dx, dy = DIR_VEC[self.direction]
        bx, by = self.x + dx, self.y + dy
        if not (0 <= bx < GRID_W and 0 <= by < GRID_H):
            return None
        b = Bullet(bx, by, self.direction, owner)
        self.bullet  = b
        self.fire_cd = self.firerate
        return b

    def take_hit(self):
        self.hp   -= 1
        self.flash = 14
        if self.hp <= 0:
            self.active = False

    @staticmethod
    def los(grid, x1, y1, x2, y2):
        """True if (x1,y1) has line-of-sight to (x2,y2) (same row or column, no walls)."""
        if y1 == y2:
            for x in range(min(x1, x2) + 1, max(x1, x2)):
                if grid[y1][x] in (STEEL, WATER, BRICK):
                    return False
            return True
        if x1 == x2:
            for y in range(min(y1, y2) + 1, max(y1, y2)):
                if grid[y][x1] in (STEEL, WATER, BRICK):
                    return False
            return True
        return False

    @staticmethod
    def find_eagle(grid):
        for y in range(GRID_H):
            for x in range(GRID_W):
                if grid[y][x] == EAGLE:
                    return (x, y)
        return None

    def _draw_body(self, surface, rx, ry, col, detail_col, barrel_col):
        ts = TILE_SIZE
        pygame.draw.rect(surface, col, pygame.Rect(rx + 2, ry + 3, ts - 4, ts - 4),
                         border_radius=2)
        # Tracks
        pygame.draw.rect(surface, (0, 0, 0),   pygame.Rect(rx + 1, ry + 3, 4, ts - 6))
        pygame.draw.rect(surface, detail_col,  pygame.Rect(rx + 2, ry + 4, 2, ts - 8))
        pygame.draw.rect(surface, (0, 0, 0),   pygame.Rect(rx + ts - 5, ry + 3, 4, ts - 6))
        pygame.draw.rect(surface, detail_col,  pygame.Rect(rx + ts - 4, ry + 4, 2, ts - 8))
        # Turret
        cx, cy = rx + ts // 2, ry + ts // 2
        pygame.draw.circle(surface, detail_col, (cx, cy + 1), ts // 4 + 1)
        pygame.draw.circle(surface, col,        (cx, cy),     ts // 4)
        pygame.draw.circle(surface, (0, 0, 0),  (cx, cy),     ts // 4, 1)
        # Barrel
        ddx, ddy = DIR_VEC[self.direction]
        blen = ts // 2 + 2
        pygame.draw.line(surface, barrel_col, (cx, cy),
                         (cx + ddx * blen, cy + ddy * blen), 4)
        pygame.draw.line(surface, (0, 0, 0), (cx, cy),
                         (cx + ddx * blen, cy + ddy * blen), 1)

    def draw(self, surface, ox, oy, game=None):
        if game and game.grid[self.y][self.x] == FOREST:
            if self is not getattr(game, 'player', None):
                return
        if not self.active:
            return
        rx  = ox + self.x * TILE_SIZE
        ry  = oy + self.y * TILE_SIZE
        col = WHITE if (self.flash % 2 == 1) else self.color
        self._draw_body(surface, rx, ry, col, GRAY, col)
        if self.max_hp > 1:
            ts     = TILE_SIZE
            bw     = ts - 4
            filled = max(0, int(bw * self.hp / self.max_hp))
            pygame.draw.rect(surface, (60, 0, 0),  (rx + 2, ry - 7, bw, 5))
            pygame.draw.rect(surface, GREEN,        (rx + 2, ry - 7, filled, 5))
            pygame.draw.rect(surface, WHITE,        (rx + 2, ry - 7, bw, 5), 1)
        if self.bullet and self.bullet.active:
            self.bullet.draw(surface, ox, oy)


# ═══════════════════════════════════════════════════════════════
# PLAYER TANK
# ═══════════════════════════════════════════════════════════════
class PlayerTank(Tank):
    """
    Player-controlled tank.
    HP = 1 per spec: "Player tank: 1 hit to destroy one life."
    One life is lost per hit; player starts with 10 lives.
    """
    color    = (80, 210, 90)
    firerate = 18

    def __init__(self, x, y):
        super().__init__(x, y, UP)
        self.hp = self.max_hp = 1   # spec: 1 hit → lose one life
        self.lives      = 10
        self.score      = 0
        self.invincible = 0
        self.combo      = 0
        self.combo_t    = 0

    def handle_input(self, keys, grid, tanks):
        """
        Process keyboard input for movement and update cooldowns.
        Movement is gated by move_timer so each key press advances
        exactly ONE tile.  _tick_cds is called here (and nowhere else
        for the player) to avoid double-decrement bugs.
        """
        self._tick_cds()
        if self.invincible > 0:
            self.invincible -= 1
        if self.combo_t > 0:
            self.combo_t -= 1
            if self.combo_t == 0:
                self.combo = 0

        # Only move when timer has expired (ensures 1-cell-per-interval)
        if self.move_timer > 0:
            return

        moved = False
        for key, d in ((pygame.K_UP, UP),    (pygame.K_w, UP),
                       (pygame.K_DOWN, DOWN), (pygame.K_s, DOWN),
                       (pygame.K_LEFT, LEFT), (pygame.K_a, LEFT),
                       (pygame.K_RIGHT, RIGHT),(pygame.K_d, RIGHT)):
            if keys[key]:
                self.direction = d
                dx, dy = DIR_VEC[d]
                if self.try_move(self.x + dx, self.y + dy, grid, tanks):
                    moved = True
                # Reset timer regardless (prevents rapid repeat on same key)
                self.move_timer = PLAYER_MOVE_INTERVAL
                break   # Only process one direction per frame

        return moved

    def player_shoot(self):
        return self.shoot(owner='player')

    def take_hit(self):
        if self.invincible > 0:
            return
        # Per spec: 1 hit destroys the tank (one life lost)
        self.hp    = 0
        self.flash = 18
        self.active = False

    def add_kill(self):
        self.combo   += 1
        self.combo_t  = 90

    def draw(self, surface, ox, oy, game=None):
        if game and game.grid[self.y][self.x] == FOREST:
            if self is not getattr(game, 'player', None):
                return
        if not self.active:
            return
        if self.invincible > 0 and (self.invincible // 4) % 2 == 0:
            return
        rx  = ox + self.x * TILE_SIZE
        ry  = oy + self.y * TILE_SIZE
        ts  = TILE_SIZE
        col = WHITE if (self.flash % 2 == 1) else (80, 210, 90)
        if self.flash > 0:
            self.flash -= 1
        # Shield ring
        if self.invincible > 0:
            t_val = pygame.time.get_ticks()
            a     = min(120, self.invincible * 3)
            ss    = pygame.Surface((ts + 10, ts + 10), pygame.SRCALPHA)
            sc    = (60, int(160 + 80 * math.sin(t_val / 100)), 255, a)
            pygame.draw.ellipse(ss, sc, (0, 0, ts + 10, ts + 10), 3)
            surface.blit(ss, (rx - 5, ry - 5))
        self._draw_body(surface, rx, ry, col, LIME, YELLOW)
        # Lives as small pips (show up to 10)
        pip_count = min(self.lives, 10)
        for i in range(pip_count):
            pygame.draw.circle(surface, GREEN, (rx + 4 + i * 7, ry - 5), 2)
            pygame.draw.circle(surface, WHITE, (rx + 4 + i * 7, ry - 5), 2, 1)
        if self.bullet and self.bullet.active:
            self.bullet.draw(surface, ox, oy)


# ═══════════════════════════════════════════════════════════════
# BASIC TANK — Simple Reflex Agent + BFS
# ═══════════════════════════════════════════════════════════════
class BasicTank(Tank):
    """
    Simple Reflex Agent — no internal state between decisions.
    Rule set (IF-THEN only, no memory):
      IF player in same row/col with LOS → shoot
      IF next BFS step is BRICK → shoot to clear it
      IF BFS step available → move there
      ELSE → random free direction
    BFS is recomputed fresh each decision (no cached path).
    Re-trigger: every 300 frames (~5 s at 60 fps) per spec.
    """
    color    = (130, 140, 145)
    firerate = 90

    def __init__(self, x, y):
        super().__init__(x, y)
        self.move_interval  = BASIC_MOVE_INTERVAL
        self.path_timer     = 0

    def update(self, grid, player, tanks):
        if not self.active:
            return None
        self._tick_cds()
        self.path_timer += 1
        bullet = None

        # ── Reflex rule 1: LOS shoot at player ───────────────
        if player and player.active:
            if self.los(grid, self.x, self.y, player.x, player.y):
                dx, dy = player.x - self.x, player.y - self.y
                self.direction = ((RIGHT if dx > 0 else LEFT)
                                  if abs(dx) > abs(dy)
                                  else (DOWN if dy > 0 else UP))
                b = self.shoot('enemy')
                if b:
                    bullet = b

        # ── Movement: one step per interval ──────────────────
        if self.move_timer > 0:
            return bullet

        self.move_timer = self.move_interval

        # Recompute BFS fresh (Simple Reflex: no stored path)
        eg   = self.find_eagle(grid)
        path = []
        if eg and self.path_timer >= 300:
            path = bfs_path(grid, (self.x, self.y), eg)
            self.path_timer = 0
        elif eg:
            # Still within interval — recompute anyway (no state stored)
            path = bfs_path(grid, (self.x, self.y), eg)

        if path:
            nx, ny = path[0]
            dx, dy = nx - self.x, ny - self.y
            self.direction = ((RIGHT if dx > 0 else LEFT)
                              if dx != 0
                              else (DOWN if dy > 0 else UP))
            tile = grid[ny][nx]
            # ── Reflex rule 2: shoot brick blocking path ──────
            if tile == BRICK:
                b = self.shoot('enemy')
                if b:
                    bullet = b
            else:
                moved = self.try_move(nx, ny, grid, tanks)
                if not moved:
                    # Blocked by another tank — try random free direction
                    dirs = list(DIR_VEC.keys())
                    random.shuffle(dirs)
                    for d in dirs:
                        ddx, ddy = DIR_VEC[d]
                        rnx, rny = self.x + ddx, self.y + ddy
                        if (0 <= rnx < GRID_W and 0 <= rny < GRID_H
                                and grid[rny][rnx] in (EMPTY, FOREST)):
                            self.direction = d
                            self.try_move(rnx, rny, grid, tanks)
                            break
        else:
            # ── Reflex rule 3: shoot brick in current direction or random move ──
            ddx, ddy = DIR_VEC[self.direction]
            fnx, fny = self.x + ddx, self.y + ddy
            if (0 <= fnx < GRID_W and 0 <= fny < GRID_H
                    and grid[fny][fnx] == BRICK):
                b = self.shoot('enemy')
                if b:
                    bullet = b
            else:
                dirs = list(DIR_VEC.keys())
                random.shuffle(dirs)
                for d in dirs:
                    ddx2, ddy2 = DIR_VEC[d]
                    nx, ny = self.x + ddx2, self.y + ddy2
                    if (0 <= nx < GRID_W and 0 <= ny < GRID_H
                            and grid[ny][nx] in (EMPTY, FOREST)):
                        self.direction = d
                        self.try_move(nx, ny, grid, tanks)
                        break
                    elif (0 <= nx < GRID_W and 0 <= ny < GRID_H
                          and grid[ny][nx] == BRICK):
                        self.direction = d
                        b = self.shoot('enemy')
                        if b:
                            bullet = b
                        break

        return bullet


# ═══════════════════════════════════════════════════════════════
# FAST TANK — Goal-Based Agent + Greedy Best-First
# ═══════════════════════════════════════════════════════════════
class FastTank(Tank):
    """
    Goal-Based Agent: single goal = destroy Eagle.
    Ignores the player entirely (never checks LOS for player).
    Uses Greedy Best-First (single-step, no caching) → may get stuck
    in local minima — this is intentional per spec to show the
    weakness of greedy search vs A*.
    Wall Rule: shoot BRICK blocking next greedy step; never detour.
    """
    color    = (40, 210, 230)
    firerate = 45

    def __init__(self, x, y):
        super().__init__(x, y)
        self.move_interval = FAST_MOVE_INTERVAL

    def update(self, grid, player, tanks):
        if not self.active:
            return None
        self._tick_cds()
        bullet = None
        eg = self.find_eagle(grid)

        if self.move_timer > 0:
            return None

        self.move_timer = self.move_interval

        if eg:
            nxt = greedy_step(grid, (self.x, self.y), eg)
            if nxt:
                nx, ny = nxt
                dx, dy = nx - self.x, ny - self.y
                self.direction = ((RIGHT if dx > 0 else LEFT)
                                  if dx != 0
                                  else (DOWN if dy > 0 else UP))
                tile = grid[ny][nx]
                if tile == BRICK:
                    # Never detour — shoot through
                    b = self.shoot('enemy')
                    if b:
                        bullet = b
                else:
                    moved = self.try_move(nx, ny, grid, tanks)
                    if not moved:
                        # Blocked by tank — shuffle sideways
                        dirs = list(DIR_VEC.keys())
                        random.shuffle(dirs)
                        for d in dirs:
                            ddx, ddy = DIR_VEC[d]
                            rnx, rny = self.x + ddx, self.y + ddy
                            if (0 <= rnx < GRID_W and 0 <= rny < GRID_H
                                    and grid[rny][rnx] in (EMPTY, FOREST)):
                                self.direction = d
                                self.try_move(rnx, rny, grid, tanks)
                                break
            else:
                # Local minimum: pick random free neighbour
                dirs = list(DIR_VEC.keys())
                random.shuffle(dirs)
                for d in dirs:
                    ddx, ddy = DIR_VEC[d]
                    nx, ny   = self.x + ddx, self.y + ddy
                    if (0 <= nx < GRID_W and 0 <= ny < GRID_H
                            and grid[ny][nx] in (EMPTY, FOREST)):
                        self.direction = d
                        self.try_move(nx, ny, grid, tanks)
                        break
        return bullet


# ═══════════════════════════════════════════════════════════════
# ARMOR TANK — Model-Based Reflex Agent + A*
# ═══════════════════════════════════════════════════════════════
class ArmorTank(Tank):
    color = (230, 120, 25)
    firerate = 60

    def __init__(self, x, y):
        super().__init__(x, y)
        self.hp = self.max_hp = 4
        self.move_interval = ARMOR_MOVE_INTERVAL
        self.hit_count = 0
        self.retreating = False
        self.cover_target = None
        self.wait_timer = 0
        self.path_timer = 0
        self._path = []

    def take_hit(self):
        self.hp -= 1
        self.flash = 15
        self.hit_count += 1
       
        if self.hp == 1:
            self.retreating = True
            self.cover_target = None

        if self.hp <= 0:
            self.active = False

    def notify_map_change(self):
        self._path = []
        self.path_timer = 300

    def find_nearest_steel_cover(self, grid):
        q = deque([(self.x, self.y)])
        visited = {(self.x, self.y)}

        while q:
            x, y = q.popleft()

            for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
                nx, ny = x + dx, y + dy

                if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                    continue

                if (nx, ny) in visited:
                    continue

                visited.add((nx, ny))

                if grid[ny][nx] == STEEL:
                    # hide behind steel
                    bx = nx + dx
                    by = ny + dy

                    if 0 <= bx < GRID_W and 0 <= by < GRID_H:
                        if grid[by][bx] == EMPTY:
                            return (bx, by)

                if grid[ny][nx] not in (STEEL, WATER):
                    q.append((nx, ny))

        return None

    def update_ai(self, game):
        self._tick_cds()

        if self.move_timer > 0:
            return None

        if self.retreating:
            if self.cover_target is None:
                self.cover_target = self.find_nearest_steel_cover(game.grid)

            if self.cover_target:
                path = astar_path(game.grid,
                                (self.x, self.y),
                                self.cover_target)

                if path:
                    nx, ny = path[0]
                    dx, dy = nx - self.x, ny - self.y
                    self.face(dx, dy)
                    self.try_move(nx, ny, game.grid, game.all_tanks())
                    self.move_timer = ARMOR_MOVE_INTERVAL
                    return None

                # Reached cover
                if (self.x, self.y) == self.cover_target:
                    self.wait_timer += 1

                    if self.wait_timer >= 120:
                        self.retreating = False
                        self.wait_timer = 0
            return None

        # Attack Phase
        self.path_timer += 1
        if self.path_timer >= 300 or not self._path:
            eg = self.find_eagle(game.grid)
            self._path = astar_path(game.grid, (self.x, self.y), eg) if eg else []
            self.path_timer = 0

        px, py = (game.player.x, game.player.y) if game.player and game.player.active else (999, 999)
        if game.player and game.player.active and self.los(game.grid, self.x, self.y, px, py):
            dx, dy = px - self.x, py - self.y
            self.direction = ((RIGHT if dx > 0 else LEFT)
                              if abs(dx) > abs(dy)
                              else (DOWN if dy > 0 else UP))
            bullet = self.shoot()
            if bullet:
                game.enemy_bullets.append(bullet)
                game.sound.play('shoot')

        if self._path:
            nx, ny = self._path[0]
            dx, dy = nx - self.x, ny - self.y
            self.direction = ((RIGHT if dx > 0 else LEFT)
                              if dx != 0
                              else (DOWN if dy > 0 else UP))
            tile = game.grid[ny][nx]
            if tile == BRICK:
                bullet = self.shoot()
                if bullet:
                    game.enemy_bullets.append(bullet)
                    game.sound.play('shoot')
            elif self.try_move(nx, ny, game.grid, game.all_tanks()):
                self._path.pop(0)
            else:
                self._path = []
                dirs = list(DIR_VEC.keys())
                random.shuffle(dirs)
                for d in dirs:
                    ddx, ddy = DIR_VEC[d]
                    rnx, rny = self.x + ddx, self.y + ddy
                    if (0 <= rnx < GRID_W and 0 <= rny < GRID_H
                            and game.grid[rny][rnx] in (EMPTY, FOREST)):
                        self.direction = d
                        self.try_move(rnx, rny, game.grid, game.all_tanks())
                        break

        self.move_timer = ARMOR_MOVE_INTERVAL
        return None

    def draw(self, surface, ox, oy, game=None):
        if game and game.grid[self.y][self.x] == FOREST:
            if self is not getattr(game, 'player', None):
                return
        if not self.active:
            return
        stage_cols = [(230, 120, 25), (200, 90, 15), (170, 60, 10), (140, 30, 5)]
        self.color = stage_cols[min(self.hit_count, 3)]
        super().draw(surface, ox, oy, game)
        if self.retreating:
            rx  = ox + self.x * TILE_SIZE
            ry  = oy + self.y * TILE_SIZE
            ts  = TILE_SIZE
            t_v = pygame.time.get_ticks()
            a   = int(128 + 127 * math.sin(t_v / 150))
            pygame.draw.rect(surface, (40, 80, a), (rx, ry, ts, ts), 2)


# ═══════════════════════════════════════════════════════════════
# POWER TANK — Utility-Based Agent + A*
# (Not in core spec but required by Level 2 enemy pool)
# ═══════════════════════════════════════════════════════════════
class PowerTank(Tank):
    color = PURPLE
    firerate = 35

    def __init__(self, x, y):
        super().__init__(x, y, DOWN)
        self.hp = self.max_hp = 2

    def update_ai(self, game):
        self._tick_cds()

        if self.move_timer > 0:
            return None

        px, py = (game.player.x, game.player.y) if game.player and game.player.active else (999, 999)
        eagle = self.find_eagle(game.grid)

        # Utility calculations
        dist_player = abs(self.x - px) + abs(self.y - py)
        dist_eagle  = abs(self.x - eagle[0]) + abs(self.y - eagle[1]) if eagle else 999

        attack_utility = 20 - dist_player
        eagle_utility  = 30 - dist_eagle

        # Choose goal dynamically
        if attack_utility > eagle_utility and game.player and game.player.active:
            target = (px, py)
        else:
            target = eagle

        if target:
            path = astar_path(game.grid, (self.x, self.y), target)

            if path:
                nx, ny = path[0]
                dx, dy = nx - self.x, ny - self.y
                self.face(dx, dy)

                tile = game.grid[ny][nx]

                if tile == BRICK:
                    bullet = self.shoot()
                    if bullet:
                        game.enemy_bullets.append(bullet)
                        game.sound.play('shoot')
                else:
                    self.try_move(nx, ny, game.grid, game.all_tanks())

        # Shoot player if visible
        if game.player and game.player.active and self.los(game.grid, self.x, self.y, px, py):
            bullet = self.shoot()
            if bullet:
                game.enemy_bullets.append(bullet)
                game.sound.play('shoot')

        self.move_timer = 12
        return None


# ═══════════════════════════════════════════════════════════════
# BOSS TANK (Minimax + Alpha-Beta Pruning)
# ═══════════════════════════════════════════════════════════════
class BossTank(Tank):
    color = RED

    def __init__(self, x, y):
        super().__init__(x, y, DOWN)

        self.hp = self.max_hp = 10
        self.phase = 1
        self.nodes_total = 0
        self.nodes_pruned = 0
        self.last_depth = 2

    def update_phase(self):

        if self.hp >= 7:
            self.phase = 1
            self.firerate = 120

        elif self.hp >= 3:
            self.phase = 2
            self.firerate = 90

        else:
            self.phase = 3
            self.firerate = 50

    def get_depth(self):
        if self.phase == 1:
            return 2
        elif self.phase == 2:
            return 3
        return 4

    def evaluate(self, game):
        if not game.player or not game.player.active: return 0
        px, py = game.player.x, game.player.y

        score = 0

        dist = abs(self.x - px) + abs(self.y - py)

        if dist <= 3:
            score += 60

        if self.los(game.grid, self.x, self.y, px, py):
            score += 50

        # cover bonus
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
            nx, ny = self.x + dx, self.y + dy

            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                if game.grid[ny][nx] == STEEL:
                    score += 30

        score += (10 - game.player.lives) * 20

        score -= (10 - self.hp) * 40

        if game.grid[py][px] == FOREST:
            score -= 20

        return score

    def minimax(self, game, depth, alpha, beta, maximizing):
        self.nodes_total += 1
        if depth == 0 or not game.player or not game.player.active:
            return self.evaluate(game), None

        actions = [UP, DOWN, LEFT, RIGHT, 'SHOOT']

        best_action = None

        if maximizing:
            max_eval = -float('inf')

            for action in actions:
                if action == 'SHOOT':
                    val, _ = self.minimax(game, depth - 1, alpha, beta, False)
                else:
                    dx, dy = DIR_VEC[action]
                    nx = self.x + dx
                    ny = self.y + dy

                    if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                        continue

                    if game.grid[ny][nx] in (STEEL, WATER, BRICK):
                        continue

                    oldx, oldy = self.x, self.y
                    self.x, self.y = nx, ny

                    val, _ = self.minimax(game,
                                          depth - 1,
                                          alpha,
                                          beta,
                                          False)

                    self.x, self.y = oldx, oldy

                if val > max_eval:
                    max_eval = val
                    best_action = action

                alpha = max(alpha, val)

                if beta <= alpha:
                    self.nodes_pruned += 1
                    break

            return max_eval, best_action

        else:
            min_eval = float('inf')

            px, py = game.player.x, game.player.y

            for action in actions:
                if action == 'SHOOT':
                    val, _ = self.minimax(game, depth - 1, alpha, beta, True)
                else:
                    dx, dy = DIR_VEC[action]
                    nx = px + dx
                    ny = py + dy

                    if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                        continue

                    if game.grid[ny][nx] in (STEEL, WATER, BRICK):
                        continue

                    oldx, oldy = game.player.x, game.player.y
                    game.player.x, game.player.y = nx, ny

                    val, _ = self.minimax(game,
                                          depth - 1,
                                          alpha,
                                          beta,
                                          True)

                    game.player.x, game.player.y = oldx, oldy

                min_eval = min(min_eval, val)

                beta = min(beta, val)

                if beta <= alpha:
                    self.nodes_pruned += 1
                    break

            return min_eval, None

    def update_ai(self, game):

        self._tick_cds()
        self.update_phase()

        if self.move_timer > 0:
            return

        if not game.player or not game.player.active:
            return

        self.nodes_total = 0
        self.nodes_pruned = 0
        self.last_depth = self.get_depth()

        _, action = self.minimax(game,
                                 self.last_depth,
                                 -float('inf'),
                                 float('inf'),
                                 True)

        if action is not None:
            if action == 'SHOOT':
                bullet = self.shoot()
                if bullet:
                    game.enemy_bullets.append(bullet)
                    game.sound.play('boss_fire')
            else:
                dx, dy = DIR_VEC[action]
                nx = self.x + dx
                ny = self.y + dy

                self.face(dx, dy)

                self.try_move(nx,
                              ny,
                              game.grid,
                              game.all_tanks())

        px, py = game.player.x, game.player.y

        if self.los(game.grid, self.x, self.y, px, py):
            bullet = self.shoot()

            if bullet:
                game.enemy_bullets.append(bullet)
                game.sound.play('boss_fire')

        self.move_timer = BOSS_MOVE_INTERVAL[self.phase]

    def draw(self, surface, ox, oy, game=None):
        if game and game.grid[self.y][self.x] == FOREST:
            if self is not getattr(game, 'player', None):
                return
        if not self.active:
            return
        rx  = ox + self.x * TILE_SIZE
        ry  = oy + self.y * TILE_SIZE
        ts  = TILE_SIZE
        t_v = pygame.time.get_ticks()
        phase_c = {1: (200, 40, 40), 2: (220, 100, 20), 3: (240, 40, 200)}
        col = WHITE if (self.flash % 2 == 1) else phase_c[self.phase]
        if self.flash > 0:
            self.flash -= 1

        if self.phase == 3:
            gr = int(ts // 2 + 4 * math.sin(t_v / 80))
            gs = pygame.Surface((ts * 3, ts * 3), pygame.SRCALPHA)
            pygame.draw.circle(gs, (240, 40, 200, 40), (ts * 3 // 2, ts * 3 // 2), gr)
            surface.blit(gs, (rx - ts, ry - ts))

        pygame.draw.rect(surface, (0, 0, 0), pygame.Rect(rx + 1, ry + 1, ts - 2, ts - 2))
        pygame.draw.rect(surface, col,       pygame.Rect(rx + 2, ry + 2, ts - 4, ts - 4),
                         border_radius=2)
        pygame.draw.rect(surface, (0, 0, 0), pygame.Rect(rx + 6, ry + 6, ts - 12, ts - 12))
        pygame.draw.rect(surface, col,       pygame.Rect(rx + 6, ry + 6, ts - 12, ts - 12), 1)
        pygame.draw.rect(surface, GOLD,      pygame.Rect(rx + 1, ry + 1, ts - 2, ts - 2), 2)

        cx, cy   = rx + ts // 2, ry + ts // 2
        ddx, ddy = DIR_VEC[self.direction]
        pygame.draw.line(surface, GOLD, (cx, cy),
                         (cx + ddx * (ts // 2 + 4), cy + ddy * (ts // 2 + 4)), 6)

        bw     = ts - 2
        filled = max(0, int(bw * self.hp / self.max_hp))
        pygame.draw.rect(surface, (80, 0, 0), (rx + 1, ry - 9, bw, 6))
        pygame.draw.rect(surface, RED,        (rx + 1, ry - 9, filled, 6))
        pygame.draw.rect(surface, GOLD,       (rx + 1, ry - 9, bw, 6), 1)

        fnt = pygame.font.SysFont('Consolas', 9, bold=True)
        surface.blit(fnt.render(f'P{self.phase}', True, GOLD), (rx + 2, ry + 2))
        if self.bullet and self.bullet.active:
            self.bullet.draw(surface, ox, oy)


# ═══════════════════════════════════════════════════════════════
# LEVEL MANAGER
# ═══════════════════════════════════════════════════════════════
class LevelManager:
    """
    Controls enemy pool, spawning, and per-level configuration.
    Level 1: 7 Basic + 13 Fast = 20 total (Fast after first 7 kills)
    Level 2: 6 Fast + 6 Armor + 4 Power + 4 Fast = 20 total
    Level 3: Boss only (1 tank, 10 HP)
    Spawn fairness: no enemy spawns within 10 Manhattan tiles of player.
    """
    CFGS = {
        1: dict(name='Brick Maze',
                max_active=3,
                pool=(['basic'] * 7 + ['fast'] * 13)),
        2: dict(name='Steel Fortress',
                max_active=3,
                pool=(['fast'] * 8 + ['armor'] * 7 + ['power'] * 5)),
        3: dict(name='Tank Commander',
                max_active=1,
                pool=['boss']),
    }
    SPAWN_POINTS = [(0, 0), (12, 0), (24, 0)]
    _TYPE_MAP = {
        'basic': BasicTank,
        'fast':  FastTank,
        'armor': ArmorTank,
        'power': PowerTank,
        'boss':  BossTank,
    }

    def __init__(self, level, player):
        cfg            = self.CFGS[level]
        self.name      = cfg['name']
        self.pool      = list(cfg['pool'])
        self.max_active = cfg['max_active']
        self.player    = player
        self.total     = len(self.pool)
        self.killed    = 0
        self._spawn_cd = 60
        self._sidx     = 0

    def try_spawn(self, active_enemies):
        if not self.pool:
            return None
        if len(active_enemies) >= 3:
            return None
        self._spawn_cd -= 1
        if self._spawn_cd > 0:
            return None
        self._spawn_cd = 60

        valid_spawns = []
        if self.player and self.player.active:
            for sx, sy in self.SPAWN_POINTS:
                dist = abs(sx - self.player.x) + abs(sy - self.player.y)
                if dist >= 10:
                    valid_spawns.append((sx, sy))
        else:
            valid_spawns = list(self.SPAWN_POINTS)

        if valid_spawns:
            chosen = valid_spawns[self._sidx % len(valid_spawns)]
            self._sidx += 1
        else:
            chosen = self.SPAWN_POINTS[self._sidx % len(self.SPAWN_POINTS)]
            self._sidx += 1

        tank_type = self.pool.pop(0)
        return self._TYPE_MAP[tank_type](chosen[0], chosen[1])


# ═══════════════════════════════════════════════════════════════
# MAIN GAME
# ═══════════════════════════════════════════════════════════════
class Game:
    GRID_OX = 0
    GRID_OY = 28   # top bar height

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Battle City")
        self.clock  = pygame.time.Clock()

        self.sfx = SoundEngine()

        self.F_LG    = pygame.font.SysFont('Consolas', 32, bold=True)
        self.F_MD    = pygame.font.SysFont('Consolas', 18, bold=True)
        self.F_SM    = pygame.font.SysFont('Consolas', 13)
        self.F_XS    = pygame.font.SysFont('Consolas', 11)
        self.F_TITLE = pygame.font.SysFont('Consolas', 42, bold=True)

        self.state          = ST_MENU
        self.level          = 1
        self.grid           = None
        self.player         = None
        self.enemies        = []
        self.bullets        = []
        self.particles      = []
        self.trail_parts    = []
        self.explosions     = []
        self.score_popups   = []
        self.spawn_anims    = []
        self.lvlmgr         = None
        self.tick           = 0
        self.flash_col      = WHITE
        self.flash_t        = 0
        self.msg            = ''
        self.msg_t          = 0
        self.shake          = ScreenShake()
        self.respawn_timer  = 0
        self.hud_x          = TILE_SIZE * GRID_W + 5

    @property
    def sound(self):
        return self.sfx

    @property
    def enemy_bullets(self):
        return self.bullets

    def all_tanks(self):
        return ([self.player] if self.player else []) + self.enemies

    # ─────────────────────────────────────────────────────────────
    # BULLET VS BULLET COLLISION
    # ─────────────────────────────────────────────────────────────
    def handle_bullet_collisions(self):
        all_bullets = []

        # Collect bullets
        if self.player and self.player.bullet and self.player.bullet.active:
            all_bullets.append(self.player.bullet)

        for enemy in self.enemies:
            if enemy.bullet and enemy.bullet.active:
                all_bullets.append(enemy.bullet)

        # Check collisions
        for i in range(len(all_bullets)):
            for j in range(i + 1, len(all_bullets)):
                b1 = all_bullets[i]
                b2 = all_bullets[j]

                if not b1.active or not b2.active:
                    continue

                dist = abs(b1.px - b2.px) + abs(b1.py - b2.py)

                if dist < 10:
                    b1.active = False
                    b2.active = False

                    self.sound.play('bullet_cancel')
                    self.shake.add(0.15)

    # ── Level setup ───────────────────────────────────────────
    def _start_level(self, level, carry_player=True):
        """
        Initialize a level. If carry_player=True, preserve score/lives
        from the previous level (used for level progression).
        """
        self.level       = level
        self.enemies     = []
        self.bullets     = []
        self.particles   = []
        self.trail_parts = []
        self.explosions  = []
        self.score_popups = []
        self.spawn_anims = []
        self.tick        = 0
        self.flash_t     = 0
        self.respawn_timer = 0

        if level == 3:
            self.grid, _ = MapGenerator.boss_arena()
            px, py = 8, 17
        else:
            self.grid = MapGenerator(level).generate()
            px, py   = 4, 24

        if self.player is None or not carry_player:
            self.player = PlayerTank(px, py)
        else:
            # Carry over score and lives; reset position and HP
            self.player.x, self.player.y = px, py
            self.player.hp       = self.player.max_hp
            self.player.active   = True
            self.player.invincible = 150
            self.player.bullet   = None
            self.player.move_timer = 0
            self.player.fire_cd  = 0

        self.lvlmgr = LevelManager(level, self.player)
        self.state  = ST_PLAYING
        self.sfx.stop_boss_ambience()
        self.msg   = f'LEVEL {level}  —  {self.lvlmgr.name}'
        self.msg_t = 120

    # ── Event handling ────────────────────────────────────────
    def _events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if ev.type == pygame.KEYDOWN:
                k = ev.key

                if self.state == ST_MENU:
                    if k in (pygame.K_RETURN, pygame.K_SPACE):
                        self.player = None
                        self._start_level(1, carry_player=False)
                    elif k == pygame.K_2:
                        self.player = None
                        self._start_level(2, carry_player=False)
                    elif k == pygame.K_3:
                        self.player = None
                        self._start_level(3, carry_player=False)
                    elif k == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

                elif self.state == ST_PLAYING:
                    if k == pygame.K_SPACE:
                        b = self.player.player_shoot()
                        if b:
                            self.bullets.append(b)
                            self.sfx.play('shoot', 0.55)
                    elif k == pygame.K_m:
                        muted = self.sfx.toggle_mute()
                        self.msg   = 'SOUND OFF' if muted else 'SOUND ON'
                        self.msg_t = 60
                    elif k == pygame.K_ESCAPE:
                        self.sfx.stop_engine()
                        self.sfx.stop_boss_ambience()
                        self.state  = ST_MENU
                        self.flash_t = 0
                        self.player = None

                elif self.state == ST_LVLDONE:
                    if k == pygame.K_n:
                        # Progress to next level, carrying player stats
                        self._start_level(self.level + 1, carry_player=True)
                    elif k in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_ESCAPE):
                        self.sfx.stop_engine()
                        self.sfx.stop_boss_ambience()
                        self.state  = ST_MENU
                        self.flash_t = 0
                        self.player = None

                elif self.state in (ST_GAMEOVER, ST_WIN, ST_BOSSDEAD):
                    if k in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_ESCAPE):
                        self.sfx.stop_engine()
                        self.sfx.stop_boss_ambience()
                        self.state  = ST_MENU
                        self.flash_t = 0
                        self.player = None

    # ── Explosion helper ─────────────────────────────────────
    def _explode(self, gx, gy, size=1):
        self.shake.add(0.3 * size)
        self.explosions.append(Explosion(gx, gy, size))
        for _ in range(10 + size * 8):
            col = random.choice([YELLOW, ORANGE, RED, WHITE, (255, 160, 60)])
            self.particles.append(Particle(gx, gy, col, size=4 + size, speed=3 + size))
        vol = 1.0
        if self.player and self.player.active:
            dist = abs(gx - self.player.x) + abs(gy - self.player.y)
            vol  = max(0.2, 1.0 - dist / 30.0)
        self.sfx.play('explode_lg' if size >= 2 else 'explode_sm', vol)

    # ── Game update ───────────────────────────────────────────
    def _update(self):
        if self.state != ST_PLAYING:
            return
        self.tick += 1
        self.shake.update()
        if self.msg_t > 0:
            self.msg_t -= 1
        if self.flash_t > 0:
            self.flash_t -= 1

        keys = pygame.key.get_pressed()
        p    = self.player

        # ── Player movement ───────────────────────────────────
        if p and p.active:
            p.handle_input(keys, self.grid, [p] + self.enemies)
            # Engine sound
            moving = (p.move_timer > 0)
            if moving:
                self.sfx.play_engine(0.12)
            else:
                self.sfx.stop_engine()

        # ── Player respawn ────────────────────────────────────
        elif p and not p.active:
            if self.respawn_timer == 0:
                self.respawn_timer = 120   # 2 s
            self.respawn_timer -= 1
            if self.respawn_timer <= 0:
                self.respawn_timer = 0
                p.lives -= 1
                if p.lives <= 0:
                    self.sfx.play('game_over')
                    self._trigger(ST_GAMEOVER, RED)
                    return
                px, py = (8, 17) if self.level == 3 else (4, 24)
                p.x, p.y     = px, py
                p.hp         = p.max_hp
                p.active     = True
                p.invincible = 180
                p.bullet     = None
                p.move_timer = 0
                self.spawn_anims.append(SpawnAnim(p.x, p.y))

        # ── Spawn enemies ─────────────────────────────────────
        active_en = [e for e in self.enemies if e.active]
        new_t = self.lvlmgr.try_spawn(active_en)
        if new_t:
            self.enemies.append(new_t)
            self.spawn_anims.append(SpawnAnim(new_t.x, new_t.y))
            self.sfx.play('spawn', 0.4)

        # ── Spawn animations ──────────────────────────────────
        for sa in self.spawn_anims:
            sa.update()
        self.spawn_anims = [sa for sa in self.spawn_anims if sa.alive]

        # ── Enemy AI update ───────────────────────────────────
        all_tanks = ([p] if p else []) + self.enemies
        for e in self.enemies:
            if not e.active:
                continue
            if hasattr(e, 'update_ai'):
                b = e.update_ai(self)
                if b:
                    self.bullets.append(b)
            else:
                b = e.update(self.grid, p, all_tanks)
                if b:
                    self.bullets.append(b)

        # Boss ambience
        boss = next((e for e in self.enemies
                     if isinstance(e, BossTank) and e.active), None)
        if boss:
            self.sfx.play_boss_ambience(boss.phase)
        else:
            self.sfx.stop_boss_ambience()

        # ── Bullet updates ────────────────────────────────────
        map_changed = False
        for b in list(self.bullets):
            if not b.active:
                continue
            res = b.update(self.grid, self.trail_parts)
            if res is None:
                continue
            rx, ry, tt = res
            if tt == BRICK:
                map_changed = True
                self._explode(rx, ry, size=1)
            elif tt == STEEL:
                self.shake.add(0.1)
                self.sfx.play('hit_steel', 0.5)
                for _ in range(6):
                    self.particles.append(Particle(rx, ry, SILVER, size=3, speed=2))
            elif tt == EAGLE:
                self._explode(rx, ry, size=3)
                self.sfx.play('explode_lg')
                self._trigger(ST_GAMEOVER, RED)
                return

        if map_changed:
            for e in self.enemies:
                if hasattr(e, 'notify_map_change'):
                    e.notify_map_change()

        # ── Bullet vs Bullet ──────────────────────────────────
        self.handle_bullet_collisions()

        # ── Bullet vs Tank ────────────────────────────────────
        for b in [x for x in self.bullets if x.active]:
            # Enemy bullet hits player
            if b.owner == 'enemy' and p and p.active:
                if b.x == p.x and b.y == p.y:
                    b.active = False
                    p.take_hit()
                    self.shake.add(0.4)
                    self.sfx.play('hit_player', 0.7)
                    for _ in range(12):
                        self.particles.append(Particle(p.x, p.y, GREEN, size=4))
                    if not p.active:
                        self._explode(p.x, p.y, size=2)
                        self.flash_t   = 25
                        self.flash_col = RED
                        self.respawn_timer = 0

            # Player bullet hits enemy
            if b.owner == 'player':
                for e in self.enemies:
                    if not (e.active and b.active):
                        continue
                    if b.x == e.x and b.y == e.y:
                        b.active = False
                        e.take_hit()
                        self.sfx.play('hit_player', 0.5)
                        for _ in range(8):
                            self.particles.append(Particle(e.x, e.y, ORANGE, size=4))
                        if not e.active:
                            self.lvlmgr.killed += 1
                            base  = {BossTank: 5000, ArmorTank: 400,
                                     PowerTank: 300,  FastTank: 200}.get(type(e), 100)
                            p.add_kill()
                            combo_bonus = (p.combo - 1) * 50 if p.combo > 1 else 0
                            total       = base + combo_bonus
                            p.score    += total
                            self._explode(e.x, e.y, size=2)
                            col = GOLD if combo_bonus > 0 else YELLOW
                            self.score_popups.append(ScorePopup(e.x, e.y, total, col))

        # ── Cleanup ───────────────────────────────────────────
        self.bullets     = [b for b in self.bullets if b.active]
        self.enemies     = [e for e in self.enemies if e.active]
        for pt in self.particles:  pt.update()
        self.particles   = [pt for pt in self.particles if pt.life > 0]
        for tp in self.trail_parts: tp.update()
        self.trail_parts = [tp for tp in self.trail_parts if tp.life > 0]
        for ex in self.explosions:  ex.update()
        self.explosions  = [ex for ex in self.explosions if ex.alive]
        for sp in self.score_popups: sp.update()
        self.score_popups = [sp for sp in self.score_popups if sp.life > 0]

        # ── Win / Level-complete check ────────────────────────
        if not self.lvlmgr.pool and not self.enemies:
            if self.level == 3:
                self.sfx.play('level_win')
                self._trigger(ST_BOSSDEAD, GOLD)
            elif self.level == 2:
                self.sfx.play('level_win')
                self._trigger(ST_WIN, GOLD)
            else:
                self.sfx.play('level_win')
                self._trigger(ST_LVLDONE, GREEN)

    def _trigger(self, state, col):
        self.state     = state
        self.flash_col = col
        self.flash_t   = 50

    # ── Drawing ───────────────────────────────────────────────
    def _draw(self):
        sox, soy = self.shake.offset()
        self.screen.fill(DARK)

        if self.state == ST_MENU:
            self._draw_menu()
        else:
            self._draw_game(sox, soy)
            if self.state != ST_PLAYING:
                texts = {
                    ST_LVLDONE:  ("LEVEL COMPLETE!", GREEN),
                    ST_GAMEOVER: ("GAME  OVER",      RED),
                    ST_WIN:      ("VICTORY!",         GOLD),
                    ST_BOSSDEAD: ("BOSS DEFEATED!",   GOLD),
                }
                if self.state in texts:
                    self._draw_overlay(*texts[self.state])

        # Full-screen flash
        if self.flash_t > 0:
            a  = min(160, self.flash_t * 5)
            fs = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            fs.fill((*self.flash_col, a))
            self.screen.blit(fs, (0, 0))

        # Message banner
        if self.msg_t > 0:
            s  = self.F_MD.render(self.msg, True, YELLOW)
            r  = s.get_rect(center=(TILE_SIZE * GRID_W // 2,
                                    self.GRID_OY + TILE_SIZE * GRID_H // 2))
            bg = pygame.Surface((r.w + 28, r.h + 14), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 200))
            self.screen.blit(bg, (r.x - 14, r.y - 7))
            self.screen.blit(s, r)

        # Respawn countdown
        if self.player and not self.player.active and self.respawn_timer > 0:
            secs = math.ceil(self.respawn_timer / 60)
            cs   = self.F_LG.render(f"RESPAWN  {secs}", True, YELLOW)
            cr   = cs.get_rect(center=(TILE_SIZE * GRID_W // 2,
                                       self.GRID_OY + TILE_SIZE * GRID_H // 2 + 60))
            cbg  = pygame.Surface((cr.w + 20, cr.h + 10), pygame.SRCALPHA)
            cbg.fill((0, 0, 0, 180))
            self.screen.blit(cbg, (cr.x - 10, cr.y - 5))
            self.screen.blit(cs, cr)

        # Combo indicator
        if self.player and self.player.combo > 1 and self.player.combo_t > 0:
            ca = min(255, self.player.combo_t * 3)
            cs = self.F_MD.render(f"COMBO ×{self.player.combo}!", True, GOLD)
            cs.set_alpha(ca)
            self.screen.blit(cs, (10, self.GRID_OY + TILE_SIZE * GRID_H // 2 - 40))

        pygame.display.flip()

    # ── Menu ─────────────────────────────────────────────────
    def _draw_menu(self):
        # Dark background with subtle grid pattern
        self.screen.fill((8, 10, 14))
        for gx in range(0, SCREEN_W, 32):
            pygame.draw.line(self.screen, (16, 18, 24), (gx, 0), (gx, SCREEN_H))
        for gy in range(0, SCREEN_H, 32):
            pygame.draw.line(self.screen, (16, 18, 24), (0, gy), (SCREEN_W, gy))

        t_val = self.tick
        cx = SCREEN_W // 2

        # ── Animated tank parade background ───────────────────
        for i, (col, yy) in enumerate([(GRAY, 548), (CYAN, 565), (ORANGE, 548),
                                        (RED, 565), (PURPLE, 548)]):
            spd = 1.2 + i * 0.3
            xx  = int((t_val * spd + i * 180) % (SCREEN_W + 80)) - 40
            pygame.draw.rect(self.screen, col, (xx, yy, 18, 16), border_radius=2)
            pygame.draw.rect(self.screen, (0, 0, 0), (xx, yy, 18, 16), 1, border_radius=2)
            pygame.draw.line(self.screen, col, (xx + 9, yy + 8), (xx + 9, yy - 2), 3)

        # ── Glowing title with pixel style ────────────────────
        pulse = abs(math.sin(t_val * 0.03))
        title_col = (int(230 + 25 * pulse), int(170 + 20 * pulse), 0)

        # Draw blocky shadow layers for depth
        for off in [(4, 4), (2, 2)]:
            sh = self.F_TITLE.render("BATTLE  CITY", True, (30 + off[0]*3, 20, 0))
            self.screen.blit(sh, sh.get_rect(center=(cx + off[0], 68 + off[1])))
        title = self.F_TITLE.render("BATTLE  CITY", True, title_col)
        self.screen.blit(title, title.get_rect(center=(cx, 66)))

        # Subtitle line
        sub_txt = "TANK 1990  —  Classic Arcade"
        sub = self.F_XS.render(sub_txt, True, (120, 120, 130))
        self.screen.blit(sub, sub.get_rect(center=(cx, 108)))

        # Decorative horizontal rule
        for y_off, col in [(128, (60, 50, 0)), (130, GOLD), (132, (60, 50, 0))]:
            pygame.draw.line(self.screen, col, (50, y_off), (SCREEN_W - 50, y_off), 1)

        # ── Level select boxes ────────────────────────────────
        entries = [
            ("ENTER", "Level 1 — Brick Maze",     GREEN),
            ("  2  ", "Level 2 — Steel Fortress",  CYAN),
            ("  3  ", "Level 3 — Boss Battle",      RED),
            (" ESC ", "Exit",                       GRAY),
        ]
        for i, (key, label, col) in enumerate(entries):
            yb = 155 + i * 46
            bx = cx - 230
            bw, bh = 460, 38

            # Box background
            bg = pygame.Surface((bw, bh), pygame.SRCALPHA)
            bg.fill((col[0] // 10, col[1] // 10, col[2] // 10, 100))
            self.screen.blit(bg, (bx, yb))
            pygame.draw.rect(self.screen, col, (bx, yb, bw, bh), 1, border_radius=3)

            # Key badge
            kbg = pygame.Surface((54, bh - 4), pygame.SRCALPHA)
            kbg.fill((col[0] // 5, col[1] // 5, col[2] // 5, 180))
            self.screen.blit(kbg, (bx + 2, yb + 2))
            pygame.draw.rect(self.screen, col, (bx + 2, yb + 2, 54, bh - 4), 1, border_radius=2)
            ks = self.F_XS.render(key, True, YELLOW)
            self.screen.blit(ks, ks.get_rect(center=(bx + 29, yb + bh // 2)))

            ls = self.F_SM.render(label, True, col)
            self.screen.blit(ls, (bx + 66, yb + bh // 2 - ls.get_height() // 2))

        # ── Controls section ─────────────────────────────────
        pygame.draw.line(self.screen, (50, 50, 60), (50, 345), (SCREEN_W - 50, 345), 1)
        ch = self.F_XS.render("CONTROLS", True, (160, 160, 170))
        self.screen.blit(ch, ch.get_rect(center=(cx, 360)))

        ctrl_pairs = [("WASD / Arrows", "Move"),
                      ("SPACE", "Shoot"),
                      ("M", "Mute / Unmute"),
                      ("ESC", "Menu")]
        col_w = 240
        for i, (k, v) in enumerate(ctrl_pairs):
            col_idx = i % 2
            row_idx = i // 2
            tx = cx - col_w + col_idx * col_w
            ty = 378 + row_idx * 20
            ks = self.F_XS.render(k, True, YELLOW)
            vs = self.F_XS.render(f": {v}", True, LIGHT_GRAY)
            self.screen.blit(ks, (tx, ty))
            self.screen.blit(vs, (tx + ks.get_width() + 2, ty))

        # ── AI info strip ────────────────────────────────────
        pygame.draw.line(self.screen, (50, 50, 60), (50, 422), (SCREEN_W - 50, 422), 1)
        ai_line = "BFS  ·  Greedy  ·  A*  ·  Minimax + Alpha-Beta  ·  CSP"
        ai = self.F_XS.render(ai_line, True, (70, 140, 180))
        self.screen.blit(ai, ai.get_rect(center=(cx, 436)))

    # ── Game world ────────────────────────────────────────────
    def _draw_game(self, sox=0, soy=0):
        ox = self.GRID_OX + sox
        oy = self.GRID_OY + soy

        pygame.draw.rect(self.screen, (10, 12, 16),
                         (self.GRID_OX, self.GRID_OY,
                          GRID_W * TILE_SIZE, GRID_H * TILE_SIZE))

        t_val        = pygame.time.get_ticks()
        forest_tiles = []

        for y in range(GRID_H):
            for x in range(GRID_W):
                tile = self.grid[y][x]
                if tile == EMPTY:
                    continue
                rx = ox + x * TILE_SIZE
                ry = oy + y * TILE_SIZE
                ts = TILE_SIZE

                if tile == BRICK:
                    pygame.draw.rect(self.screen, (158, 62, 18), (rx, ry, ts, ts))
                    pygame.draw.line(self.screen, (100, 40, 10),
                                     (rx, ry + ts // 2), (rx + ts, ry + ts // 2), 1)
                    pygame.draw.line(self.screen, (100, 40, 10),
                                     (rx + ts // 2, ry), (rx + ts // 2, ry + ts // 2), 1)
                    pygame.draw.rect(self.screen, (195, 88, 35),
                                     (rx + 1, ry + 1, ts // 2 - 2, ts // 2 - 2))
                    pygame.draw.rect(self.screen, (195, 88, 35),
                                     (rx + ts // 2 + 1, ry + ts // 2 + 2, ts // 2 - 2, ts // 2 - 3))

                elif tile == STEEL:
                    pygame.draw.rect(self.screen, (78, 92, 108), (rx, ry, ts, ts))
                    for ri in range(2):
                        for rj in range(2):
                            cx2 = rx + 5 + ri * (ts - 10)
                            cy2 = ry + 5 + rj * (ts - 10)
                            pygame.draw.circle(self.screen, (140, 158, 178), (cx2, cy2), 3)
                    pygame.draw.rect(self.screen, (150, 168, 188), (rx + 2, ry + 2, ts - 4, 3))
                    pygame.draw.rect(self.screen, (150, 168, 188), (rx + 2, ry + 2, 3, ts - 4))

                elif tile == WATER:
                    wave  = math.sin(t_val * 0.003 + x * 0.5 + y * 0.3)
                    r_c   = int(18 + wave * 6)
                    g_c   = int(55 + wave * 10)
                    b_c   = int(165 + wave * 20)
                    pygame.draw.rect(self.screen, (r_c, g_c, b_c), (rx, ry, ts, ts))
                    for wr in range(3):
                        wy   = ry + 3 + wr * 7 + int(wave * 2)
                        wcol = (50 + wr * 10, 110 + wr * 15, 210 + wr * 10)
                        pygame.draw.line(self.screen, wcol, (rx + 1, wy), (rx + ts - 2, wy), 2)

                elif tile == FOREST:
                    forest_tiles.append((x, y))
                    pygame.draw.rect(self.screen, (14, 58, 16), (rx, ry, ts, ts))

                elif tile == EAGLE:
                    pygame.draw.rect(self.screen, (120, 88, 15), (rx, ry, ts, ts))
                    cx2, cy2 = rx + ts // 2, ry + ts // 2
                    threatened = any(abs(e.x - x) + abs(e.y - y) < 5
                                     for e in self.enemies if e.active)
                    ecol = RED if threatened else GOLD
                    pulse = int(2 * math.sin(t_val * 0.005))
                    for angle in range(0, 360, 72):
                        rad = math.radians(angle + t_val * 0.05)
                        ex2 = cx2 + int(math.cos(rad) * (7 + pulse))
                        ey2 = cy2 + int(math.sin(rad) * (7 + pulse))
                        pygame.draw.line(self.screen, ecol, (cx2, cy2), (ex2, ey2), 2)
                    pygame.draw.circle(self.screen, ecol, (cx2, cy2), 4)
                    pygame.draw.rect(self.screen, ecol, (rx, ry, ts, ts), 2)
                    if threatened:
                        wa  = int(128 + 127 * math.sin(t_val * 0.01))
                        ws  = pygame.Surface((ts + 8, ts + 8), pygame.SRCALPHA)
                        pygame.draw.rect(ws, (255, 0, 0, wa), (0, 0, ts + 8, ts + 8), 2)
                        self.screen.blit(ws, (rx - 4, ry - 4))

        # Trail particles
        for tp in self.trail_parts:
            tp.draw(self.screen, ox, oy)
        # Particles
        for pt in self.particles:
            pt.draw(self.screen, ox, oy)
        # Explosions
        for ex in self.explosions:
            ex.draw(self.screen, ox, oy)
        # Spawn animations
        for sa in self.spawn_anims:
            sa.draw(self.screen, ox, oy)
        # Enemies
        for e in self.enemies:
            if e.active:
                e.draw(self.screen, ox, oy, self)
        # Player
        if self.player and self.player.active:
            self.player.draw(self.screen, ox, oy, self)
        # Bullets
        for b in self.bullets:
            if b.active:
                b.draw(self.screen, ox, oy)
        # Forest overlay (hides tanks beneath)
        for fx, fy in forest_tiles:
            rx  = ox + fx * TILE_SIZE
            ry  = oy + fy * TILE_SIZE
            ts  = TILE_SIZE
            sway = math.sin(t_val * 0.004 + fx * 0.8) * 1.5
            pygame.draw.rect(self.screen, (16, 72, 18), (rx, ry, ts, ts))
            for fdx in range(3):
                for fdy in range(3):
                    if (fdx + fdy) % 2 == 0:
                        cx2 = rx + 4 + fdx * 7 + int(sway)
                        cy2 = ry + 4 + fdy * 7
                        pygame.draw.circle(self.screen, (30, 130, 35), (cx2, cy2), 4)
                        pygame.draw.circle(self.screen, (50, 165, 52), (cx2 - 1, cy2 - 1), 2)
        # Score popups
        for sp in self.score_popups:
            sp.draw(self.screen, ox, oy, self.F_SM)

        # ── Top bar ───────────────────────────────────────────
        pygame.draw.rect(self.screen, (16, 18, 26), (0, 0, SCREEN_W, 28))
        pygame.draw.line(self.screen, (55, 60, 80), (0, 28), (SCREEN_W, 28), 1)
        fps_s = self.F_XS.render(
            f"FPS:{int(self.clock.get_fps())}",
            True, GREEN if self.clock.get_fps() > 50 else ORANGE)
        self.screen.blit(fps_s, (TILE_SIZE * GRID_W - 60, 9))

        self._draw_hud()

    # ── HUD ───────────────────────────────────────────────────
    def _draw_hud(self):
        hx = self.hud_x
        pygame.draw.rect(self.screen, (12, 14, 20), (hx - 2, 0, HUD_WIDTH + 4, SCREEN_H))
        pygame.draw.line(self.screen, (55, 60, 80), (hx - 2, 0), (hx - 2, SCREEN_H), 2)

        yc = [32]

        def txt(s, col=WHITE, f=None):
            f = f or self.F_XS
            surf = f.render(s, True, col)
            self.screen.blit(surf, (hx + 8, yc[0]))
            yc[0] += surf.get_height() + 3

        def sep():
            yc[0] += 4
            pygame.draw.line(self.screen, (40, 44, 60),
                             (hx + 4, yc[0]), (hx + HUD_WIDTH - 6, yc[0]), 1)
            yc[0] += 7

        def bar(val, mx, col, h=7):
            bw     = HUD_WIDTH - 16
            filled = max(0, int(bw * val / max(mx, 1)))
            pygame.draw.rect(self.screen, (35, 0, 0),  (hx + 8, yc[0], bw, h))
            pygame.draw.rect(self.screen, col,          (hx + 8, yc[0], filled, h))
            pygame.draw.rect(self.screen, (80, 80, 80), (hx + 8, yc[0], bw, h), 1)
            yc[0] += h + 5

        def header(s, col):
            w    = HUD_WIDTH - 16
            surf = self.F_XS.render(s, True, col)
            bh   = surf.get_height() + 4
            pygame.draw.rect(self.screen,
                             (col[0] // 6, col[1] // 6, col[2] // 6),
                             (hx + 6, yc[0], w, bh))
            pygame.draw.rect(self.screen, col, (hx + 6, yc[0], w, bh), 1)
            self.screen.blit(surf,
                             (hx + (HUD_WIDTH - surf.get_width()) // 2, yc[0] + 2))
            yc[0] += bh + 5

        header("◆ PLAYER ◆", CYAN)
        p = self.player
        if p:
            txt(f"Score : {p.score:,}", GOLD)
            txt(f"Lives : {'♥ ' * min(p.lives, 10)}", (220, 60, 80))
            txt("HP :", WHITE)
            bar(p.hp, p.max_hp, GREEN)
            if p.invincible > 0:
                txt(f"SHIELD {int(p.invincible / 180 * 100)}%", (100, 180, 255))
                bar(p.invincible, 180, (60, 140, 255))
            if p.combo > 1 and p.combo_t > 0:
                txt(f"COMBO ×{p.combo}", GOLD, self.F_SM)

        sep()
        header("◆ ENEMIES ◆", RED)
        if self.lvlmgr:
            txt(f"Active  : {len(self.enemies)}", ORANGE)
            txt(f"Queue   : {len(self.lvlmgr.pool)}", YELLOW)
            txt(f"Killed  : {self.lvlmgr.killed} / {self.lvlmgr.total}", GREEN)
            bar(self.lvlmgr.killed, self.lvlmgr.total, GREEN)

        sep()
        header("◆ AI AGENTS ◆", CYAN)
        cnts = {}
        for e in self.enemies:
            tn = type(e).__name__
            cnts[tn] = cnts.get(tn, 0) + 1
        for tn, (sn, algo, col) in [
            ('BasicTank',  ('Basic', 'BFS',      (130, 140, 145))),
            ('FastTank',   ('Fast',  'Greedy',    CYAN)),
            ('ArmorTank',  ('Armor', 'A*',        ORANGE)),
            ('PowerTank',  ('Power', 'A*+Util',   PURPLE)),
            ('BossTank',   ('Boss',  'Minimax',   RED)),
        ]:
            if tn in cnts:
                pygame.draw.rect(self.screen, col, (hx + 8, yc[0] + 3, 8, 8), border_radius=2)
                s = self.F_XS.render(f" {sn}×{cnts[tn]}  [{algo}]", True, col)
                self.screen.blit(s, (hx + 18, yc[0]))
                yc[0] += s.get_height() + 3

        # Boss stats
        boss = next((e for e in self.enemies
                     if isinstance(e, BossTank) and e.active), None)
        if boss:
            sep()
            header("◆ BOSS STATS ◆", RED)
            pc   = {1: 'Aggressive', 2: 'Balanced', 3: 'Desperate'}
            pcol = {1: ORANGE, 2: (220, 100, 20), 3: PINK}
            txt(f"Phase : {boss.phase} — {pc[boss.phase]}", pcol[boss.phase])
            txt("HP:", WHITE)
            bar(boss.hp, boss.max_hp, RED)
            txt(f"Depth   : {boss.last_depth}", YELLOW)
            txt(f"Nodes   : {boss.nodes_total}", CYAN)
            txt(f"Pruned  : {boss.nodes_pruned}", GREEN)
            if boss.nodes_total > 0:
                unpruned  = 5 ** boss.last_depth
                speedup   = unpruned / max(1, boss.nodes_total)
                txt(f"Speedup : ×{speedup:.1f}", LIME)
            txt("α-β Pruning: ON", GREEN)

        sep()
        # Minimap
        header("◆ MINIMAP ◆", GRAY)
        mw, mh = HUD_WIDTH - 16, 80
        mx, my = hx + 8, yc[0]
        tw, th = mw / GRID_W, mh / GRID_H
        ms = pygame.Surface((mw, mh))
        ms.fill((8, 10, 14))
        tile_cols = {BRICK: (158, 62, 18), STEEL: (78, 92, 108),
                     WATER: (20, 60, 160),  FOREST: (20, 80, 22),
                     EAGLE: (180, 150, 20)}
        if self.grid:
            for gy in range(GRID_H):
                for gx in range(GRID_W):
                    t = self.grid[gy][gx]
                    if t in tile_cols:
                        pygame.draw.rect(ms, tile_cols[t],
                                         (int(gx * tw), int(gy * th),
                                          max(1, int(tw)), max(1, int(th))))
            for e in self.enemies:
                if e.active:
                    ec = {BossTank: RED, ArmorTank: ORANGE,
                          PowerTank: PURPLE, FastTank: CYAN}.get(type(e), GRAY)
                    pygame.draw.rect(ms, ec,
                                     (int(e.x * tw), int(e.y * th),
                                      max(2, int(tw)), max(2, int(th))))
            if p and p.active:
                pygame.draw.rect(ms, GREEN,
                                 (int(p.x * tw), int(p.y * th),
                                  max(2, int(tw)), max(2, int(th))))
        pygame.draw.rect(ms, GRAY, (0, 0, mw, mh), 1)
        self.screen.blit(ms, (mx, my))
        yc[0] += mh + 8

        sep()
        header("◆ CONTROLS ◆", WHITE)
        for line, col in [("WASD / ↑↓←→  Move",  LIGHT_GRAY),
                          ("SPACE        Shoot",  LIGHT_GRAY),
                          ("N            Next Lvl", LIGHT_GRAY),
                          ("M            Mute",   LIGHT_GRAY),
                          ("ESC          Menu",   LIGHT_GRAY)]:
            if yc[0] + 14 > SCREEN_H - 4:
                break
            txt(line, col)

        sep()
        header("◆ LEGEND ◆", WHITE)
        legend = [
            (BROWN,      "Brick (destroyable)"),
            (SILVER,     "Steel (solid)"),
            (BLUE,       "Water (impassable)"),
            (DARK_GREEN, "Forest (camouflage)"),
            (GOLD,       "Eagle (protect!)"),
            (GREEN,      "You"),
            (GRAY,       "Basic — BFS"),
            (CYAN,       "Fast — Greedy"),
            (ORANGE,     "Armor — A*"),
            (PURPLE,     "Power — Utility"),
            (RED,        "Boss — Minimax"),
        ]
        for col, lbl in legend:
            if yc[0] + 14 > SCREEN_H - 4:
                break
            pygame.draw.rect(self.screen, col, (hx + 8, yc[0] + 2, 9, 9), border_radius=2)
            pygame.draw.rect(self.screen, WHITE, (hx + 8, yc[0] + 2, 9, 9), 1, border_radius=2)
            self.screen.blit(self.F_XS.render(lbl, True, LIGHT_GRAY), (hx + 21, yc[0]))
            yc[0] += 14

    # ── Overlay (level complete / game over / win) ─────────────
    def _draw_overlay(self, text, color):
        ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 160))
        self.screen.blit(ov, (0, 0))
        gx = TILE_SIZE * GRID_W // 2
        gy = SCREEN_H // 2

        sh = self.F_LG.render(text, True, (0, 0, 0))
        self.screen.blit(sh, sh.get_rect(center=(gx + 3, gy - 46)))
        s  = self.F_LG.render(text, True, color)
        self.screen.blit(s,  s.get_rect(center=(gx, gy - 48)))

        if self.player:
            sc = self.F_MD.render(f"Score : {self.player.score:,}", True, GOLD)
            self.screen.blit(sc, sc.get_rect(center=(gx, gy)))
            lv = self.F_SM.render(f"Lives remaining : {self.player.lives}", True, WHITE)
            self.screen.blit(lv, lv.get_rect(center=(gx, gy + 32)))

        pr = self.F_SM.render("ENTER / SPACE → Menu", True, LIGHT_GRAY)
        self.screen.blit(pr, pr.get_rect(center=(gx, gy + 72)))

        if self.state == ST_LVLDONE and self.level < 3:
            nx_s = self.F_SM.render(
                f"N  →  Level {self.level + 1}: "
                f"{LevelManager.CFGS[self.level + 1]['name']}",
                True, GREEN)
            self.screen.blit(nx_s, nx_s.get_rect(center=(gx, gy + 100)))

        if self.state == ST_BOSSDEAD:
            full = self.F_MD.render("ALL LEVELS COMPLETE — YOU WIN!", True, GOLD)
            self.screen.blit(full, full.get_rect(center=(gx, gy + 110)))

    # ── Main loop ─────────────────────────────────────────────
    def run(self):
        while True:
            self._events()
            self._update()
            self._draw()
            self.clock.tick(FPS)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    Game().run()