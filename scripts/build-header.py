"""Renders assets/intro.gif: the shreys.me intro as a pixel-art banner.
Square pixel stars drift and twinkle into crosses while a terminal types
"hi, i'm shrey." then "i think deeply about things.", then the stars scatter
and the loop starts over. Everything sits on a 3px grid.

    python3 scripts/build-header.py      # needs numpy + pillow
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

W, H, PX = 280, 72, 3                # logical pixels; each one is drawn 3x3
MS = 70                              # ms per frame
BG, INK = (14, 14, 14), (242, 242, 242)
FONT = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 10)
PHRASES = ["hi, i'm shrey.", "i think deeply about things."]
rng = np.random.default_rng(7)

# ---- timeline: list of (text, cursor_on, phase) per frame -------------------
tl = []
def hold(text, n, phase='idle'):
    for _ in range(n):
        tl.append((text, phase))
hold('', 8)
for ch in range(1, len(PHRASES[0]) + 1): hold(PHRASES[0][:ch], 1)
hold(PHRASES[0], 16)
for ch in range(len(PHRASES[0]) - 1, -1, -2): hold(PHRASES[0][:max(ch, 0)], 1)
hold('', 5)
for ch in range(1, len(PHRASES[1]) + 1): hold(PHRASES[1][:ch], 1)
hold(PHRASES[1], 22)
EXPLODE_AT = len(tl)
hold('', 14, 'explode')
hold('', 6, 'dark')
N = len(tl)

# ---- stars --------------------------------------------------------------------
S = 70
sx, sy = rng.uniform(0, W, S), rng.uniform(0, H, S)
ang = rng.uniform(0, 2 * np.pi, S)
spd = rng.uniform(0.03, 0.12, S)
vx, vy = np.cos(ang) * spd, np.sin(ang) * spd
big = rng.random(S) < 0.2
cross = rng.random(S) < 0.08
alpha = rng.uniform(0.12, 0.45, S)
tw_at = rng.integers(0, N, S)                        # each star twinkles once a loop
tw_on = rng.random(S) < 0.35

cx, cy = W / 2, H / 2

def shade(a):
    return tuple(int(BG[i] + (INK[i] - BG[i]) * a) for i in range(3))

frames = []
for f, (text, phase) in enumerate(tl):
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    d.fontmode = '1'                                 # hard pixel edges, no antialiasing
    fade_in = min(1, f / 10)
    for i in range(S):
        x, y = sx[i] + vx[i] * f, sy[i] + vy[i] * f
        a = alpha[i] * fade_in
        if phase in ('explode', 'dark'):
            k = f - EXPLODE_AT
            dx, dy = x - cx, y - cy
            dist = np.hypot(dx, dy) or 1
            boost = 2.5 + 3 * (i % 5) / 4
            x += dx / dist * boost * k
            y += dy / dist * boost * k
            a = max(0, alpha[i] - 0.05 * k)
        x, y = x % W if phase == 'idle' else x, y % H if phase == 'idle' else y
        if a <= 0 or not (0 <= x < W and 0 <= y < H):
            continue
        # twinkle: brighten in steps, flare a cross at the peak
        t = (f - tw_at[i]) % N
        flare = tw_on[i] and t < 8 and phase == 'idle'
        if flare:
            a = min(1, a + [0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2][t])
        ix, iy = int(x), int(y)
        col = shade(a)
        if cross[i] or (flare and 2 <= t <= 5):
            d.point([(ix, iy), (ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)], fill=col)
        elif big[i]:
            d.rectangle([ix, iy, ix + 1, iy + 1], fill=col)
        else:
            d.point((ix, iy), fill=col)

    # terminal line with a block cursor
    full = text if phase == 'idle' else ''
    tw = int(d.textlength(full, font=FONT))
    total = int(d.textlength(PHRASES[1], font=FONT))
    x0 = (W - tw - 5) // 2
    ty = H // 2 - 6
    if phase == 'idle':
        d.text((x0, ty), full, font=FONT, fill=INK)
        if (f // 7) % 2 == 0 or text not in PHRASES:   # solid while typing, blinks while holding
            d.rectangle([x0 + tw + 1, ty + 1, x0 + tw + 5, ty + 11], fill=INK)

    frames.append(im.resize((W * PX, H * PX), Image.NEAREST).quantize(colors=24, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))

out = Path(__file__).resolve().parent.parent / 'assets'
out.mkdir(exist_ok=True)
frames[0].save(out / 'intro.gif', save_all=True, append_images=frames[1:], duration=MS, loop=0, optimize=True)
print(f'wrote assets/intro.gif — {N} frames, {N * MS / 1000:.1f}s loop')
