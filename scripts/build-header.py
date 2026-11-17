"""Renders assets/star-{light,dark}.gif: a 1-bit Bayer-dithered beveled star
that blinks, with a soft dithered glow. Pure NumPy + Pillow.

    python3 scripts/build-header.py      # needs numpy + pillow
"""
import numpy as np
from PIL import Image
from pathlib import Path

SIZE, FRAMES, MS = 200, 40, 60
OUTER, INNER = 0.78, 0.33          # star tip radius and notch radius

BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 + 1 / 32
THR = np.tile(BAYER, (SIZE // 4 + 1, SIZE // 4 + 1))[:SIZE, :SIZE]

u = np.linspace(-1, 1, SIZE)
X, Y = np.meshgrid(u, -u)
R = np.hypot(X, Y)
TH = np.arctan2(X, Y) % (2 * np.pi)          # 0 = straight up

# Star edge radius at each angle: a polygon through 5 tips and 5 notches.
seg = 2 * np.pi / 10
k = np.floor(TH / seg).astype(int)
f = TH / seg - k
a0, a1 = k * seg, (k + 1) * seg
r0 = np.where(k % 2 == 0, OUTER, INNER)
r1 = np.where(k % 2 == 0, INNER, OUTER)
p0 = np.stack([r0 * np.sin(a0), r0 * np.cos(a0)], -1)
p1 = np.stack([r1 * np.sin(a1), r1 * np.cos(a1)], -1)
d = p1 - p0
# where the ray from the center at angle TH crosses segment p0-p1
dirv = np.stack([np.sin(TH), np.cos(TH)], -1)
cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
EDGE = cross(p0, d) / cross(dirv, d)
INSIDE = R <= EDGE

# Bevel: each of the 10 wedges is a facet, lit from the upper left.
# Facets on the left of a tip catch the light; facets on the right fall in shade.
mid = (k + 0.5) * seg
light_dir = np.radians(315)
FACET = 0.55 + 0.45 * np.cos(mid - light_dir) * np.where(k % 2 == 0, 1.0, 0.6)
FACET = np.clip(FACET + 0.25 * (1 - R / np.maximum(EDGE, 1e-6)), 0, 1)   # brighter toward the center


def frame(i):
    t = i / FRAMES
    # Mostly steady, then a quick bright blink and a gentle dip after it.
    blink = np.exp(-((t - 0.5) / 0.06) ** 2)
    level = 0.62 + 0.38 * blink - 0.12 * np.exp(-((t - 0.68) / 0.07) ** 2)
    scale = 0.9 + 0.1 * blink
    rr = R / scale
    inside = rr <= EDGE
    star = np.where(inside, np.clip(FACET * (0.55 + 0.6 * level), 0, 0.97), 0)
    glow = (0.06 + 0.55 * blink) * np.exp(-((rr - 0.25) / 0.38) ** 2) * (~inside)
    # thin sparkle rays on the blink
    rays = blink * 0.9 * (np.exp(-(X / 0.012) ** 2) + np.exp(-(Y / 0.012) ** 2)) * np.exp(-R / 0.5) * (~inside)
    lum = np.clip(star + glow + rays, 0, 0.97)
    return lum > THR


def save(path, fg):
    frames = []
    for i in range(FRAMES):
        im = Image.fromarray(frame(i).astype(np.uint8)).convert('P')
        im.putpalette([0, 0, 0] + list(fg) + [0] * 762)
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=MS,
                   loop=0, transparency=0, disposal=2, optimize=False)


out = Path(__file__).resolve().parent.parent / 'assets'
out.mkdir(exist_ok=True)
save(out / 'star-light.gif', (0x1E, 0x3F, 0xD8))   # cobalt on light
save(out / 'star-dark.gif', (0x5B, 0x78, 0xFF))    # brighter cobalt on dark
print('wrote assets/star-{light,dark}.gif')
