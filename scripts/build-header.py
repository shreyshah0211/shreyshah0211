"""Renders assets/moon-{light,dark}.gif: a small cratered moon turning once per
loop, lit from the side, Bayer-dithered to 1-bit pixels, with a single pixel
satellite in orbit. No text. Loops seamlessly.

    python3 scripts/build-header.py      # needs numpy + pillow
"""
import numpy as np
from PIL import Image
from pathlib import Path

L, PX = 96, 2                  # 96x96 logical pixels, each drawn 2x2
FRAMES, MS = 96, 90
R = 0.62                       # moon radius in [-1, 1] space
rng = np.random.default_rng(11)

BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 + 1 / 32
THR = np.tile(BAYER, (L // 4, L // 4))

u = (np.arange(L) + 0.5) / L * 2 - 1
X, Y = np.meshgrid(u, -u)
Z2 = R * R - X * X - Y * Y
DISC = Z2 > 0
Z = np.sqrt(np.clip(Z2, 0, None))
N = np.stack([X, Y, Z], -1) / R                     # unit normals in view space

# craters: random spots on the sphere, a dark bowl with a bright rim
K = 34
C = rng.normal(size=(K, 3)); C /= np.linalg.norm(C, axis=1, keepdims=True)
CR = rng.uniform(0.08, 0.32, K) ** 1.3
# broad "maria": low-frequency dark patches
M = rng.normal(size=(5, 3)); M /= np.linalg.norm(M, axis=1, keepdims=True)

LIGHT = np.array([-0.55, 0.35, 0.76]); LIGHT /= np.linalg.norm(LIGHT)
TILT = np.radians(18)


def albedo(p):
    a = np.full(p.shape[:-1], 0.82)
    for m in M:
        a -= 0.3 * np.clip((p @ m - 0.5) / 0.25, 0, 1)
    for c, r in zip(C, CR):
        ang = np.arccos(np.clip(p @ c, -1, 1))
        q = ang / r
        a -= 0.45 * np.clip(1 - q, 0, 1) ** 0.5 * (q < 1)
        a += 0.25 * np.exp(-((q - 1.0) / 0.1) ** 2)
    return np.clip(a, 0, 1)


def frame(i):
    th = 2 * np.pi * i / FRAMES
    ct, st = np.cos(TILT), np.sin(TILT)
    tilt = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    spin = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    body = N @ (tilt @ spin)                          # view normal -> surface point on the moon
    lum = albedo(body) * (0.08 + 0.92 * np.clip(N @ LIGHT, 0, 1) ** 0.7)
    lum = np.where(DISC, np.clip(lum * 1.15, 0, 0.97), 0)
    ink = (lum > THR) & DISC
    # one-pixel satellite on a tilted orbit, behind the moon for half the loop
    a = 2 * np.pi * i / FRAMES * 2
    sx, sy, sz = 0.88 * np.cos(a), 0.2 * np.sin(a), np.sin(a)
    ix, iy = int((sx + 1) / 2 * L), int((1 - sy) / 2 * L)
    hidden = sz < 0 and (sx * sx + sy * sy) < R * R
    if not hidden and 0 <= ix < L and 0 <= iy < L:
        ink[iy:iy + 2, ix:ix + 2] = True
    return ink


def save(path, fg):
    frames = []
    for i in range(FRAMES):
        im = Image.fromarray(frame(i).astype(np.uint8)).convert('P')
        im = im.resize((L * PX, L * PX), Image.NEAREST)
        im.putpalette([0, 0, 0] + list(fg) + [0] * 762)
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=MS,
                   loop=0, transparency=0, disposal=2, optimize=False)


out = Path(__file__).resolve().parent.parent / 'assets'
out.mkdir(exist_ok=True)
save(out / 'moon-light.gif', (0x19, 0x17, 0x14))   # ink on light mode
save(out / 'moon-dark.gif', (0xF2, 0xF2, 0xF2))    # off-white on dark mode, like shreys.me
print('wrote assets/moon-{light,dark}.gif')
