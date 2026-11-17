"""Renders assets/sky-{light,dark}.gif: a small cratered moon turning once per
loop, lit from the side, Bayer-dithered to 1-bit pixels. A one-pixel satellite
orbits with a short trail, a few pixel stars flare into crosses, and one shooting
star crosses per loop. No text. Loops seamlessly.

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
    return ink


# ---- the sky around the moon ---------------------------------------------------
CW, CH = 208, 112                       # full canvas in logical pixels
OX, OY = (CW - L) // 2, (CH - L) // 2   # where the moon sits
MCX, MCY, MR = OX + L / 2, OY + L / 2, R * L / 2

STARS = []
while len(STARS) < 26:
    x, y = int(rng.integers(2, CW - 2)), int(rng.integers(2, CH - 2))
    if np.hypot(x - MCX, y - MCY) > MR + 22:
        STARS.append((x, y, int(rng.integers(0, FRAMES)), rng.random() < 0.15))


def sat_pos(i):
    a = 2 * np.pi * i / FRAMES * 2
    sx, sy, sz = 0.88 * np.cos(a), 0.2 * np.sin(a), np.sin(a)
    x, y = OX + (sx + 1) / 2 * L, OY + (1 - sy) / 2 * L
    behind = sz < 0 and np.hypot(x - MCX, y - MCY) < MR + 1
    return int(x), int(y), behind


def sky(i):
    c = np.zeros((CH, CW), bool)
    c[OY:OY + L, OX:OX + L] = frame(i)
    moon = c.copy()

    # stars: single pixels; once a loop each one dims, then flares into a cross
    for x, y, t0, big in STARS:
        t = (i - t0) % FRAMES
        if t in (0, 1):
            continue
        c[y, x] = True
        if big:
            c[y, x + 1] = True
        if 2 <= t <= 5:
            c[y - 1, x] = c[y + 1, x] = c[y, x - 1] = c[y, x + 1] = True
            if t in (3, 4):
                c[y - 2, x] = c[y + 2, x] = c[y, x - 2] = c[y, x + 2] = True

    # satellite with a short fading trail; it shows as a cutout when crossing the lit moon
    for back, keep in ((6, 2), (4, 1), (2, 1)):        # older points drawn sparser
        x, y, behind = sat_pos(i - back)
        if not behind and (i - back) % keep == 0 and not moon[y, x]:
            c[y, x] = True
    x, y, behind = sat_pos(i)
    if not behind:
        if moon[y:y + 2, x:x + 2].any():
            c[y - 1:y + 3, x - 1:x + 3] = False
            c[y:y + 2, x:x + 2] = True
            c[y - 1, x - 1] = c[y - 1, x + 2] = c[y + 2, x - 1] = c[y + 2, x + 2] = False
        else:
            c[y:y + 2, x:x + 2] = True

    # one shooting star per loop, top left, gone in under a second
    t = i - 60
    if 0 <= t < 10:
        hx, hy = 12 + t * 7, 6 + t * 2.8
        for k in range(16):
            if k > 2 and (k + t) % 2:
                continue                                  # tail breaks up as it fades
            px, py = hx - k * 2.5, hy - k
            if k < 16 - t and 0 <= px < CW and 0 <= py < CH:
                c[int(py), int(px)] = True
    return c


def save(path, fg):
    frames = []
    for i in range(FRAMES):
        im = Image.fromarray(sky(i).astype(np.uint8)).convert('P')
        im = im.resize((CW * PX, CH * PX), Image.NEAREST)
        im.putpalette([0, 0, 0] + list(fg) + [0] * 762)
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=MS,
                   loop=0, transparency=0, disposal=2, optimize=False)


out = Path(__file__).resolve().parent.parent / 'assets'
out.mkdir(exist_ok=True)
save(out / 'sky-light.gif', (0x19, 0x17, 0x14))   # ink on light mode
save(out / 'sky-dark.gif', (0xF2, 0xF2, 0xF2))    # off-white on dark mode, like shreys.me
print('wrote assets/sky-{light,dark}.gif')
