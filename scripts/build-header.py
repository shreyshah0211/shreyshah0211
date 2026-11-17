"""Renders assets/gear-{light,dark}.gif: a 1-bit Bayer-dithered gear spinning on
a turntable. Raymarched in NumPy, no 3D software needed.

    python3 scripts/build-header.py      # needs numpy + pillow
"""
import numpy as np
from PIL import Image
from pathlib import Path

SIZE, FRAMES, MS = 200, 48, 70
TEETH, R0, TOOTH, THICK = 10, 0.6, 0.12, 0.2
TILT = np.radians(12)

BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 + 1 / 32


def sdf(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]  # gear lies in the x-y plane
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    prof = R0 + TOOTH * np.clip(np.cos(TEETH * th) * 2.2, -1, 1)
    d2 = (r - prof) * 0.22  # teeth are steep; keep the march conservative
    d2 = np.maximum(d2, 0.16 - r)                       # axle hole
    for k in range(5):                                  # lightening holes
        a = 2 * np.pi * k / 5
        d2 = np.maximum(d2, 0.11 - np.hypot(x - 0.37 * np.cos(a), y - 0.37 * np.sin(a)))
    wz = np.abs(z) - THICK
    return np.minimum(np.maximum(d2, wz), 0) + np.hypot(np.maximum(d2, 0), np.maximum(wz, 0))


def rot(a, b):
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    spin = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])     # turntable (around up)
    tilt = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])     # lean toward camera
    stand = np.eye(3)                                          # gear faces the camera at angle 0
    return tilt @ spin @ stand


def frame(angle):
    u = np.linspace(-1.05, 1.05, SIZE)
    X, Y = np.meshgrid(u, -u)
    ro = np.stack([X, Y, np.full_like(X, -3.0)], -1)
    rd = np.array([0, 0, 1.0])
    M = rot(angle, TILT).T                                 # world -> gear space
    t = np.zeros(X.shape)
    hit = np.zeros(X.shape, bool)
    for _ in range(220):
        p = (ro + t[..., None] * rd) @ M.T
        d = sdf(p)
        hit |= d < 1e-3
        t = np.where(hit, t, t + np.maximum(d, 2e-4))
    p = (ro + t[..., None] * rd) @ M.T
    e = 1e-3
    n = np.stack([sdf(p + v) - sdf(p - v) for v in np.eye(3) * e], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    n = n @ M                                               # gear -> world
    light = np.array([-0.5, 0.7, -0.6]); light /= np.linalg.norm(light)
    key = np.clip((n @ light + 0.35) / 1.35, 0, 1)          # wrap light, so no face goes flat
    head = np.clip(-n[..., 2], 0, 1)                        # a little camera light
    lum = 0.12 + 0.6 * key + 0.28 * head
    lum = np.where(hit & (t < 6), lum, 0)
    thr = np.tile(BAYER, (SIZE // 4 + 1, SIZE // 4 + 1))[:SIZE, :SIZE]
    ink = (lum > thr) & hit
    return ink


def save(path, fg):
    frames = []
    for i in range(FRAMES):
        ink = frame(2 * np.pi * i / FRAMES)
        im = Image.fromarray(ink.astype(np.uint8)).convert('P')
        im.putpalette([0, 0, 0] + list(fg) + [0] * 762)
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=MS,
                   loop=0, transparency=0, disposal=2, optimize=False)


out = Path(__file__).resolve().parent.parent / 'assets'
out.mkdir(exist_ok=True)
save(out / 'gear-light.gif', (0x1E, 0x3F, 0xD8))   # cobalt on light
save(out / 'gear-dark.gif', (0x5B, 0x78, 0xFF))    # brighter cobalt on dark
print('wrote assets/gear-{light,dark}.gif')
