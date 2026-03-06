import numpy as np
import PIL
import PIL.ImageOps
from PIL import Image


def imadjust(x, a, b, c, d, gamma=1.0):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


def poisson_gaussian_noise(x, severity):
    # severity: 0..4
    c_poisson = 10 * [60, 25, 12, 5, 3][severity]
    x = np.array(x) / 255.0
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255.0

    c_gauss = 0.1 * [0.08, 0.12, 0.18, 0.26, 0.38][severity]
    x = x / 255.0
    x = np.clip(x + np.random.normal(size=x.shape, scale=c_gauss), 0, 1) * 255.0
    return Image.fromarray(np.uint8(x))


def low_light(x, severity):
    # x: HxWx3 uint8/float
    c = [0.60, 0.50, 0.40, 0.30, 0.20][severity - 1]
    x = np.array(x) / 255.0
    x_scaled = imadjust(x, float(x.min()), float(x.max()), 0, c, gamma=2.0) * 255.0
    x_scaled = poisson_gaussian_noise(x_scaled, severity=severity - 1)
    return x_scaled


def color_quant(x: Image.Image, severity: int):
    bits = 5 - severity + 1
    return PIL.ImageOps.posterize(x, bits)


def iso_noise(x, severity: int):
    # x: HxWx3 uint8/float
    c_poisson = 25
    x = np.array(x) / 255.0
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255.0

    c_gauss = 0.7 * [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    x = x / 255.0
    x = np.clip(x + np.random.normal(size=x.shape, scale=c_gauss), 0, 1) * 255.0
    return Image.fromarray(np.uint8(x))


def glass_blur_cv2(x, severity: int):
    """Fallback glass blur implementation without skimage.

    x: HxWx3 uint8
    severity: 1..5
    """
    x = np.array(x).astype(np.uint8)
    h, w = x.shape[:2]

    # parameters roughly increase with severity
    # (sigma, max_delta, iterations)
    params = [
        (0.7, 1, 1),
        (0.9, 2, 1),
        (1.1, 2, 2),
        (1.3, 3, 2),
        (1.5, 4, 3),
    ]
    sigma, max_delta, iters = params[severity - 1]

    # local pixel shuffling
    out = x.copy()
    for _ in range(iters):
        for _ in range(2000):
            y = np.random.randint(0, h)
            x0 = np.random.randint(0, w)
            dy = np.random.randint(-max_delta, max_delta + 1)
            dx = np.random.randint(-max_delta, max_delta + 1)
            y2 = np.clip(y + dy, 0, h - 1)
            x2 = np.clip(x0 + dx, 0, w - 1)
            out[y, x0], out[y2, x2] = out[y2, x2].copy(), out[y, x0].copy()

    # gaussian blur (use cv2 if available, otherwise numpy fallback)
    try:
        import cv2

        k = int(2 * round(3 * sigma) + 1)
        out = cv2.GaussianBlur(out, (k, k), sigmaX=sigma, sigmaY=sigma)
    except Exception:
        # simple fallback: average blur
        k = 3
        pad = k // 2
        padded = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        out2 = np.zeros_like(out)
        for yy in range(h):
            for xx in range(w):
                out2[yy, xx] = padded[yy:yy+k, xx:xx+k].mean(axis=(0, 1))
        out = out2

    return Image.fromarray(out)
