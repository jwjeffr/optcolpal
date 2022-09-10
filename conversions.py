import numpy as np


def rgb2lab(rgb):

    r, g, b = rgb

    if r > 0.04045:
        r = ((r + 0.055) / 1.055) ** 2.4
    else:
        r /= 12.92

    if g > 0.04045:
        g = ((g + 0.055) / 1.055) ** 2.4
    else:
        g /= 12.92

    if b > 0.04045:
        b = ((b + 0.055) / 1.055) ** 2.4
    else:
        b /= 12.92

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    if x > 0.008856:
        x = x ** (1 / 3)
    else:
        x = (7.787 * x) + 16 / 116

    if y > 0.008856:
        y = y ** (1 / 3)
    else:
        y = (7.787 * y) + 16 / 116

    if z > 0.008856:
        z = z ** (1 / 3)
    else:
        z = (7.787 * z) + 16 / 116

    return (
        (116 * y) - 16,
        500 * (x - y),
        200 * (y - z)
    )


def lab2rgb(lab):

    L, a, b = lab

    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200

    if x ** 3 > 0.008856:
        x = 0.95047 * x ** 3
    else:
        x = 0.95047 * (x - 16 / 116) / 7.787

    if y ** 3 > 0.008856:
        y = 0.95047 * y ** 3
    else:
        y = 0.95047 * (y - 16 / 116) / 7.787

    if z ** 3 > 0.008856:
        z = 0.95047 * z ** 3
    else:
        z = 0.95047 * (z - 16 / 116) / 7.787

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    if r > 0.0031308:
        r = 1.055 * r ** (1 / 2.4) - 0.055
    else:
        r *= 12.92

    if g > 0.0031308:
        g = 1.055 * g ** (1 / 2.4) - 0.055
    else:
        g *= 12.92

    if b > 0.0031308:
        b = 1.055 * b ** (1 / 2.4) - 0.055
    else:
        b *= 12.92

    return (
        max(0, min(1, r)),
        max(0, min(1, g)),
        max(0, min(1, b))
    )


def delta_E(labA, labB):

    deltaL = labA[0] - labB[0]

    deltaA = labA[1] - labB[1]

    deltaB = labA[2] - labB[2]

    c1 = np.sqrt(labA[1] * labA[1] + labA[2] * labA[2])

    c2 = np.sqrt(labB[1] * labB[1] + labB[2] * labB[2])

    deltaC = c1 - c2

    deltaH = deltaA * deltaA + deltaB * deltaB - deltaC * deltaC
    if deltaH < 0:
        deltaH = 0
    else:
        deltaH = np.sqrt(deltaH)

    sc = 1.0 + 0.045 * c1

    sh = 1.0 + 0.015 * c1

    deltaLKlsl = deltaL / (1.0)

    deltaCkcsc = deltaC / (sc)

    deltaHkhsh = deltaH / (sh)

    i = deltaLKlsl * deltaLKlsl + deltaCkcsc * deltaCkcsc + deltaHkhsh * deltaHkhsh;

    if i < 0:
        return 0
    else:
        return np.sqrt(i)