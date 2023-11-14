import numpy as np
from itertools import combinations
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time


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


class OptimizationLengthError(Exception):

    pass


@dataclass
class RGBColor:

    r: float
    g: float
    b: float

    @property
    def code(self):
        return np.array([
            self.r, self.g, self.b
        ])

    def to_lab(self):

        lab_code = rgb2lab(self.code)

        return LabColor(*lab_code)

    def plot(self):

        code = self.code.reshape(
            (1, 1, 3)
        )
        plt.imshow(code)
        plt.title(self.code)
        plt.xticks([])
        plt.yticks([])
        plt.show()


@dataclass
class LabColor:

    L: float
    a: float
    b: float

    @property
    def code(self):
        return np.array([
            self.L, self.a, self.b
        ])

    def to_rgb(self):

        rgb_code = lab2rgb(self.code)

        return RGBColor(*rgb_code)

    def plot(self):

        code = self.to_rgb().code.reshape(
            (1, 1, 3)
        )
        plt.imshow(code)
        plt.xticks([])
        plt.yticks([])
        plt.show()


@dataclass
class Palette:

    colors: np.ndarray

    def __post_init__(self):

        if type(self.colors) is list:

            rgb_list = all(
                [type(color) is RGBColor for color in self.colors]
            )

            lab_list = all(
                [type(color) is LabColor for color in self.colors]
            )

            if not (rgb_list or lab_list):
                raise TypeError

            return

        elif type(self.colors) is np.ndarray:

            if 3 not in self.colors.shape:
                raise ValueError

            if self.colors.shape[1] != 3:
                self.colors = self.colors.T

            lst = []
            for color in self.colors:
                rgb = LabColor(*color).to_rgb()
                lst.append(rgb)

            self.colors = lst

    def __iter__(self):
        lst = [(c.r, c.b, c.g) for c in self.colors]
        return iter(lst)

    def plot(self, grid_size=None, save_file=None, show=False, **save_kwargs):

        if grid_size is None:
            grid_size = (int(np.ceil(len(self.colors) / 2)), 2)

        fig, axs = plt.subplots(*grid_size)
        axs = axs.flatten()

        for ax, color in zip(axs, self.colors):
            code = color.code.reshape(
                (1, 1, 3)
            )
            ax.imshow(code)
            rounded_code = np.array(str([
                round(c, 2) for c in color.code
            ]))
            ax.set_title(
                rounded_code,
                fontdict=dict(size=10)
            )
            ax.set_xticks([])
            ax.set_yticks([])

        if len(self.colors) < len(axs):
            axs[-1].remove()

        fig.tight_layout()
        if show:
            fig.show()

        if save_file is not None:
            fig.savefig(save_file, **save_kwargs)


@dataclass
class ObjectiveFunc:

    metric_function: callable
    num_colors: int

    def __call__(self, color_arr):

        color_arr = color_arr.reshape((self.num_colors, 3))

        all_distances = [
            self.metric_function(*combo) for combo in combinations(color_arr, r=2)
        ]

        return sum(all_distances)


def objective_func(first_lab, second_lab):

    return np.exp(-delta_E(first_lab, second_lab) ** 2)


@dataclass
class OptimizationKiller:

    max_time: float

    def __post_init__(self):
        self.initial = time.time()

    def __call__(self, x):
        elapsed = time.time() - self.initial

        if self.max_time <= elapsed:
            raise OptimizationLengthError('max time elapsed in optimization')


def generate_palette(
        num_colors=None,
        max_opt_time_sec=None,
        seed=None,
        l_bounds=(0, 100),
        a_bounds=(-128, 127),
        b_bounds=(-128, 127)
):

    if num_colors is None:
        raise ValueError('num_colors needed')

    if seed is not None:
        np.random.seed(seed)

    objective = ObjectiveFunc(
        metric_function=objective_func,
        num_colors=num_colors
    )

    arr = np.array([
        [
            np.random.uniform(*l_bounds),
            np.random.uniform(*a_bounds),
            np.random.uniform(*b_bounds)
        ] for _ in range(num_colors)
    ]).flatten()

    bounds = [l_bounds, a_bounds, b_bounds] * num_colors

    minimization_options = {
        'fun': objective,
        'x0': arr,
        'method': 'nelder-mead',
        'bounds': bounds
    }
    if max_opt_time_sec is not None:
        minimization_options['callback'] = OptimizationKiller(max_time=max_opt_time_sec)

    minimization = minimize(**minimization_options)
    optimized_colors = minimization.x.reshape(num_colors, 3)

    return Palette(optimized_colors)
