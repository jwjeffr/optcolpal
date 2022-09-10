import numpy as np
from itertools import combinations
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Union
from conversions import rgb2lab, lab2rgb, delta_E


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

    colors: Union[np.ndarray, list[RGBColor], list[LabColor]]

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

    def plot(self, grid_size):

        fig, axs = plt.subplots(*grid_size)
        axs = axs.flatten()

        for ax, color in zip(axs, self.colors):
            code = color.code.reshape(
                (1, 1, 3)
            )
            ax.imshow(code)
            rounded_code = np.array([
                round(c, 2) for c in color.code
            ])
            ax.set_title(
                rounded_code,
                fontdict=dict(size=10)
            )
            ax.set_xticks([])
            ax.set_yticks([])

        if len(self.colors) < len(axs):
            axs[-1].remove()

        fig.tight_layout()
        fig.show()


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


def main():

    num_colors = 5
    objective = ObjectiveFunc(
        metric_function=objective_func,
        num_colors=num_colors
    )

    colors = [
        RGBColor(*color) for color in np.random.uniform(low=0.0, high=1.0, size=(num_colors, 3))
    ]

    arr = np.array([
        color.to_lab().code for color in colors
    ])

    arr = arr.flatten()

    bounds = [(0, 100), (-80, 80), (-80, 80)] * num_colors

    minimization = minimize(
        objective,
        x0=arr,
        method='nelder-mead',
        bounds=bounds
    )

    optimized_colors = minimization.x.reshape(num_colors, 3)

    palette = Palette(optimized_colors)

    palette.plot(grid_size=(int(np.ceil(num_colors / 2)), 2))


if __name__ == '__main__':
    main()
