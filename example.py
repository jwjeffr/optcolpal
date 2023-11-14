import numpy as np
import matplotlib.pyplot as plt
import optcolpal


class PowerFunction:

    def __init__(self, power):
        self.power = power

    def __call__(self, x):
        return x ** self.power


def main():

    """
    plot f(x) = x^n for multiple values of n
    """

    # define x to plot
    step = 1
    x_array = np.arange(1, 8 + step, step)
    powers = range(6)

    # define functions to plot
    functions = [PowerFunction(power=n) for n in powers]
    labels = [f'$x^{{{n}}}$' for n in powers]

    # some plot options
    plot_options = {'edgecolor': 'black', 'zorder': 5}

    # palette options
    palette_options = {
        'num_colors': 6, # only required argument
        'max_opt_time_sec': 60, # kill optimization if it takes longer than 60 seconds
        'seed': 13, # seed for initial guess
        'l_bounds': (0, 100), # "brightness" bounds
        'a_bounds': (-127, 128), # "greenness" to "redness" bounds
        'b_bounds': (-127, 128) # "blueness" to "yellowness" bounds
    }

    palette = optcolpal.generate_palette(**palette_options)

    # plot using colors from the palette
    plt.figure()
    for function, color, label in zip(functions, palette, labels):
        plt.scatter(x_array, function(x_array), facecolor=color, label=label, **plot_options)

    # plot options
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.yscale('log')
    plt.legend(title=r'$f(x)$', ncol=2)
    plt.grid()
    plt.savefig('example_plot.png', dpi=800, bbox_inches='tight')

    # plot the palette
    palette.plot(save_file='example_palette.png', dpi=800, bbox_inches='tight')


if __name__ == '__main__':

    main()
