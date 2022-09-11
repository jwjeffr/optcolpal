# Optimized Color Palette (optcolpal)

This piece of code generates a palette of an arbitrary amount of distinct colors.

This works by minimizing the summed Gaussian pairwise [LAB](https://en.wikipedia.org/wiki/CIELAB_color_space) distance between each color in the palette:

<p align="center">
  <img src="https://raw.githubusercontent.com/jwjeffr/optcolpal/main/lab.png">
</p>
