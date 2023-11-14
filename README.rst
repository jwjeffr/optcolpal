.. role:: raw-math(raw)
    :format: latex html
.. _LAB: https://en.wikipedia.org/wiki/CIELAB_color_space

Optimized Color Palette (optcolpal)
-----------------------------------

This library generates a palette of an arbitrary amount of distinct colors.

This works by minimizing the summed Gaussian pairwise `LAB`_ distance between each color in the palette:

:raw-math:`$$c_1, c_2, \cdots, c_n = \underset{c_1, c_2, \cdots, c_n}{\arg\min} \sum_{j=1}^N \sum_{i=1}^{j-1} e^{-L(c_i, c_j)^2}$$`

