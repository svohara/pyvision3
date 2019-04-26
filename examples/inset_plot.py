"""
Demonstration of creating a plot using matplotlib
and using it as an overlay on a pyvision image.
"""

import pyvision3 as pv3
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("You must have matplotlib installed to run this example.")
    raise


def _fake_data(k):
    bar_chart_vals = 3*np.random.rand(k) + 8
    line_plot_vals = np.random.randint(8000, 9300, k)
    return [bar_chart_vals, line_plot_vals]


def _plot_data(dat):
    bar_vals, line_vals = dat
    x_vals = np.arange(len(bar_vals))

    w, h = plt.figaspect(0.125)
    fig = plt.figure(figsize=(8, 1), dpi=100)  # 800x100 pixel plot
    plt.bar(x_vals, bar_vals)
    ax1 = plt.gca()
    ax1.set_ylabel('mice')
    ax1.set_ylim([5, 15])

    ax1.tick_params('y', colors='b')
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax2 = ax1.twinx()
    ax2.plot(x_vals, line_vals, 'r-')
    ax2.set_ylabel('purring')
    ax2.set_ylim([7000, 9500])
    ax2.tick_params('y', colors='r')

    return fig


def _get_inset_image(k):
    dat = _fake_data(k)
    fig = _plot_data(dat)
    img = pv3.matplot_fig_to_image(fig)
    return img


def main():
    img = pv3.Image(pv3.IMG_SLEEPYCAT)
    overlay = _get_inset_image(20)

    x_pos = 5
    y_pos = img.size[1]-overlay.size[1]-5

    img.annotate_inset_image(overlay, (x_pos, y_pos), size=None)
    img.show(annotations_opacity=0.8)


if __name__ == '__main__':
    print("====================================================")
    print("With image window in focus, hit spacebar to quit")
    print("====================================================")
    main()
