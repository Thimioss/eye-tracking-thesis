import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.backends.backend_template import FigureCanvas
from scipy.stats import kde
from scipy.stats import stats


def generate_heat_map(x, y, c_v):
    nbins = 500

    # fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
    #
    # axes[0, 0].set_title('Scatterplot')
    # axes[0, 0].plot(x, y, 'ko')
    #
    # axes[0, 1].set_title('Hexbin plot')
    # axes[0, 1].hexbin(x, y, gridsize=nbins)
    #
    # axes[1, 0].set_title('2D Histogram')
    # axes[1, 0].hist2d(x, y, bins=nbins)

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(np.vstack([x, y]), bw_method=0.1)
    xi, yi = np.mgrid[0:c_v.window[2]:nbins * 1j, 0:c_v.window[3]:nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # axes[1, 1].set_title('Gaussian KDE')
    # axes[1, 1].pcolormesh(xi, yi, zi.reshape(xi.shape))
    #
    # axes[1, 1].set_ylim(axes[1, 1].get_ylim()[::-1])
    # axes[1, 1].xaxis.tick_top()
    # axes[1, 1].yaxis.set_ticks(np.arange(0, 16, 1))
    # axes[1, 1].yaxis.tick_left()

    f = np.arctan(c_v.window[2] / c_v.window[3])
    h_cm = np.cos(f) * c_v.screen_diagonal_in_cm
    pixel_size_cm = h_cm / c_v.window[3]
    pixel_size_inch = pixel_size_cm/2.54
    w = c_v.window[2]*pixel_size_inch
    h = c_v.window[3]*pixel_size_inch
    dpi = 1/pixel_size_inch

    cmap = plt.get_cmap("viridis")

    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()  # get the axis
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.7, cmap=cmap)
    # pcolormesh or contourf

    im = plt.imread('bg_image.jpg')
    ax.imshow(im, extent=[0, c_v.window[2], 0, c_v.window[3]], aspect='auto')

    fig.savefig(c_v.last_file_name+'_heatmap.png', dpi=dpi)

    plt.show()

    image = cv2.imread(c_v.last_file_name+'_heatmap.png')
    return image
