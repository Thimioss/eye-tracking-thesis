import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.backends.backend_template import FigureCanvas
from scipy.stats import kde
from scipy.stats import stats


def show_smoothing_effect(x, y, c_v):
    a = []
    b = []
    for i in range(len(x)):
        temp_x = 0
        temp_y = 0
        count = 0
        for j in range(i, i - 10, -1):
            if j > 0:
                count += 1
                temp_x += x[j]
                temp_y += y[j]
        if count > 0:
            a.append(int(temp_x / count))
            b.append(int(temp_y / count))

    fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=True)

    axes[0].set_title('Raw data')
    axes[0].scatter(x, y, s=1)

    axes[1].set_title('Data after smoothing')
    axes[1].scatter(a, b, s=1)

    axes[0].set_xlim([0, c_v.window[2]])
    axes[0].set_ylim([0, c_v.window[3]])
    axes[1].set_xlim([0, c_v.window[2]])
    axes[1].set_ylim([0, c_v.window[3]])

    axes[0].set_aspect(1)
    axes[1].set_aspect(1)

    plt.show()


def show_scatter_plot(x, y, c_v):
    nbins = 500
    fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=True)

    axes[0].set_title('Scatterplot')
    axes[0].scatter(x, y, s=1)

    k = kde.gaussian_kde(np.vstack([x, y]), bw_method=0.2)
    xi, yi = np.mgrid[0:c_v.window[2]:nbins * 1j, 0:c_v.window[3]:nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    cmap = plt.get_cmap('jet')
    axes[1].set_title('Gaussian KDE')
    axes[1].pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.7, norm=colors.PowerNorm(gamma=0.2), cmap=cmap, shading='auto')

    axes[0].set_xlim([0, c_v.window[2]])
    axes[0].set_ylim([0, c_v.window[3]])
    axes[1].set_xlim([0, c_v.window[2]])
    axes[1].set_ylim([0, c_v.window[3]])

    axes[0].set_aspect(1)
    axes[1].set_aspect(1)

    plt.show()


def generate_heat_map(x, y, c_v):
    nbins = 500

    show_scatter_plot(x, y, c_v)
    show_smoothing_effect(x, y, c_v)

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(np.vstack([x, y]), bw_method=0.2)
    xi, yi = np.mgrid[0:c_v.window[2]:nbins * 1j, 0:c_v.window[3]:nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    f = np.arctan(c_v.window[2] / c_v.window[3])
    h_cm = np.cos(f) * c_v.screen_diagonal_in_cm
    pixel_size_cm = h_cm / c_v.window[3]
    pixel_size_inch = pixel_size_cm / 2.54
    w = c_v.window[2] * pixel_size_inch
    h = c_v.window[3] * pixel_size_inch
    dpi = 1 / pixel_size_inch

    cmap = plt.get_cmap('jet')

    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()  # get the axis
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.7, norm=colors.PowerNorm(gamma=0.2), cmap=cmap, shading='auto')
    # pcolormesh or contourf

    im = plt.imread('bg_image.jpg')
    ax.imshow(im, extent=[0, c_v.window[2], 0, c_v.window[3]], aspect='auto')

    fig.savefig(c_v.last_file_name + '_heatmap.png', dpi=dpi)

    plt.show()

    image = cv2.imread(c_v.last_file_name + '_heatmap.png')
    return image


def rgb_white2alpha(rgb, ensure_increasing=True):
    """
    Convert a set of RGB colors to RGBA with maximum transparency.

    The transparency is maximised for each color individually, assuming
    that the background is white.

    Parameters
    ----------
    rgb : array_like shaped (N, 3)
        Original colors.
    ensure_increasing : bool, default=False
        Ensure that alpha values are strictly increasing.

    Returns
    -------
    rgba : numpy.ndarray shaped (N, 4)
        Colors with maximum possible transparency, assuming a white
        background.
    """
    # The most transparent alpha we can use is given by the min of RGB
    # Convert it from saturation to opacity
    alpha = 1. - np.min(rgb, axis=1)
    if ensure_increasing:
        # Let's also ensure the alpha value is monotonically increasing
        a_max = alpha[0]
        for i, a in enumerate(alpha):
            alpha[i] = a_max = np.maximum(a, a_max)
    alpha = np.expand_dims(alpha, -1)
    # Rescale colors to discount the white that will show through from transparency
    rgb = (rgb + alpha - 1) / alpha
    # Concatenate our alpha channel
    return np.concatenate((rgb, alpha), axis=1)


def cmap_white2alpha(name, ensure_increasing=True, register=False):
    """
    Convert colormap to have the most transparency possible, assuming white background.

    Parameters
    ----------
    name : str
        Name of builtin (or registered) colormap.
    ensure_increasing : bool, default=False
        Ensure that alpha values are strictly increasing.
    register : bool, default=True
        Whether to register the new colormap.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        Colormap with alpha set as low as possible.
    """
    # Fetch the cmap callable
    cmap = plt.get_cmap(name)
    # Get the colors out from the colormap LUT
    rgb = cmap(np.arange(cmap.N))[:, :3]  # N-by-3
    # Convert white to alpha
    rgba = rgb_white2alpha(rgb, ensure_increasing=ensure_increasing)
    # Create a new Colormap object
    cmap_alpha = colors.ListedColormap(rgba, name=name + "_alpha")
    if register:
        cm.register_cmap(name=name + "_alpha", cmap=cmap_alpha)
    return cmap_alpha
