import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from scipy.interpolate import interp1d

def apply_theme(path):
    plt.style.use(path)
    return

def add_colorbar(ax, mappable, width=0.15, pad=0.1, loc='right', mode='share'):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    
    fig = ax.figure
    if mode == 'share':
        last_axes = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(loc, size=width, pad=pad)
        plt.sca(last_axes)
    else:    
        pos = ax.get_position()
        pad /= 2.
        width /= 2.
        if loc in ['right', 'left']:
            cbar_width = width * pos.width
            cbar_height = pos.height
            if loc == 'right':
                cbar_x = pos.x1 + pad * pos.width
            else:
                cbar_x = pos.x0 - pad * pos.width - cbar_width
            cbar_y = pos.y0
        elif loc in ['top', 'bottom']:
            cbar_height = width * pos.height
            cbar_width = pos.width
            if loc == 'top':
                cbar_y = pos.y1 + pad * pos.height
            else:
                cbar_y = pos.y0 - pad * pos.height - cbar_height
            cbar_x = pos.x0
        cax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])

    cbar = fig.colorbar(mappable, cax=cax, location=loc)
        
    return cbar


def enumerate_plots(fig, axes=None, labels=None):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if axes is None:
        axes = fig.axes

    for i, ax in enumerate(axes):
        ylabel_bbox = ax.yaxis.label.get_tightbbox(renderer)
        pixel_coords = ylabel_bbox.bounds
        centers = (pixel_coords[0] + pixel_coords[2] / 2, pixel_coords[1] + pixel_coords[3] / 2)
        axes_coords = ax.transAxes.inverted().transform(centers)
        if not labels:
            ax.text(axes_coords[0], 1, s=r'{(' + chr(97+i) + ')}', va='top', ha='center', transform=ax.transAxes)
        else:
            ax.text(axes_coords[0], 1, s=labels[i], va='top', ha='center', transform=ax.transAxes)
    return fig

def zoom_axis(ax, bounds, xlim=None, ylim=None, xticklabels=[], yticklabels=[], lw=1, lc="k", ls='-', alpha=1.,
              ticks=False, remove_lines=True):

    axins = ax.inset_axes(bounds,
                          xlim=xlim,
                          ylim=ylim,
                          xticklabels=xticklabels,
                          yticklabels=yticklabels,
                          transform=ax.transAxes)

    for spine in axins.spines.values():
        spine.set_color(lc)
        spine.set_linewidth(lw)
        spine.set_linestyle(ls)

    rect, lines = ax.indicate_inset_zoom(axins, linewidth=lw, edgecolor=lc, alpha=alpha, linestyle=ls)
    for line in lines:
        line.set_color(lc)
        line.set_alpha(alpha)
        if remove_lines:
            line.set_linewidth(0)

    if not ticks:
        axins.tick_params(axis='both', which='both', left=False, right=False, top=False, bottom=False)

    return axins


def cmap_line(ax, x, y, values, cmap='viridis', lw=1, vmax=None, vmin=None):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(vmin, vmax)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(values)
    lc.set_linewidth(lw)
    line = ax.add_collection(lc)
    lc.set_antialiased(False)
    return ax, line, lc

def latex_sci_formatter(decimals=3, skip=1, tight_spacing=True):
    def formatter(x, pos):
        if skip > 1 and int(pos) % skip != 0:
            return ""
        if x == 0:
            return "0"
        exponent = int(np.floor(np.log10(abs(x))))
        coeff = x / 10**exponent
        if tight_spacing:
            return rf"${coeff:.{decimals}f}\! \cdot \! 10^{{{exponent}}}$"
        else:
            return rf"${coeff:.{decimals}f} \cdot 10^{{{exponent}}}$"
    return formatter

def parula():
    from matplotlib.colors import LinearSegmentedColormap
    # Parula RGB data sampled from MATLAB
    parula_colors = np.array([
    [0.2081, 0.1663, 0.5292],
    [0.2116, 0.1898, 0.5777],
    [0.2123, 0.2138, 0.6270],
    [0.2081, 0.2386, 0.6771],
    [0.1959, 0.2645, 0.7279],
    [0.1707, 0.2919, 0.7792],
    [0.1253, 0.3242, 0.8303],
    [0.0591, 0.3598, 0.8683],
    [0.0117, 0.3875, 0.8820],
    [0.0060, 0.4086, 0.8828],
    [0.0165, 0.4266, 0.8786],
    [0.0329, 0.4430, 0.8720],
    [0.0498, 0.4586, 0.8641],
    [0.0629, 0.4737, 0.8554],
    [0.0723, 0.4887, 0.8467],
    [0.0779, 0.5040, 0.8384],
    [0.0793, 0.5200, 0.8312],
    [0.0749, 0.5375, 0.8263],
    [0.0641, 0.5570, 0.8240],
    [0.0488, 0.5772, 0.8228],
    [0.0343, 0.5966, 0.8199],
    [0.0265, 0.6137, 0.8135],
    [0.0239, 0.6287, 0.8038],
    [0.0231, 0.6418, 0.7913],
    [0.0228, 0.6535, 0.7768],
    [0.0267, 0.6642, 0.7607],
    [0.0384, 0.6743, 0.7436],
    [0.0590, 0.6838, 0.7254],
    [0.0843, 0.6928, 0.7062],
    [0.1133, 0.7015, 0.6859],
    [0.1453, 0.7098, 0.6646],
    [0.1801, 0.7177, 0.6424],
    [0.2178, 0.7250, 0.6193],
    [0.2586, 0.7317, 0.5954],
    [0.3022, 0.7376, 0.5712],
    [0.3482, 0.7424, 0.5473],
    [0.3953, 0.7459, 0.5244],
    [0.4420, 0.7481, 0.5033],
    [0.4871, 0.7491, 0.4840],
    [0.5300, 0.7491, 0.4661],
    [0.5709, 0.7485, 0.4494],
    [0.6099, 0.7473, 0.4337],
    [0.6473, 0.7456, 0.4188],
    [0.6834, 0.7435, 0.4044],
    [0.7184, 0.7411, 0.3905],
    [0.7525, 0.7384, 0.3768],
    [0.7858, 0.7356, 0.3633],
    [0.8185, 0.7327, 0.3498],
    [0.8507, 0.7299, 0.3360],
    [0.8824, 0.7274, 0.3217],
    [0.9139, 0.7258, 0.3063],
    [0.9450, 0.7261, 0.2886],
    [0.9739, 0.7314, 0.2666],
    [0.9938, 0.7455, 0.2403],
    [0.9990, 0.7653, 0.2164],
    [0.9955, 0.7861, 0.1967],
    [0.9880, 0.8066, 0.1794],
    [0.9789, 0.8271, 0.1633],
    [0.9697, 0.8481, 0.1475],
    [0.9626, 0.8705, 0.1309],
    [0.9589, 0.8949, 0.1132],
    [0.9598, 0.9218, 0.0948],
    [0.9661, 0.9514, 0.0755],
    [0.9763, 0.9831, 0.0538],
    ])  # Shape: (N, 3)

    return LinearSegmentedColormap.from_list("parula", parula_colors)

def truncate_colormap(cmap, min_val=0.2, max_val=1.0, n=256):
    import matplotlib.colors as mcolors
    new_colors = cmap(np.linspace(min_val, max_val, n))
    new_cmap = mcolors.LinearSegmentedColormap.from_list(f'trunc({cmap.name},{min_val},{max_val})', new_colors)
    return new_cmap

def twin_bottom(ax, offset=-0.15, color='k'):
    twin = ax.twiny()
    twin.xaxis.set_ticks_position("bottom")
    twin.xaxis.set_label_position("bottom")

    # Offset the twin axis below the host
    twin.spines["bottom"].set_position(("axes", offset))

    # Turn on the frame for the twin axis, but then hide all
    # but the bottom spine
    twin.set_frame_on(True)
    twin.patch.set_visible(False)
    for sp in twin.spines.values():
        sp.set_visible(False)
    twin.spines["bottom"].set_visible(True)
    twin.spines["bottom"].set_color(color)
    twin.tick_params(axis='x', colors=color)
    return twin

def get_colors(i=None):
    from matplotlib import rcParams
    colors = rcParams['axes.prop_cycle'].by_key()['color']
    if i is None:
        return colors
    else:
        return colors[i]

def move_ticks_labels(axes, x=None, y=None):

    for ax in axes:
        if y in ("left", "right"):
            ax.yaxis.set_label_position(y)
            if y == "left":
                ax.yaxis.tick_left()
            else:
                ax.yaxis.tick_right()

        if x in ("top", "bottom"):
            ax.xaxis.set_label_position(x)
            if x == "bottom":
                ax.xaxis.tick_bottom()
            else:
                ax.xaxis.tick_top()
                
                