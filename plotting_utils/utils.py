import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba, Colormap
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
from skimage.measure import marching_cubes
from scipy.ndimage import map_coordinates
from matplotlib.tri import Triangulation
from scipy.ndimage import gaussian_filter

def apply_theme(path):
    plt.style.use(path)
    return

def init_theme(theme):
    if theme=='classic':
        apply_theme(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oldschool.mplstyle'))
    else:
        apply_theme(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ub.mplstyle'))

apply_theme(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ub.mplstyle'))

def add_colorbar2(ax, mappable,
                  width=0.05,
                  length=1.0,
                  pad=0.05,
                  offset=0.0,
                  loc='right',
                  mode='new',
                  norm=None,
                  rect=True,
                  nticks=5,
                  **kwargs):

    if loc not in ['right', 'left', 'top', 'bottom']:
        raise ValueError('Invalid location')

    fig = ax.figure
    fig.canvas.draw()
    pos = ax.get_position()
    x0, y0, x1, y1 = pos.x0, pos.y0, pos.x1, pos.y1
    h, w = pos.height, pos.width

    if norm is None:
        norm = min(h, w)
    elif norm == 'width':
        norm = w
    elif norm == 'height':
        norm = h
    else:
        raise ValueError('Invalid norm')

    width = width * norm
    pad = pad * norm
    #offset= offset*norm

    if mode == 'new':
        if loc == 'right':
            cbar_pos = [x1+pad, y0 + h*(1-length+offset)/2, width, length*h]
        elif loc == 'left':
            cbar_pos = [x0 - pad - width, y0 + h*(1-length+offset)/2, width, length*h]
        elif loc == 'top':
            cbar_pos = [x0 + w*(1-length+offset)/2, y1 + pad, length*w, width]
        elif loc == 'bottom':
            cbar_pos = [x0 + w*(1-length+offset)/2, y0 - pad - width, length*w, width]
        ax_pos = [x0, y0, w, h]

    elif mode == 'share':
        if loc == 'right':
            cbar_pos = [x1 -width, y0 + h*(1-length+offset)/2, width, length*h]
            ax_pos = [x0, y0, w - width - pad, h]
        elif loc == 'left':
            cbar_pos = [x0, y0 + h*(1 - length+offset)/2, width, length*h]
            ax_pos = [x0 + width + pad, y0, w - width - pad, h]
        elif loc == 'top':
            cbar_pos = [x0+w*(1-length+offset)/2, y1 - width, length*w, width]
            ax_pos = [x0, y0, w, h - width - pad]
        elif loc == 'bottom':
            cbar_pos = [x0 + w * (1 - length + offset) / 2, y0 - width - pad, length * w, width]
            ax_pos = [x0, y0+width, w, h - width]

    cax = fig.add_axes(cbar_pos)
    ax.set_position(ax_pos)

    cax.grid(False)
    cax.set_rasterized(True)
    cbar = fig.colorbar(mappable, cax=cax,  extendrect=rect, location=loc, **kwargs)

    vmin, vmax = mappable.get_clim()
    cbar.set_ticks(np.linspace(vmin, vmax, nticks))
    return cbar




def enumerate_plots(fig, axes=None, labels=None, anchor='ylabel', counter=0, offset=(0,0), va=None, ha=None):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if axes is None:
        axes = fig.axes

    for ax in axes:
        if ax.title._text == '':
            ax.set_title(" ")

    for i, ax in enumerate(axes):

        if anchor=='title':
            _bbox = ax.title.get_tightbbox(renderer)
            pixel_coords = _bbox.bounds
            centers = (pixel_coords[0] + pixel_coords[2] / 2, pixel_coords[1] + pixel_coords[3] / 2)
            axes_coords = ax.transAxes.inverted().transform(centers)
            x, y = 0.+offset[0], axes_coords[1]+offset[1]
            if not va: va = 'center'
            if not ha: ha = 'left'

        else:
            _bbox = ax.yaxis.label.get_tightbbox(renderer)
            pixel_coords = _bbox.bounds
            centers = (pixel_coords[0] + pixel_coords[2] / 2, pixel_coords[1] + pixel_coords[3] / 2)
            axes_coords = ax.transAxes.inverted().transform(centers)
            x, y = axes_coords[0] + offset[0], 1. + offset[1]
            if not va: va = 'top'
            if not ha: ha = 'center'
        if not labels:
            ax.text(x, y, s=r'{(' + chr(97+i+counter) + ')}', va=va, ha=ha, transform=ax.transAxes)
        else:
            ax.text(x, y, s=labels[i], va=va, ha=ha, transform=ax.transAxes)
    return fig



def zoom_axis(ax, bounds, bounds_coords="axes", xlim=None, ylim=None, xticklabels=[], yticklabels=[], lw=1, lc="k", ls='-', alpha=1., box=True, ticks=False, remove_lines=True, style_spines=False, zorder=10):


    if bounds_coords == "data":
        transform = ax.transData
    else:
        transform = ax.transAxes

    axins = ax.inset_axes(bounds,
                          xlim=xlim,
                          ylim=ylim,
                          xticklabels=xticklabels,
                          yticklabels=yticklabels,
                          transform=transform,
                          zorder=zorder)
    
    for spine in axins.spines.values():
        spine.set_zorder(zorder)
        if style_spines: 
            spine.set_color(lc)
            spine.set_linewidth(lw)
            spine.set_linestyle(ls)
            

    rect, lines = ax.indicate_inset_zoom(axins, linewidth=lw, edgecolor=lc, alpha=alpha, linestyle=ls, zorder=zorder)
    #rect.set_linestyle(ls)
    for line in lines:

        line.set_color(lc)
        line.set_alpha(alpha)
        if remove_lines:
            line.set_alpha(0)
    
    if not box:
        rect.set_edgecolor('none')

    if not ticks:
        axins.tick_params(axis='both', which='both', left=False, right=False, top=False, bottom=False)

    return axins


def colored_line(ax, x, y, values, cmap='viridis', lw=1, vmax=None, vmin=None, lognorm=False):
    default_kwargs = {"capstyle": "butt"}
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    values = np.asarray(values).ravel()
    # --- norm ---
    if lognorm:
        if vmin is None:
            vmin = np.nanmin(values[values > 0])
        if vmax is None:
            vmax = np.nanmax(values)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(vmin, vmax)

    lc = LineCollection(segments, norm=norm, cmap=cmap, **default_kwargs)
    lc.set_array(values)          # now correct length (N-1)
    lc.set_linewidth(lw)
    line = ax.add_collection(lc)
    lc.set_antialiased(False)

    return ax, line, lc

def latex_sci_formatter(decimals=3, skip=1, tight_spacing=True, sign=False):
    def formatter(x, pos):
        if skip > 1 and int(pos) % skip != 0:
            return ""
        if x == 0:
            return "0"
        exponent = int(np.floor(np.log10(abs(x))))
        coeff = x / 10**exponent

        sgn = ""
        if sign:
            if x > 0:
                sgn = "+"
            if x == 0.:
                sgn = r"\pm"

        if tight_spacing:
            return rf"${sgn}{coeff:.{decimals}f}\! \cdot \! 10^{{{exponent}}}$"
        else:
            return rf"${sgn}{coeff:.{decimals}f} \cdot 10^{{{exponent}}}$"
    return formatter

def float_formatter(decimals=3, skip=1, sign=False, end=""):
    def formatter(x, pos):
        if skip > 1 and int(pos) % skip != 0:
            return ""
        sgn = ""
        if sign:
            if x > 0:
                sgn = "+"
            if x == 0.:
                sgn = r"\pm"
            
        return rf"${sgn}{x:.{decimals}f}$" + end
    return formatter

def latex_sci_number_formatter(x, decimals=3, tight_spacing=True, multiply=r'\cdot'):
    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exponent
    if tight_spacing:
        return rf"${coeff:.{decimals}f}\! {multiply} \! 10^{{{exponent}}}$"
    else:
        return rf"${coeff:.{decimals}f} {multiply} 10^{{{exponent}}}$"
        

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

def move_ticks_labels(axes, x="bottom", y="right"):

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

    
def truncate_colormap(cmap, min_val=0.2, max_val=1.0, n=256):
    import matplotlib.colors as mcolors
    new_colors = cmap(np.linspace(min_val, max_val, n))
    new_cmap = mcolors.LinearSegmentedColormap.from_list(f'trunc({cmap.name},{min_val},{max_val})', new_colors)
    return new_cmap
  

def dummy_legend(ax, labels, colors=None, linestyles=None, linewidths=None, title=None, loc="best", **legend_kwargs):
    from matplotlib.lines import Line2D
    n = len(labels)
    if colors is None:
        colors = [None] * n
    if linestyles is None:
        linestyles = ["-"] * n
    if linewidths is None:
        linewidths = [1] * n
    text_only = all(c is None for c in colors)
    handles = [Line2D([], [], linestyle="none", linewidth=0, label=label) if color is None else Line2D([0], [0], color=color, linestyle=linestyle, linewidth=linewidth, label=label) for label, color, linestyle, linewidth in zip(labels, colors, linestyles, linewidths)]
    if text_only:
        legend_kwargs.setdefault("handlelength", 0)
        legend_kwargs.setdefault("handletextpad", 0)
    return ax.legend(handles=handles, title=title, loc=loc, **legend_kwargs)

def get_colormap(name):
    if 'parula' in name:
        colors = np.array([
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
        ])
    elif 'white' in name:
        colors = np.array([
        [0.        , 0.447     , 0.741     ],
        [0.03225806, 0.46483871, 0.74935484],
        [0.06451613, 0.48267742, 0.75770968],
        [0.09677419, 0.50051613, 0.76606452],
        [0.12903226, 0.51835484, 0.77441935],
        [0.16129032, 0.53619355, 0.78277419],
        [0.19354839, 0.55403226, 0.79112903],
        [0.22580645, 0.57187097, 0.79948387],
        [0.25806452, 0.58970968, 0.80783871],
        [0.29032258, 0.60754839, 0.81619355],
        [0.32258065, 0.6253871 , 0.82454839],
        [0.35483871, 0.64322581, 0.83290323],
        [0.38709677, 0.66106452, 0.84125806],
        [0.41935484, 0.67890323, 0.8496129 ],
        [0.4516129 , 0.69674194, 0.85796774],
        [0.48387097, 0.71458065, 0.86632258],
        [0.51612903, 0.73241935, 0.87467742],
        [0.5483871 , 0.75025806, 0.88303226],
        [0.58064516, 0.76809677, 0.8913871 ],
        [0.61290323, 0.78593548, 0.89974194],
        [0.64516129, 0.80377419, 0.90809677],
        [0.67741935, 0.8216129 , 0.91645161],
        [0.70967742, 0.83945161, 0.92480645],
        [0.74193548, 0.85729032, 0.93316129],
        [0.77419355, 0.87512903, 0.94151613],
        [0.80645161, 0.89296774, 0.94987097],
        [0.83870968, 0.91080645, 0.95822581],
        [0.87096774, 0.92864516, 0.96658065],
        [0.90322581, 0.94648387, 0.97493548],
        [0.93548387, 0.96432258, 0.98329032],
        [0.96774194, 0.98216129, 0.99164516],
        [1.        , 1.        , 1.        ],
        [1.        , 1.        , 1.        ],
        [1.        , 1.        , 1.        ],
        [1.        , 1.        , 1.        ],
        [0.99516129, 0.97822581, 0.97090323],
        [0.99032258, 0.95645161, 0.94180645],
        [0.98548387, 0.93467742, 0.91270968],
        [0.98064516, 0.91290323, 0.8836129 ],
        [0.97580645, 0.89112903, 0.85451613],
        [0.97096774, 0.86935484, 0.82541935],
        [0.96612903, 0.84758065, 0.79632258],
        [0.96129032, 0.82580645, 0.76722581],
        [0.95645161, 0.80403226, 0.73812903],
        [0.9516129 , 0.78225806, 0.70903226],
        [0.94677419, 0.76048387, 0.67993548],
        [0.94193548, 0.73870968, 0.65083871],
        [0.93709677, 0.71693548, 0.62174194],
        [0.93225806, 0.69516129, 0.59264516],
        [0.92741935, 0.6733871 , 0.56354839],
        [0.92258065, 0.6516129 , 0.53445161],
        [0.91774194, 0.62983871, 0.50535484],
        [0.91290323, 0.60806452, 0.47625806],
        [0.90806452, 0.58629032, 0.44716129],
        [0.90322581, 0.56451613, 0.41806452],
        [0.8983871 , 0.54274194, 0.38896774],
        [0.89354839, 0.52096774, 0.35987097],
        [0.88870968, 0.49919355, 0.33077419],
        [0.88387097, 0.47741935, 0.30167742],
        [0.87903226, 0.45564516, 0.27258065],
        [0.87419355, 0.43387097, 0.24348387],
        [0.86935484, 0.41209677, 0.2143871 ],
        [0.86451613, 0.39032258, 0.18529032],
        [0.85967742, 0.36854839, 0.15619355],
        [0.85483871, 0.34677419, 0.12709677],
        [0.85      , 0.325     , 0.098     ]])
    elif 'bwr' in name:
        blue = "#1F4E99"   # muted blue
        red = "#B23A3A"    # muted red
        white = "#FFFFFF"

        # Small extended white region around the center
        colors = [
            (0.00, blue),
            (0.48, white),
            (0.52, white),
            (1.00, red),
        ]

    if '_r' in name:
        colors = [(1 - x, c) for x, c in colors[::-1]]


    return LinearSegmentedColormap.from_list(name, colors, N=256)



def cmapcycler(cmap, n, vmin=0.05, vmax=0.95):
    from cycler import cycler
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(vmin, vmax, n))
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
    return cycler(color=colors)



def merge(pdf_path_left, pdf_path_right, output_path, padding=10):
    from pypdf import PdfReader, PdfWriter, Transformation

    reader_left = PdfReader(pdf_path_left)
    reader_right = PdfReader(pdf_path_right)

    page_left = reader_left.pages[0]
    page_right = reader_right.pages[0]

    w_left, h_left = page_left.mediabox.width, page_left.mediabox.height
    w_right, h_right = page_right.mediabox.width, page_right.mediabox.height

    out_height = max(h_left, h_right)
    out_width = w_left + w_right + padding

    writer = PdfWriter()
    output_page = writer.add_blank_page(
        width=out_width,
        height=out_height
    )

    # --- bottom alignment ---
    y_left = 0
    y_right = 0

    # Left page
    transform_left = Transformation().translate(tx=0, ty=y_left)
    output_page.merge_transformed_page(page_left, transform_left)

    # Right page (shifted by left width + padding)
    transform_right = Transformation().translate(
        tx=w_left + padding,
        ty=y_right
    )
    output_page.merge_transformed_page(page_right, transform_right)

    with open(output_path, "wb") as f:
        writer.write(f)


def figsize(nrows, ncols, init='classic'):
    import matplotlib as mpl

    try: 
        init_theme(init)
    except:
        pass

    mpl.rcParams["figure.subplot.bottom"] = mpl.rcParams["figure.subplot.bottom"] / nrows
    mpl.rcParams["figure.subplot.top"] = 1 - (1-mpl.rcParams["figure.subplot.top"])/ nrows
    
    mpl.rcParams["figure.subplot.left"] = mpl.rcParams["figure.subplot.left"] / ncols
    mpl.rcParams["figure.subplot.right"] = 1 - (1-mpl.rcParams["figure.subplot.right"])/ ncols

    

    rc = mpl.rcParams
    (ax_w, ax_h) = rc['figure.figsize']
    left   = rc["figure.subplot.left"]
    right  = rc["figure.subplot.right"]
    bottom = rc["figure.subplot.bottom"]
    top    = rc["figure.subplot.top"]
    wspace = rc["figure.subplot.wspace"]
    hspace = rc["figure.subplot.hspace"]


    #print(mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"])
    # Total grid size in inches
    grid_w = ncols * ax_w + (ncols - 1) * wspace * ax_w
    grid_h = nrows * ax_h + (nrows - 1) * hspace * ax_h

    # Convert margins from fractions → inches
    fig_w = grid_w / (right - left)
    fig_h = grid_h / (top - bottom)
    return fig_w, fig_h

def box_save(fig, pth, dpi=1000, top=0., bot=0., left=0., right=0.):
    """
    Save a Matplotlib figure after cropping by fractions of figure size.

    top, bot, left, right are fractions from 0 to 1.

    Example:
        left=0.1  cuts 10% of the figure width from the left
        top=0.2   cuts 20% of the figure height from the top
    """
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    bbox = fig.bbox_inches

    width = bbox.width
    height = bbox.height

    new_bbox = Bbox.from_extents(
        bbox.x0 + left * width,
        bbox.y0 + bot * height,
        bbox.x1 - right * width,
        bbox.y1 - top * height
    )

    fig.savefig(pth, dpi=dpi, bbox_inches=new_bbox)
    
    
def set_aspect_ratio(ax, aspect=16/9, anchor="bottom", shift=(0.,0.)):
    """
    Set the physical box aspect ratio of a Matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    aspect : float or tuple or str
        Desired width / height ratio.
        Examples:
            16/9
            (16, 9)
            "16:9"
    anchor : {"center", "bottom", "top"}
        Vertical anchoring when resizing height.
    """
    if isinstance(aspect, str):
        w, h = map(float, aspect.split(":"))
        aspect = w / h
    elif isinstance(aspect, tuple):
        w, h = aspect
        aspect = w / h

    fig = ax.figure
    fig_w, fig_h = fig.get_size_inches()

    p = ax.get_position()

    # p.width is relative to figure width.
    # p.height is relative to figure height.
    new_h = p.width * (fig_w / fig_h) / aspect

    if anchor == "center":
        new_y0 = p.y0 + (p.height - new_h) / 2
    elif anchor == "bottom":
        new_y0 = p.y0
    elif anchor == "top":
        new_y0 = p.y1 - new_h
    else:
        raise ValueError("anchor must be 'center', 'bottom', or 'top'")

    ax.set_position([p.x0+shift[0], new_y0+shift[1], p.width, new_h])




def add_isolevel(ax, x, y, z, q, level=None,
                 view_dir=(0, -1, 0), shading=0.0,
                 colors="lightgrey", cmap="turbo", vmin=None, vmax=None,
                 **kwargs):

    q = np.asarray(q)
    x, y, z = np.unique(np.asarray(x)), np.unique(np.asarray(y)), np.unique(np.asarray(z))

    if level is None:
        level = np.nanpercentile(q, 5)
        print(f"Selected level is {level:.3e}")

    def compute_face_normals(vertices, faces):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        return np.divide(normals,norms, out=np.zeros_like(normals), where=norms!=0)

    # view dir needs to be axis aligned!!!
    view_dir = np.asarray(view_dir, dtype=float)
    view_dir /= np.linalg.norm(view_dir)
    view_axis = np.flatnonzero(view_dir)[0]
    projection_axes = np.delete(np.arange(3), view_axis)

    dx, dy, dz = (x[1] - x[0]), (y[1] - y[0]), (z[1] - z[0])

    # marching cubes in voxel coordinates
    verts, faces, _, _ = marching_cubes(q, level=level, spacing=(dx, dy, dz))
    verts_index = verts / np.array([dx, dy, dz])

    # shift to physical mins
    verts[:, 0] += x.min()
    verts[:, 1] += y.min()
    verts[:, 2] += z.min()
    
    # project triangles onto plane
    polys = [verts[triangle][:, projection_axes] for triangle in faces]

    normals = compute_face_normals(verts, faces)
    ndotv = normals @ view_dir
    ndotv = np.clip(np.abs(ndotv), 0.0, 1.0)
    shade = (1.0 - ndotv) ** 0.5

    colors_array = np.asarray(colors)
    colors_is_field = colors_array.shape == q.shape

    if colors_is_field:
        vertex_values = map_coordinates(colors_array, verts_index.T, order=1, mode="nearest")
        face_values = vertex_values[faces].mean(axis=1)

        if vmin is None: vmin = face_values.min()
        if vmax is None: vmax = face_values.max()
        norm = Normalize(vmin=vmin, vmax=vmax)

        if isinstance(cmap, Colormap):
            cmap_object = cmap
        else:
            try:
                cmap_object = cm.get_cmap(cmap)
            except:
                cmap_object = get_colormap(cmap)

        face_colors = cmap_object(norm(face_values))

    else:
        base_color = np.asarray(to_rgba(colors))
        face_colors = np.tile(base_color, (len(faces), 1))

    if shading != 0:
        shade_factor = 1.0 - shading * shade
        face_colors[:, :3] *= shade_factor[:, None]
    
    pc_kwargs = {"facecolors": face_colors, 
                 "edgecolors": "none", 
                 "linewidths": 0, 
                 "antialiaseds": True, 
                 }
    pc_kwargs.update(kwargs)

    pc = PolyCollection(polys, **pc_kwargs)
    pc.set_rasterized(True)
    ax.add_collection(pc)

    coordinates = (x, y, z)
    ax.set_xlim(coordinates[projection_axes[0]].min(),coordinates[projection_axes[0]].max())
    ax.set_ylim(coordinates[projection_axes[1]].min(),coordinates[projection_axes[1]].max())

    return pc



def add_isolevel_tri(ax, x, y, z, q, level=None,
                 view_dir=(0, -1, 0), shading=0.0, shading_mode="gouraud",
                 colors="lightgrey", cmap="turbo",
                 vmin=None, vmax=None, **kwargs):

    q = np.asarray(q)
    x, y, z = np.unique(np.asarray(x)), np.unique(np.asarray(y)), np.unique(np.asarray(z)),

    if level is None:
        level = np.nanpercentile(q, 5)
        print(f"Selected level is {level:.3e}")

    pc_kwargs = {"antialiased": True, "rasterized": True}
    pc_kwargs.update(kwargs)

    def compute_face_normals(vertices, faces):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        return np.divide(normals,norms, out=np.zeros_like(normals), where=norms!=0)

    view_dir = np.asarray(view_dir, dtype=float)
    view_dir /= np.linalg.norm(view_dir)
    view_axis = np.flatnonzero(view_dir)[0]
    projection_axes = np.delete(np.arange(3), view_axis)

    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]

    verts, faces, vertex_normals, _ = marching_cubes(q, level=level, spacing=(dx, dy, dz))
    verts_index = verts / np.array([dx, dy, dz])
    # Shift to physical coordinates.
    verts[:, 0] += x.min()
    verts[:, 1] += y.min()
    verts[:, 2] += z.min()
    # Project vertices onto the requested plane.
    projected_verts = verts[:, projection_axes]

    face_normals = compute_face_normals(verts, faces)
    # view_dir points from the camera into the scene
    front_facing = face_normals @ view_dir < 0
    faces = faces[front_facing]
    # Draw distant faces first and nearby faces last
    face_depth = verts[faces].mean(axis=1) @ view_dir
    faces = faces[np.argsort(face_depth)[::-1]]
    triangulation = Triangulation(projected_verts[:, 0], projected_verts[:, 1], triangles=faces)

    # Shading
    ndotv = vertex_normals @ view_dir
    ndotv = np.clip(np.abs(ndotv), 0.0, 1.0)
    shade = np.sqrt(1.0 - ndotv)

    colors_array = np.asarray(colors)
    colors_is_field = colors_array.shape == q.shape

    if colors_is_field:
        # Interpolate the color field onto the isosurface vertices.
        vertex_values = map_coordinates(colors_array, verts_index.T, order=1, mode="nearest")
        if vmin is None: vmin = np.nanmin(vertex_values)
        if vmax is None: vmax = np.nanmax(vertex_values)
        norm = Normalize(vmin=vmin, vmax=vmax)

        if isinstance(cmap, Colormap):
            cmap_object = cmap
        else:
            try:
                cmap_object = cm.get_cmap(cmap)
            except:
                cmap_object = get_colormap(cmap)

        if shading == 0:
            pc = ax.tripcolor(triangulation, vertex_values, shading=shading_mode, cmap=cmap_object, norm=norm, **pc_kwargs)
        else:
            vertex_colors = cmap_object(norm(vertex_values))
            shade_factor = np.clip(1.0 - shading * shade, 0.0, 1.0)
            vertex_colors[:, :3] *= shade_factor[:, None]
            pc = ax.tripcolor(triangulation, np.zeros(len(verts)),shading=shading_mode, cmap=cm.gray, **pc_kwargs)
            pc.set_array(None)
            pc.set_facecolors(vertex_colors)
            pc.surface_values = vertex_values
            pc.surface_norm = norm
            pc.surface_cmap = cmap_object

    else:
        # one base color, optionally shaded
        vertex_colors = np.tile(to_rgba(colors), (len(verts), 1))
        if shading != 0:
            shade_factor = np.clip(1.0 - shading * shade, 0.0, 1.0)
            vertex_colors[:, :3] *= shade_factor[:, None]

        pc = ax.tripcolor(triangulation, np.zeros(len(verts)), shading=shading_mode, cmap=cm.gray, **pc_kwargs)
        pc.set_array(None)
        pc.set_facecolors(vertex_colors)

    pc.set_rasterized(True)
    coordinates = (x, y, z)
    ax.set_xlim(coordinates[projection_axes[0]].min(), coordinates[projection_axes[0]].max())
    ax.set_ylim(coordinates[projection_axes[1]].min(), coordinates[projection_axes[1]].max())

    return pc
