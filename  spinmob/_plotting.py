import os           as _os
import glob         as _glob
import wx           as _wx
import thread       as _thread
import pylab        as _pylab
import numpy        as _numpy
import matplotlib   as _mpl

from matplotlib.font_manager import FontProperties as _FontProperties


import _functions as _fun
import _data
import _pylab_tweaks as _pt
import _dialogs
import _plot_function

function = _plot_function.function_1D
zfunction = _plot_function.function_2D

# for the user to get at
tweaks = _pt


#
# General plotting routines
#
def _image():
    """
    This asks for a file and plots a color image of the data grid.
    """
    data.load_file()
    data.get_XYZ(None,None)
    data.plot_image(map='_last')
    _pt.image_sliders()
    return data


def xy_files(xscript=0, yscript=1, eyscript=None, yshift=0.0, yshift_every=1, xscale='linear', yscale='linear', axes="gca", clear=2, autoformat=True, yaxis='left', xlabel=None, ylabel=None, legend_max="auto", paths="ask", debug=0, **kwargs):
    """

    This selects a bunch of files, and plots them.

    xscript, yscript, eyscript      the scripts supplied to the data

    yshift=0.0                      artificial (progressive) yshift

    yshift_every=1                  how many lines to plot before each shift

    xscale, yscale                  can be 'linear' or 'log' or 'symlog'
                                    see the xscale() function in matplotlib.

    axes='gca'                      axes instance. 'gca' uses current axes

    clear=2                         clear=2: clear existing figure first
                                    clear=1: only clear existing plots on axes
                                    clear=0: do not clear first

    autoformat=True                 autoformat the figure after plotting
                                    NOTE: clear is set to 1 (if it's 2) and
                                    autoformat is set to False if you specify
                                    axes! Usually you don't want to clear the
                                    figure or use the auto_format() feature in
                                    this case.

    yaxis='left'                    if 'right', this will make an overlay axis
                                    on the right (also doesn't clear)

    xlabel=None, ylabel=None        x and y labels if you want to override
                                    from xscript and yscript.

    legend_max='auto'               maximum number of legend entries (if you're
                                    selecting a lot of files)

    paths='ask'                     list of full paths to data files (or we'll
                                    ask for a list)

    **kwargs are sent to data.databox().plot()

    """

    # have the user select a file
    if paths=="ask":
        paths = _dialogs.MultipleFiles("*.*", default_directory='default_directory')

    if paths in [[], None]: return

    if not isinstance(paths, type([])): paths = [paths]


    # get the figure
    f = _pylab.gcf()

    # setup axes
    if axes=="gca":
        # only clear the figure if we didn't specify an axes
        if clear==2 and yaxis=='left': f.clear()
        a = _pylab.gca()
    else:
        a = axes
        autoformat=False

    if yaxis=='right':  a = _pylab.twinx()

    # if we're clearing the axes
    if clear: a.clear()


    # determine the max legend entries
    if legend_max == "auto":
        if yshift: legend_max=40
        else:      legend_max=30


    # for each path, open the file, get the data, and plot it
    _pylab.hold(True)
    for m in range(0, len(paths)):

        # fill up the xdata, ydata, and key
        if debug: print "FILE: "+paths[m]
        data = _data.load(paths[m])

        data.plot(axes=a, yshift=(m/yshift_every)*yshift, clear=0, xscript=xscript, yscript=yscript, eyscript=eyscript, autoformat=False, **kwargs)

        # now fix the legend up nice like
        if m > legend_max-2 and m != len(paths)-1:
            a.get_lines()[-1].set_label('_nolegend_')
        elif m == legend_max-2:
            a.get_lines()[-1].set_label('...')

    # set the scale
    if not xscale=='linear': _pylab.xscale(xscale)
    if not yscale=='linear': _pylab.yscale(yscale)

    # add the axis labels
    if not xlabel==None: a.set_xlabel(xlabel)
    if not ylabel==None: a.set_ylabel(ylabel)

    # fix up the title if there's an yshift
    if yshift: data.title += ', progressive y-yshift='+str(yshift)
    if yaxis=="right": data.title = data.title + "\n"
    a.set_title(data.title)

    # leave it unformatted unless the user tells us to autoformat
    a.title.set_visible(0)

    if autoformat:
        if yshift: _pt.format_figure(f, tall=True)
        else:      _pt.format_figure(f, tall=False)

    _pylab.draw()
    _pt.get_figure_window()
    _pt.get_pyshell()

    # return the axes
    return data





def _files_as_points(xscript, yscript, clear=True):
    """

    This is antiquated and will be updated when I need it again.

    This selects a bunch of files, and plots some quantity, one data point per file.

    xscript  is an instance of a data class that takes a file
                list and assembles header data
    yscript  is the same type, but will be the y-values.

    clear=True  should we clear the axes?

    """

    # have the user select a file
    if paths=="ask":
        paths = _dialogs.MultipleFiles(data.file_extension, default_directory=data.directory)

    if paths in [[], None]: return

    if not isinstance(paths, type([])): paths = [paths]

    # get the header data arrays
    datax.get_data(paths)
    datay.get_data(paths)

    # get and clear the figure and axes
    f = _pylab.gcf()
    if clear: f.clf()
    a = _pylab.gca()

    # plot it
    a.plot(datax.data, datay.data)

    # now get the pieces of the string for the plot title
    pathparts = paths[-1].split('\\')

    # now add the path to the title (either the full path or back a few steps
    x = min([5, len(pathparts)])
    title = ".../"
    for n in range(0, x-2): title += pathparts[n-x+1] + '/'

    # now split the first line and use it to get the axes title
    s     = datax.lines[0].split(' ')
    title += '\nLast trace on:'+s[2]+'/'+s[3]+'/'+s[4]

    # format the axes and refresh
    # set the axis labels
    axes.set_title(title)
    axes.set_xlabel(datax.xlabel)
    axes.set_ylabel(datay.ylabel)
    format_figure(axes.figure)

    return axes



def _massive(data, offset=0.0, print_plots=False, arguments="-color", pause=True, threaded=False, f=xy_files):
    """

    This selects a directory full of directories, and makes a series of plots, one per subdirectory.

    data                (class) data to extract from the files
    offset=0.0          artificial offset
    clear=1             clear existing plot first
    line=''             line specifier for pylab
    marker=''           symbol specifier for pylab
    yaxis='left'        if 'right', this will make an overlay axis on the right (also doesn't clear)
    paths='ask'         list of full paths to data files (or we'll ask for a list)
    style=None          matplotlib plotting style, such as '-o'

    """


    d = _dialogs.Directory();
    if d == "": return
    contents = _os.listdir(d) # doesn't include root path
    contents.sort()

    for file in contents:
        if _os.path.isdir(d+"\\"+file):
            paths = _glob.glob(d+"\\"+file+"\\*.DAT")
            f(data, offset, paths=paths)
            if pause:
                if raw_input("<enter>") == "q": return

            if print_plots: printer(arguments, threaded)

    return


def xy(xdata, ydata, label=None, xlabel="x", ylabel="y", title="y(x)", clear=1, axes="gca", draw=1, xscale='linear', yscale='linear', yaxis='left', **kwargs):
    """
    Plots specified data.

    xdata, ydata        Arrays (or arrays of arrays) of data to plot
    label               string or array of strings for the line labels
    xlabel, ylabel      axes labels
    title               axes title
    clear=1             clear the axes first
    axes="gca"          which axes to use, or "gca" for the current axes
    draw=1              whether or not to draw the plot after plotting
    plot='plot'         plot style: can be 'plot', 'semilogx', 'semilogy', 'loglog'
    yaxis='left'        set to 'right' for a pylab twinx() plot

    """


    # if the first element is not a list, make it a list
    if not type(xdata[0]) in [type([]), type(_numpy.array([]))]:
        xdata = [xdata]
        ydata = [ydata]
        if label: label = [label]

    # get the current axes
    if axes=="gca": axes = _pylab.gca()
    _pylab.figure(axes.figure.number)

    # get rid of the old plot
    if clear:
        axes.figure.clear()
        axes = _pylab.gca()

    # if yaxis is 'right' set up the twinx()
    if yaxis=='right':
        axes = _pylab.twinx()

    # now loop over the list of data in xdata and ydata
    for n in range(0,len(xdata)):
        if label: l = label[n]
        else:     l = str(n)
        axes.plot(xdata[n], ydata[n],  label=l, **kwargs)

    _pylab.xscale(xscale)
    _pylab.yscale(yscale)
    axes.legend(loc='best')
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)

    # update the canvas
    if draw: _pylab.draw()
    return axes



def xyz(X, Y, Z, plot="image", **kwargs):
    """
    Generates an image or 3d plot

    X                       1-d array of x-values
    Y                       1-d array of y-values
    Z                       2-d array of z-values
    plot                    What type of surface data to plot ("image", "mountains")
    """

    fig = _pylab.gcf()
    fig.clear()
    axes = _pylab.axes()

    # generate the 3d axes
    d=_data.standard()
    d.X = _numpy.array(X)
    d.Y = _numpy.array(Y)
    d.Z = _numpy.array(Z)

    d.path = ""
    d.xlabel = "X"
    d.ylabel = "Y"

    d.plot_XYZ(plot=plot, **kwargs)

    if plot=="image":
        _pt.close_sliders()
        _pt.image_sliders()

    _pt.raise_figure_window()
    _pt.raise_pyshell()
    _pylab.draw()
    return axes


def zgrid(Z, xmin=0, xmax=1, ymin=0, ymax=1, plot="image", **kwargs):
    """
    Generates an image plot on the specified range

    Z                       2-d array of z-values
    xmin,xmax,ymin,ymax     range upon which to place the image
    plot                    What type of surface data to plot ("image", "mountains")
    """

    fig = _pylab.gcf()
    fig.clear()
    axes = _pylab.axes()

    # generate the 3d axes
    X = _numpy.linspace(xmin,xmax,len(Z))
    Y = _numpy.linspace(ymin,ymax,len(Z[0]))

    return xyz(X,Y,Z,plot)










files = xy_files
data = xy
surface_data = xyz
surface_function = zfunction














