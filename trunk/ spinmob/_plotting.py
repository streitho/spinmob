import os           as _os
import glob         as _glob
import wx           as _wx
import thread       as _thread
import pylab        as _pylab
import numpy        as _numpy
import matplotlib   as _mpl

from matplotlib.font_manager import FontProperties as _FontProperties


import _functions as _fun
import _data_types as _data
import _pylab_tweaks as _pt
import _dialogs

# for the user to get at
tweaks = _pt
style  = _pt.style


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
    _pt.gui_colormap()
    return data


def files(xscript=0, yscript=1, yerror=None, yshift=0.0, yshift_every=1, clear=1, yaxis='left', legend_max="auto", paths="ask", coarsen=0, debug=0, data=_data.standard(), **kwargs):
    """

    This selects a bunch of files, and plots them.

    xscript, yscript    the scripts supplied to the data
    yshift=0.0          artificial yshift
    clear=1             clear existing plot first
    yaxis='left'        if 'right', this will make an overlay axis on the right (also doesn't clear)
    paths='ask'         list of full paths to data files (or we'll ask for a list)
    data                instance of data class used to extract plot from the files

    **kwargs are sent to data.standard().plot()

    """

    # have the user select a file
    if paths=="ask":
        paths = _dialogs.MultipleFiles(data.file_extension, default_directory=data.directory)

    if paths in [[], None]: return

    if not isinstance(paths, type([])): paths = [paths]


    # get and clear the figure and axes
    f = _pylab.gcf()
    if clear and yaxis=='left': f.clf()

    # setup the right-hand axis
    a = _pylab.gca()
    if yaxis=='right':  a = _pylab.twinx()

    # determine the max legend entries
    if legend_max == "auto":
        if yshift: legend_max=40
        else:      legend_max=30


    # for each path, open the file, get the data, and plot it
    for m in range(0, len(paths)):

        # fill up the xdata, ydata, and key
        if debug: print "FILE: "+paths[m]
        data.load_file(paths[m])
        data.plot(axes=a, yshift=(m/yshift_every)*yshift, clear=0, format=0, coarsen=coarsen, xscript=xscript, yscript=yscript, yerror=yerror, **kwargs)

        # now fix the legend up nice like
        if m > legend_max-2 and m != len(paths)-1:
            a.get_lines()[-1].set_label('_nolegend_')
        elif m == legend_max-2:
            a.get_lines()[-1].set_label('...')

    # fix up the title if there's an yshift
    if yshift: data.title += ', progressive y-yshift='+str(yshift)
    if yaxis=="right": data.title = data.title + "\n"
    a.set_title(data.title)

    if yshift: _pt.format_figure(f, tall=True)
    else:      _pt.format_figure(f, tall=False)

    _pylab.draw()
    _pt.get_figure_window()
    _pt.get_pyshell()

    # return the axes
    return data





def _files_as_points(datax, datay, clear=True):
    """

    This is antiquated and will be updated when I need it again.

    This selects a bunch of files, and plots some quantity, one data point per file.

    datax    is an instance of a data class that takes a file
                list and assembles header data
    datay    is the same type, but will be the y-values.

    clear=True  should we clear the axes?

    """

    # have the user select a file
    paths = _dialogs.MultipleFiles('DIS AND DAT|*.*',
                                default_directory=header_x.directory)

    if paths == []: return

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



def _massive(data, offset=0.0, print_plots=False, arguments="-color", pause=True, threaded=False, f=files):
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


def data(xdata, ydata, label=None, xlabel="x", ylabel="y", title="y(x)", clear=1, axes="gca", draw=1, plot='plot', yaxis='left', **kwargs):
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
        eval('axes.'+plot+'(xdata[n], ydata[n], label=l, **kwargs)')

    axes.legend(loc='best')
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)

    # update the canvas
    if draw: _pylab.draw()
    return axes

def function(function, xmin=-1, xmax=1, steps=200, clear=True, silent=False, axes="gca", legend=True, plot='plot'):
    """

    Plots the function over the specified range

    function            function or list of functions to plot
    xmin, xmax, steps   range over which to plot, and how many points to plot
    clear=True          clear the previous plot
    silent=False        whether or not to update the plot after we're done
    axes='gca'          instance of axes on which to plot (or 'gca' for current axes)
    legend=True         should we attempt to construct a legend for the plot?
    plot='plot'         plot method, can be 'plot', 'semilogx', 'semilogy', or 'loglog'

    """

    if axes=="gca": axes = _pylab.gca()
    if clear:
        axes.figure.clear()
        axes=_pylab.gca()

    # if the x-axis is a log scale, use erange
    if plot in ['semilogx', 'loglog']: r = _fun.erange(xmin, xmax, steps)
    else:                              r = _fun.frange(xmin, xmax, (float(xmax)-float(xmin))/float(steps))

    # make sure it's a list so we can loop over it
    try:
        function[0]
    except:
        function = [function]

    # loop over the list of functions
    for f in function:
        x = []
        y = []
        for z in r:
            x.append(z)
            y.append(f(z))

        # add the line to the plot
        eval('axes.'+plot+'(x, y, color=style.get_line_color(1), linestyle=style.get_linestyle(1), label=f.__name__)')

    if legend: axes.legend()

    if not silent:
        _pt.auto_zoom()
        _pt.raise_figure_window()
        _pt.raise_pyshell()

    return axes


def surface_data(zgrid, xmin=0, xmax=1, ymin=0, ymax=1):
    """
    Generates an image plot

    zgrid                   2-d array of z-values
    xmin,xmax,ymin,ymax     range upon which to place the image
    """

    fig = _pylab.gcf()
    fig.clear()

    # generate the 3d axes
    axes = _pylab.gca()

    axes.imshow(zgrid, interpolation='bilinear', origin='lower', cmap=_mpl.cm.hot, extent=(xmin,xmax,ymin,ymax), aspect=1.0)

    _pt.raise_figure_window()
    _pt.raise_pyshell()

    _pt.close_sliders();
    _pt.gui_colormap()
    _pt.image_set_aspect(1.0)
    _pylab.draw()
    return axes

def surface_function(f, xmin, xmax, ymin, ymax, xsteps=50, ysteps=50):
    """
    Plots a 2-d function over the specified range

    f                       takes two inputs and returns one value
    xmin,xmax,ymin,ymax     range over which to generate/plot the data
    xsteps,ysteps           how many points to plot on the specified range

    """

    # generate the grid x and y coordinates
    xones = _numpy.linspace(1,1,xsteps)
    x     = _numpy.linspace(xmin, xmax, xsteps)
    xgrid = _numpy.outer(xones, x)

    yones = _numpy.linspace(1,1,ysteps)
    y     = _numpy.linspace(ymin, ymax, ysteps)
    ygrid = _numpy.outer(y, yones)

    # now get the z-grid
    try:
        # try it the fast numpy way. Add 0 to assure dimensions
        zgrid = f(xgrid, ygrid) + xgrid*0.0
    except:
        print "Notice: function is not rocking hardcore. Generating grid the slow way..."
        # manually loop over the data to generate the z-grid
        zgrid = []
        for ny in range(0, len(y)):
            zgrid.append([])
            for nx in range(0, len(x)):
                zgrid[ny].append(f(x[nx], y[ny]))

        zgrid = _numpy.array(zgrid)



    # now plot!
    axes = surface_data(zgrid,xmin,xmax,ymin,ymax)

    axes.set_xlabel("x")
    axes.set_ylabel("y")

    _pylab.draw()
    _pt.raise_figure_window()
    _pt.raise_pyshell()

    return axes
























