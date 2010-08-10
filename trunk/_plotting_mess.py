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

# for the user to get at
tweaks = _pt

# expose all the eval statements to all the functions in numpy
from numpy import *


#
# General plotting routines
#


def xy_files(xscript=0, yscript=1, eyscript=None, paths='ask', **kwargs):
    """

    This selects a bunch of files, and plots them using plot.databoxes(**kwargs).
    Returns the databoxes as a list, and if the plot of a single value per
    databox, the first entry is the summary databox.

    xscript, yscript, eyscript      the scripts supplied to the data
    **kwargs                        sent to plot.databoxes

    setting xscript or yscript=None plots as a function of file number.

    """

    # have the user select a file
    if paths=="ask":
        paths = _dialogs.MultipleFiles("*.*", default_directory='default_directory')

    if paths in [[], None]: return

    if not isinstance(paths, type([])): paths = [paths]

    # for each path, open the file, get the data, and plot it
    databoxes = []
    for m in range(0, len(paths)):

        # fill up the xdata, ydata, and key
        databoxes.append(_data.load(paths[m]))

    # now plot everything
    value = xy_databoxes(databoxes, xscript=xscript, yscript=yscript, eyscript=eyscript, **kwargs)

    # return the data
    if value: databoxes.insert(0,value)
    return databoxes


def xy_databoxes(databoxes, xscript=0, yscript=1, eyscript=None, yshift=0.0, yshift_every=1, xscale='linear', yscale='linear', axes="gca", clear=2, autoformat=True, yaxis='left', xlabel=None, ylabel=None, legend_max="auto", paths="ask", debug=0, **kwargs):
    """

    This loops over the supplied databoxes and plots them. Databoxes can either
    be a list or a single databox.

    xscript, yscript, eyscript      the scripts evaluated for the x and y data
                                    setting to None plots as a function databox index+1

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

    if databoxes == None: return
    if not type(databoxes) in [list]: databoxes = [databoxes]

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

    # test and see what kind of script it is.
    if xscript==None or yscript==None or _fun.is_a_number(databoxes[0].execute_script(yscript)):
          singlemode=True
    else: singlemode=False

    # only used in single mode
    xdata = []
    ydata = []
    eydata= []
    sd    = _data.databox()

    # for each databox, open the file, get the data, and plot it
    for m in range(0, len(databoxes)):

        # get the databox
        data = databoxes[m]

        if singlemode:
            if xscript: xdata.append(data(xscript))
            else:       xdata.append(m+1)
            if yscript: ydata.append(data(yscript))
            else:       ydata.append(m+1)
            if eyscript: eydata.append(data(eyscript))
        else:
            data.plot(axes=a, yshift=(m/yshift_every)*yshift, clear=0, xscript=xscript, yscript=yscript, eyscript=eyscript, autoformat=False, **kwargs)

            # now fix the legend up nice like
            if m > legend_max-2 and m != len(paths)-1:
                a.get_lines()[-1].set_label('_nolegend_')
            elif m == legend_max-2:
                a.get_lines()[-1].set_label('...')

    if singlemode:
        sd.insert_header('xscript',xscript)
        sd.insert_header('yscript',yscript)
        sd['x']  = xdata
        sd['y']  = ydata
        if eyscript:
            sd.insert_header('eyscript',eyscript)
            sd['ey'] = eydata
            e = 'ey'
        else: e = None

        # massage the kwargs
        if not kwargs.has_key('xlabel'):  kwargs['xlabel']  = xscript
        if not kwargs.has_key('ylabel'):  kwargs['ylabel']  = yscript
        if not kwargs.has_key('lscript'):
            # if the user supplied script uses '' to reference elements
            if yscript.find("'") > 0:     kwargs['lscript'] = '"'+yscript+'"'
            else:                         kwargs['lscript'] = "'"+yscript+"'"

        sd.plot(axes=a, yshift=(m/yshift_every)*yshift, clear=0, xscript='x', yscript='y', eyscript=e, autoformat=False, **kwargs)


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

    if singlemode:  return sd
    else:           return None






def xy(xdata, ydata, label=None, xlabel="x", ylabel="y", title="", clear=1, axes="gca", draw=1, xscale='linear', yscale='linear', yaxis='left', legend='best', **kwargs):
    """
    Plots specified data.

    xdata, ydata        Arrays (or arrays of arrays) of data to plot
    label               string or array of strings for the line labels
    xlabel, ylabel      axes labels
    title               axes title
    clear=1             1=clear the axes first
                        2=clear the figure
    axes="gca"          which axes to use, or "gca" for the current axes
    draw=1              whether or not to draw the plot after plotting
    xscale,yscale       'linear' by default. Set either to 'log' for log axes
    yaxis='left'        set to 'right' for a pylab twinx() plot
    legend='best'       where to place the legend (see pylab.legend())
                        Set this to None to ignore the legend.

    """


    # if the first element is not a list, make it a list
    if not type(xdata[0]) in [type([]), type(_numpy.array([]))]:
        xdata = [xdata]
        ydata = [ydata]
        if label: label = [label]

    # clear the figure?
    if clear==2: _pylab.gcf().clear()

    # setup axes
    if axes=="gca":     axes = _pylab.gca()
    if yaxis=='right':  axes = _pylab.twinx()

    # if we're clearing the axes
    if clear: axes.clear()

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
    if legend: axes.legend(loc=legend)
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



def function_2D(f, xmin=-1, xmax=1, ymin=-1, ymax=1, xsteps=100, ysteps=100, p="x,y", g=None, plot="image", **kwargs):
    """
    Plots a 2-d function over the specified range

    f                       takes two inputs and returns one value. Can also
                            be a string function such as sin(x*y)
    xmin,xmax,ymin,ymax     range over which to generate/plot the data
    xsteps,ysteps           how many points to plot on the specified range
    p                       if using strings for functions, this is a string of parameters.
    g                       Optional additional globals. Try g=globals()!
    plot                    What type of surface data to plot ("image", "mountains")
    """

    # aggregate globals
    if not g: g = {}
    for k in globals().keys():
        if not g.has_key(k): g[k] = globals()[k]

    if type(f) == str:
        f = eval('lambda ' + p + ': ' + f, g)


    # generate the grid x and y coordinates
    xones = _numpy.linspace(1,1,ysteps)
    x     = _numpy.linspace(xmin, xmax, xsteps)
    xgrid = _numpy.outer(xones, x)

    yones = _numpy.linspace(1,1,xsteps)
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
    return xyz(x,y,zgrid,plot,**kwargs)


def function_1D(f, xmin=-1, xmax=1, steps=200, p='x', g=None, erange=False, **kwargs):
    """

    Plots the function over the specified range

    f                   function or list of functions to plot; can be string functions
    xmin, xmax, steps   range over which to plot, and how many points to plot
    p                   if using strings for functions, p is the parameter name
    g                   optional dictionary of extra globals. Try g=globals()!
    erange              Use exponential spacing of the x data?

    **kwargs are sent to plot.data()

    """

    if not g: g = {}
    for k in globals().keys():
        if not g.has_key(k): g[k] = globals()[k]

    # if the x-axis is a log scale, use erange
    if erange: r = _fun.erange(xmin, xmax, steps)
    else:      r = _numpy.linspace(xmin, xmax, steps)

    # make sure it's a list so we can loop over it
    if not type(f) in [type([]), type(())]: f = [f]

    # loop over the list of functions
    xdatas = []
    ydatas = []
    labels = []
    for fs in f:
        if type(fs) == str:
            a = eval('lambda ' + p + ': ' + fs, g)
            a.__name__ = fs
        else:
            a = fs

        x = []
        y = []
        for z in r:
            x.append(z)
            y.append(a(z))

        xdatas.append(x)
        ydatas.append(y)
        labels.append(a.__name__)

    # plot!
    return xy(xdatas, ydatas, label=labels, **kwargs)





def function_parametric(fx, fy, tmin=-1, tmax=1, steps=200, p='t', g=None, erange=False, **kwargs):
    """

    Plots the parametric function over the specified range

    f                   function or list of functions to plot; can be string functions
    xmin, xmax, steps   range over which to plot, and how many points to plot
    p                   if using strings for functions, p is the parameter name
    g                   optional dictionary of extra globals. Try g=globals()!
    erange              Use exponential spacing of the t data?

    **kwargs are sent to plot.data()

    """

    if not g: g = {}
    for k in globals().keys():
        if not g.has_key(k): g[k] = globals()[k]

    # if the x-axis is a log scale, use erange
    if erange: r = _fun.erange(tmin, tmax, steps)
    else:      r = _numpy.linspace(tmin, tmax, steps)

    # make sure it's a list so we can loop over it
    if not type(fy) in [type([]), type(())]: fy = [fy]
    if not type(fx) in [type([]), type(())]: fx = [fx]

    # loop over the list of functions
    xdatas = []
    ydatas = []
    labels = []
    for fs in fx:
        if type(fs) == str:
            a = eval('lambda ' + p + ': ' + fs, g)
            a.__name__ = fs
        else:
            a = fs

        x = []
        for z in r: x.append(a(z))

        xdatas.append(x)
        labels.append(a.__name__)

    for n in range(len(fy)):
        fs = fy[n]
        if type(fs) == str:
            a = eval('lambda ' + p + ': ' + fs, g)
            a.__name__ = fs
        else:
            a = fs

        y = []
        for z in r: y.append(a(z))

        ydatas.append(y)
        labels[n] = labels[n]+', '+a.__name__


    # plot!
    return xy(xdatas, ydatas, label=labels, **kwargs)









