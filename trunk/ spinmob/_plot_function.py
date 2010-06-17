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

# expose all the eval statements to all the functions in numpy
from numpy import *
from _plotting import xy, xyz

def function_2D(f, xmin=-1, xmax=1, ymin=-1, ymax=1, xsteps=100, ysteps=100, p="x,y", plot="image", **kwargs):
    """
    Plots a 2-d function over the specified range

    f                       takes two inputs and returns one value. Can also
                            be a string function such as sin(x*y)
    xmin,xmax,ymin,ymax     range over which to generate/plot the data
    xsteps,ysteps           how many points to plot on the specified range
    p                       if using strings for functions, this is a string of parameters.
    plot                    What type of surface data to plot ("image", "mountains")
    """

    if type(f) == str:
        f = eval('lambda ' + p + ': ' + f, globals())


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


def function_1D(f, xmin=-1, xmax=1, steps=200, p='x', erange=False, **kwargs):
    """

    Plots the function over the specified range

    f                   function or list of functions to plot; can be string functions
    xmin, xmax, steps   range over which to plot, and how many points to plot
    p                   if using strings for functions, p is the parameter name
    erange              Use exponential spacing of the x data?

    **kwargs are sent to plot.data()

    """

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
            a = eval('lambda ' + p + ': ' + fs, globals())
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


