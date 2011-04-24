import os           as _os
import pylab        as _pylab
import numpy        as _numpy
import itertools    as _itertools

import _functions as _fun
import _data
import _pylab_tweaks as _pt
import _dialogs
import spinmob as _s

# for the user to get at
tweaks = _pt
_n = _numpy

# expose all the eval statements to all the functions in numpy
from numpy import *


#
# General plotting routines
#





def complex_data(data, **kwargs):
    """
    Plots the X and Y of complex data.

    data                complex data
    
    kwargs are sent to spinmob.plot.xy.data()
    """
    try:
        rdata = _n.real(data)
        idata = _n.imag(data)
    except:
        rdata = []
        idata = []
        for x in data:
            rdata.append(_n.real(x))
            idata.append(_n.imag(x))

    return xy_data(rdata, idata, **kwargs)



def complex_databoxes(ds, script='c(1)+1j*c(2)', **kwargs):
    """
    Use script to generate data and send to harrisgroup.plot.complex_data()    
    
    ds        list of databoxes
    script    comlex script
    
    **kwargs are sent to spinmob.plot.xy.complex_data()    
    """
    
    datas  = []
    labels = []
    for d in ds: 
        datas.append(d(script))
        labels.append(_os.path.split(d.path)[-1])
    
    return complex_data(datas, label=labels, **kwargs)



def complex_files(script='c(1)+1j*c(2)', **kwargs):
    """
    Loads and plots complex data in the real-imaginary plane.
    
    **kwargs are sent to harrisgroup.plot.complex_databoxes()    
    """
    
    ds = _s.data.load_multiple()

    if len(ds) == 0: return
    
    if not kwargs.has_key('title'): 
        kwargs['title']=_os.path.split(ds[0].path)[0]

    return complex_databoxes(ds, script=script, **kwargs)




def magphase_databoxes(ds, xscript=0, yscript='c(1)+1j*c(2)', **kwargs):
    """
    Use script to generate data and send to harrisgroup.plot.complex_data()    
    
    ds        list of databoxes
    script    comlex script
    
    **kwargs are sent to harrisgroup.plot.complex_data()    
    """
    
    xdatas = []
    ydatas = []
    labels = []
    
    for d in ds: 
        xdatas.append(d(xscript))
        ydatas.append(d(yscript))
        labels.append(_os.path.split(d.path)[-1])
    
    return magphase_data(xdatas, ydatas, label=labels, **kwargs)



def magphase_files(xscript=0, yscript='c(1)+1j*c(2)', **kwargs):
    """
    Loads and plots complex data in the real-imaginary plane.
    
    **kwargs are sent to harrisgroup.plot.complex_databoxes()    
    """
    
    ds = _s.data.load_multiple()

    if len(ds) == 0: return
    
    if not kwargs.has_key('title'): 
        kwargs['title']=_os.path.split(ds[0].path)[0]

    return magphase_databoxes(ds, xscript=xscript, yscript=yscript, **kwargs)

def magphase_function(f, xmin=-1, xmax=1, steps=200, p='x', g=None, erange=False, **kwargs):
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
    return magphase_data(xdatas, ydatas, label=labels, **kwargs)



def magphase_data(xdata, ydata, xscale='linear', mscale='linear', pscale='linear', mlabel='Magnitude', plabel='Phase', phase='degrees', figure='gcf', clear=1,  **kwargs):
    """
    Plots the magnitude and phase of complex ydata.

    xdata               real-valued x-axis data
    ydata               complex data
    xscale='linear'     'log' or 'linear'
    mscale='linear'     'log' or 'linear' (only applies to the magnitude graph)
    pscale='linear'     'log' or 'linear' (only applies to the phase graph)
    mlabel='Magnitude'  y-axis label for magnitude plot
    plabel='Phase'      y-axis label for phase plot
    phase='degrees'     'degrees' or 'radians'
    figure='gcf'        figure instance
    clear=1             clear the figure?


    kwargs are sent to plot.data()
    """

    if figure == 'gcf': f = _pylab.gcf()
    if clear: f.clear()

    axes1 = _pylab.subplot(211)
    axes2 = _pylab.subplot(212,sharex=axes1)

    try:
        m = _n.abs(ydata)
        p = _n.angle(ydata)
    except:
        m = []
        p = []
        for y in ydata:
            m.append(abs(y))
            p.append(angle(y))        
    
    if phase=='degrees':
        plabel = plabel + " (degrees)"
        try:    p = p*180.0/_n.pi
        except:
            a = []
            for x in p: a.append(x*180.0/_n.pi)
            p = a
    else:
        plabel = plabel + " (radians)"

    if kwargs.has_key('xlabel'): xlabel=kwargs['xlabel']
    else:                        xlabel=''
    if not kwargs.has_key('draw'): kwargs['draw'] = False

    kwargs['xlabel'] = ''
    xy_data(xdata, m, ylabel=mlabel, axes=axes1, clear=0, xscale=xscale, yscale=mscale, **kwargs)

    kwargs['xlabel'] = xlabel
    xy_data(xdata, p, ylabel=plabel, axes=axes2, clear=0, xscale=xscale, yscale=pscale, **kwargs)

    axes2.set_title('')
    _pt.auto_zoom(axes=axes1)
    _pylab.draw()

def realimag_data(xdata, ydata, xscale='linear', rscale='linear', iscale='linear', rlabel='Real', ilabel='Imaginary', figure='gcf', clear=1, **kwargs):
    """
    Plots the magnitude and phase of complex ydata.

    xdata               real-valued x-axis data
    ydata               complex data
    xscale='linear'     'log' or 'linear'
    rscale='linear'     'log' or 'linear' for the real yscale
    iscale='linear'     'log' or 'linear' for the imaginary yscale
    rlabel='Real'       y-axis label for magnitude plot
    ilabel='Imaginary'  y-axis label for phase plot
    figure='gcf'        figure instance
    clear=1             clear the figure?

    kwargs are sent to plot.data()
    """

    if figure == 'gcf': f = _pylab.gcf()
    if clear: f.clear()

    axes1 = _pylab.subplot(211)
    axes2 = _pylab.subplot(212,sharex=axes1)

    rdata = _n.real(ydata)
    idata = _n.imag(ydata)

    if kwargs.has_key('xlabel')  : xlabel=kwargs['xlabel']
    else:                          xlabel=''
    if not kwargs.has_key('draw'): kwargs['draw'] = False


    kwargs['xlabel'] = ''
    xy_data(xdata, rdata, ylabel=rlabel, axes=axes1, clear=0, xscale=xscale, yscale=rscale, **kwargs)

    kwargs['xlabel'] = xlabel
    xy_data(xdata, idata, ylabel=ilabel, axes=axes2, clear=0, xscale=xscale, yscale=iscale, **kwargs)

    axes2.set_title('')
    _pylab.draw()

def realimag_function(f, xmin=-1, xmax=1, steps=200, p='x', g=None, erange=False, **kwargs):
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
    return realimag_data(xdatas, ydatas, label=labels, **kwargs)


def xy_data(xdata, ydata, eydata=None, exdata=None, style=None, label=None, xlabel="x", ylabel="y", title='', pyshell_history=1, clear=True, axes=None, draw=1, xscale='linear', yscale='linear', yaxis='left', legend='best', grid=False, autoformat=True, tall=False, **kwargs):
    """
    Plots specified data.

    xdata, ydata        Arrays (or arrays of arrays) of data to plot
    label               string or array of strings for the line labels
    style               style cycle object.
    xlabel, ylabel      axes labels
    title               axes title
    pyshell_history=1   how many commands from the pyshell history to include 
                        above the title
    axes=None           axes to use. Set axes='gca' to use the current axes.
    clear=True          if no axes are specified, clear the figure, otherwise
                        clear just the axes.
    axes="gca"          which axes to use, or "gca" for the current axes
    draw=1              whether or not to draw the plot after plotting
    xscale,yscale       'linear' by default. Set either to 'log' for log axes
    yaxis='left'        set to 'right' for a pylab twinx() plot
    legend='best'       where to place the legend (see pylab.legend())
                        Set this to None to ignore the legend.
    grid=False          Should we draw a grid on the axes?
    autoformat=True     Should we format the figure for printing?
    tall=False          Should the format be tall?
    """

    # if the first element is not a list, make it a list
    if not _fun.is_iterable(xdata[0]):
        xdata = [xdata]
        ydata = [ydata]
        if label: label = [label]

    if not _fun.is_iterable(label): label = [label]

    if not len(label) == len(xdata):
        l = []
        for x in xdata: l.append(label[0])
        label = l

    # clear the figure?
    if clear and not axes: _pylab.gcf().clear()

    # setup axes
    if axes=="gca":     axes = _pylab.gca()
    if yaxis=='right':  axes = _pylab.twinx()

    # if we're clearing the axes
    if clear: axes.clear()

    # if yaxis is 'right' set up the twinx()
    if yaxis=='right':
        axes = _pylab.twinx()

    # set the current axes    
    _pylab.axes(axes)

    # now loop over the list of data in xdata and ydata
    for n in range(0,len(xdata)):
        if label: l = str(label[n])
        else:     l = str(n)

        if not style==None: kwargs.update(style.next())
        axes.errorbar(xdata[n], ydata[n], label=l, yerr=eydata, xerr=exdata, **kwargs)

    _pylab.xscale(xscale)
    _pylab.yscale(yscale)
    if legend: axes.legend(loc=legend)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    
    # add the commands to the title
    if title in [None, False, 0]: title = ''
    title = str(title)
    if pyshell_history:
        for n in range(pyshell_history): 
            if not title == '': title = _pt.get_pyshell_command(n) + "\n" + title
            else:               title = _pt.get_pyshell_command(n)
    axes.set_title(title)
    if grid: _pylab.grid(True)

    if autoformat: _pt.format_figure(tall=tall, draw=draw, vertical_reshape=False)

    # update the canvas
    if draw: _pylab.draw()
    return axes

def xy_databoxes(databoxes, xscript=0, yscript=1, eyscript=None, exscript=None, lscript=None, yshift=0.0, yshift_every=1, xscale='linear', yscale='linear', axes="gca", clear=2, yaxis='left', xlabel=None, ylabel=None, legend_max="auto", paths="ask", draw=1, debug=0, **kwargs):
    """

    This loops over the supplied databoxes and plots them. Databoxes can either
    be a list or a single databox.

    xscript, yscript,               the scripts evaluated for the x and y data
    eyscript, exscript              setting to None plots as a function databox index+1

    lscript                         script for generating line labels

    yshift=0.0                      artificial (progressive) yshift

    yshift_every=1                  how many lines to plot before each shift

    xscale, yscale                  can be 'linear' or 'log' or 'symlog'
                                    see the xscale() function in matplotlib.

    axes='gca'                      axes instance. 'gca' uses current axes

    clear=2                         clear=2: clear existing figure first
                                    clear=1: only clear existing plots on axes
                                    clear=0: do not clear first

    yaxis='left'                    if 'right', this will make an overlay axis
                                    on the right (also doesn't clear)

    xlabel=None, ylabel=None        x and y labels if you want to override
                                    from xscript and yscript.

    legend_max='auto'               maximum number of legend entries (if you're
                                    selecting a lot of files)

    paths='ask'                     list of full paths to data files (or we'll
                                    ask for a list)

    **kwargs are sent to plot_data() aka xy()

    """

    if databoxes == None: return
    if not type(databoxes) in [list]: databoxes = [databoxes]

    if not kwargs.has_key('autoformat'): kwargs['autoformat'] = True

    # get the figure
    f = _pylab.gcf()

    # setup axes
    if axes=="gca":
        # only clear the figure if we didn't specify an axes
        if clear==2 and yaxis=='left': f.clear()
        a = _pylab.gca()
    else:
        a = axes

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
    xdata  = []
    ydata  = []
    if eyscript:    eydata = []
    else:           eydata = None
    if exscript:    exdata = []
    else:           exdata = None

    sd = _data.databox() # for returning

    # for each databox, open the file, get the data, and plot it
    for m in range(0, len(databoxes)):

        # get the databox
        data = databoxes[m]
        
        # get the label for the line
        if lscript==None:
            if len(data.path):  label = _os.path.split(data.path)[-1]
            else:               label = str(m)
        else:                   label = str(data(lscript))

        # get or append the data depending on the mode
        if singlemode:
            if xscript: xdata.append(data(xscript))
            else:       xdata.append(m+1)
            if yscript: ydata.append(data(yscript))
            else:       ydata.append(m+1)
            if eyscript: eydata.append(data(eyscript))
            if exscript: exdata.append(data(exscript))
        else:

            # get the data
            xdata = data(xscript)
            if yshift:   ydata = data(yscript) + (m/yshift_every)*yshift
            else:        ydata = data(yscript)
            if eyscript: eydata = data(eyscript)
            else:        eydata = None
            if exscript: exdata = data(exscript)
            else:        exdata = None

            # update the kwargs
            if not kwargs.has_key('xlabel'): kwargs['xlabel'] = xscript
            if not kwargs.has_key('ylabel'): kwargs['ylabel'] = yscript
            if not kwargs.has_key('title'):  kwargs['title' ] = _os.path.split(data.path)[0]

            # PLOT!
            xy_data(xdata, ydata, eydata, exdata, axes=a, clear=0, label=label, draw=0, **kwargs)

            # now fix the legend up nice like
            if m > legend_max-2 and m != len(databoxes)-1:
                a.get_lines()[-1].set_label('_nolegend_')
            elif m == legend_max-2:
                a.get_lines()[-1].set_label('...')

    if singlemode:
        sd.insert_header('xscript',xscript)
        sd.insert_header('yscript',yscript)
        sd.insert_header('eyscript',eyscript)
        sd.insert_header('exscript',exscript)

        sd['x']  = xdata
        sd['y']  = ydata
        sd['ey'] = eydata
        sd['ex'] = exdata

        # massage the kwargs
        if not kwargs.has_key('xlabel'):  kwargs['xlabel']  = xscript
        if not kwargs.has_key('ylabel'):  kwargs['ylabel']  = yscript
        if lscript: kwargs['label'] = sd(lscript)

        xy_data(sd['x'], sd['y'], eydata=sd['ey'], exdata=sd['ex'], label=label, axes=a, clear=0, **kwargs)

    # set the scale
    if not xscale=='linear': _pylab.xscale(xscale)
    if not yscale=='linear': _pylab.yscale(yscale)

    # add the axis labels
    if not xlabel==None: a.set_xlabel(xlabel)
    if not ylabel==None: a.set_ylabel(ylabel)

    # leave it unformatted unless the user tells us to autoformat
    if kwargs['autoformat']:
        if yshift: kwargs['tall']=True
        else:      kwargs['tall']=False

    if draw: _pylab.draw()
    _pt.get_figure_window()
    _pt.get_pyshell()

    if singlemode:  return sd
    else:           return None

def xy_files(xscript=0, yscript=1, eyscript=None, exscript=None, paths='ask', **kwargs):
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
    value = xy_databoxes(databoxes, xscript=xscript, yscript=yscript, eyscript=eyscript, exscript=exscript, **kwargs)

    # return the data
    if value: databoxes.insert(0,value)
    return databoxes


def xy_function(f, xmin=-1, xmax=1, steps=200, p='x', g=None, erange=False, **kwargs):
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
    return xy_data(xdatas, ydatas, label=labels, **kwargs)



def image_data(X, Y, Z, plot="image", **kwargs):
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


def image_autogrid(Z, xmin=0, xmax=1, ymin=0, ymax=1, plot="image", **kwargs):
    """
    Generates an image plot on the specified range

    Z                       2-d array of z-values
    xmin,xmax,ymin,ymax     range upon which to place the image
    plot                    What type of surface data to plot ("image", "mountains")
    """

    fig = _pylab.gcf()
    fig.clear()
    _pylab.axes()

    # generate the 3d axes
    X = _numpy.linspace(xmin,xmax,len(Z))
    Y = _numpy.linspace(ymin,ymax,len(Z[0]))

    return image_data(X,Y,Z,plot, **kwargs)



def image_function(f, xmin=-1, xmax=1, ymin=-1, ymax=1, xsteps=100, ysteps=100, p="x,y", g=None, plot="image", **kwargs):
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
    return image_data(x,y,zgrid,plot,**kwargs)






def parametric_function(fx, fy, tmin=-1, tmax=1, steps=200, p='t', g=None, erange=False, **kwargs):
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
    return xy_data(xdatas, ydatas, label=labels, **kwargs)



class plot_style_cycle(dict):

    iterators = {}

    def __init__(self, **kwargs):
        """
        Supply keyword arguments that would be sent to pylab.plot(), except
        as a list so there is some order to follow. For example:

        style = plot_style_cycle(color=['k','r','b'], marker='o')

        """
        # make sure everything is iterable
        for key in kwargs:
            if not getattr(kwargs[key],'__iter__',False): kwargs[key] = [kwargs[key]]

        # The base class is a dictionary, so update our own elements!
        self.update(kwargs)

        # create the auxiliary iterator dictionary
        self.reset()

    def next(self):
        """
        Returns the next dictionary of styles to send to plot as kwargs.
        For example:

        pylab.plot([1,2,3],[1,2,1], **style.next())
        """
        s = {}
        for key   in self.iterators.keys():
            s[key] = self.iterators[key].next()
        return s

    def reset(self):
        """
        Resets the style cycle.
        """
        for key in self.keys(): self.iterators[key] = _itertools.cycle(self[key])
        return self

