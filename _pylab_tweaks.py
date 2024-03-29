import pylab as _pylab
import numpy as _numpy
import matplotlib as _mpl
import wx as _wx
import time as _time
from matplotlib.font_manager import FontProperties as _FontProperties
import os as _os

import _dialogs
import _functions as _fun
import _pylab_colorslider as _pc
import _plot


line_attributes = ["linestyle","linewidth","color","marker","markersize","markerfacecolor","markeredgewidth","markeredgecolor"]

image_undo_list = []

def add_text(text, x=0.01, y=0.01, axes="gca", draw=True, **kwargs):
    """
    Adds text to the axes at the specified position.

    **kwargs go to the axes.text() function.
    """
    if axes=="gca": axes = _pylab.gca()
    axes.text(x, y, text, transform=axes.transAxes, **kwargs)
    if draw: _pylab.draw()

def auto_zoom(zoomx=1, zoomy=1, axes="gca", x_space=0.04, y_space=0.04, draw=True):
    if axes=="gca": axes = _pylab.gca()

    a = axes

    # get all the lines
    lines = a.get_lines()

    # get the current limits, in case we're not zooming one of the axes.
    x1, x2 = a.get_xlim()
    y1, y2 = a.get_ylim()

    xdata = []
    ydata = []
    for n in range(0,len(lines)):
        # store this line's data

        # build up a huge data array
        if isinstance(lines[n], _mpl.lines.Line2D):
            x, y = lines[n].get_data()

            for n in range(len(x)):
                # if we're not zooming x and we're in range, append
                if not zoomx and x[n] >= x1 and x[n] <= x2:
                    xdata.append(x[n])
                    ydata.append(y[n])

                elif not zoomy and y[n] >= y1 and y[n] <= y2:
                    xdata.append(x[n])
                    ydata.append(y[n])

                elif zoomy and zoomx:
                    xdata.append(x[n])
                    ydata.append(y[n])

    if len(xdata):
        xmin = min(xdata)
        xmax = max(xdata)
        ymin = min(ydata)
        ymax = max(ydata)


        # we want a 3% white space boundary surrounding the data in our plot
        # so set the range accordingly
        if zoomx: a.set_xlim(xmin-x_space*(xmax-xmin), xmax+x_space*(xmax-xmin))
        if zoomy: a.set_ylim(ymin-y_space*(ymax-ymin), ymax+y_space*(ymax-ymin))

        if draw: _pylab.draw()

    else:
        return



def click_estimate_slope():
    """
    Takes two clicks and returns the slope.

    Right-click aborts.
    """

    c1 = ginput()
    if len(c1)==0:
        raise_pyshell()
        return None

    c2 = ginput()
    if len(c2)==0:
        raise_pyshell()
        return None

    raise_pyshell()

    return (c1[0][1]-c2[0][1])/(c1[0][0]-c2[0][0])

def click_estimate_curvature():
    """
    Takes two clicks and returns the curvature, assuming the first click
    was the minimum of a parabola and the second was some other point.

    Returns the second derivative of the function giving this parabola.

    Right-click aborts.
    """

    c1 = ginput()
    if len(c1)==0:
        raise_pyshell()
        return None

    c2 = ginput()
    if len(c2)==0:
        raise_pyshell()
        return None

    raise_pyshell()

    return 2*(c2[0][1]-c1[0][1])/(c2[0][0]-c1[0][0])**2

def click_estimate_difference():
    """
    Takes two clicks and returns the difference vector [dx, dy].

    Right-click aborts.
    """

    c1 = ginput()
    if len(c1)==0:
        raise_pyshell()
        return None

    c2 = ginput()
    if len(c2)==0:
        raise_pyshell()
        return None

    raise_pyshell()

    return [c2[0][0]-c1[0][0], c2[0][1]-c1[0][1]]

def close_sliders():

    # get the list of open windows
    w = _wx.GetTopLevelWindows()

    # loop over them and close all the type colorsliderframe's
    for x in w:
        # if it's of the right class
        if x.__class__.__name__ == _pc._pcf.ColorSliderFrame.__name__:
            x.Close()


def differentiate_shown_data(neighbors=1, fyname=1, **kwargs):
    """
    Differentiates the data visible on the specified axes using
    fun.derivative_fit() (if neighbors > 0), and derivative() otherwise.
    Modifies the visible data using manipulate_shown_data(**kwargs)
    """

    if neighbors:
        def D(x,y): return _fun.derivative_fit(x,y,neighbors)
    else:
        def D(x,y): return _fun.derivative(x,y)

    if fyname==1: fyname = str(neighbors)+'-neighbor D'

    manipulate_shown_data(D, fxname=None, fyname=fyname, **kwargs)

def integrate_shown_data(scale=1, fyname=1, autozero=0, **kwargs):
    """
    Numerically integrates the data visible on the current/specified axes using
    scale*fun.integrate_data(x,y). Modifies the visible data using
    manipulate_shown_data(**kwargs)
    
    autozero is the number of data points used to estimate the background
    for subtraction. If autozero = 0, no background subtraction is performed.    
    """

    def I(x,y):
        xout, iout = _fun.integrate_data(x,y, autozero=autozero)
        print "Total =", scale*iout[-1]
        return xout, scale*iout

    if fyname==1: fyname = str(scale)+" * Integral"

    manipulate_shown_data(I, fxname=None, fyname=fyname, **kwargs)



def image_sliders(image="top", colormap="_last"):
    close_sliders()
    _pc.GuiColorMap(image, colormap)


def _old_format_figure(figure='gcf', tall=False, draw=True, figheight=10.5, figwidth=8.0, **kwargs):
    """

    This formats the figure in a compact way with (hopefully) enough useful
    information for printing large data sets. Used mostly for line and scatter
    plots with long, information-filled titles.

    Chances are somewhat slim this will be ideal for you but it very well might
    and is at least a good starting point.

    """

    for k in kwargs.keys(): print "NOTE: '"+k+"' is not an option used by spinmob.tweaks.format_figure()"

    if figure == 'gcf': figure = _pylab.gcf()

    # get the window of the figure
    figure_window = get_figure_window(figure)
    #figure_window.SetPosition([0,0])

    # assume two axes means twinx
    window_width=645
    legend_position=1.01

    # set the size of the window
    if(tall): figure_window.SetSize([window_width,680])
    else:     figure_window.SetSize([window_width,520])

    figure.set_figwidth(figwidth)
    figure.set_figheight(figheight)

    # first, find overall bounds of the figure.
    ymin = 1.0
    ymax = 0.0
    xmin = 1.0
    xmax = 0.0
    for axes in figure.get_axes():
        (x,y,dx,dy) = axes.get_position().bounds
        if y    < ymin: ymin = y
        if y+dy > ymax: ymax = y+dy
        if x    < xmin: xmin = x
        if x+dx > xmax: xmax = x+dx

    # Fraction of the figure's height to use for all the plots.
    if tall: h = 0.7
    else:    h = 0.5
    
    # buffers around edges
    bt = 0.07
    bb = 0.05
    w  = 0.55
    bl = 0.20    
    
    xscale =  w        / (xmax-xmin)
    yscale = (h-bt-bb) / (ymax-ymin)
    
    for axes in figure.get_axes():

        (x,y,dx,dy) = axes.get_position().bounds
        
        y  = 1-h+bb + (y-ymin)*yscale
        dy = dy * yscale
            
        x  = bl
        dx = dx * xscale

        axes.set_position([x,y,dx,dy])

        # set the position of the legend
        _pylab.axes(axes) # set the current axes
        if len(axes.lines)>0:
            _pylab.legend(loc=[legend_position, 0], borderpad=0.02, prop=_FontProperties(size=7))

        # set the label spacing in the legend
        if axes.get_legend():
            if tall: axes.get_legend().labelsep = 0.007
            else:    axes.get_legend().labelsep = 0.01
            axes.get_legend().set_visible(1)

        # set up the title label
        axes.title.set_horizontalalignment('right')
        axes.title.set_size(8)
        axes.title.set_position([1.4,1.02])
        axes.title.set_visible(1)
        #axes.yaxis.label.set_horizontalalignment('center')
        #axes.xaxis.label.set_horizontalalignment('center')

    # get the shell window
    if draw:
        shell_window = get_pyshell()
        figure_window.Raise()
        shell_window.Raise()

def format_figure(figure='gcf', tall=False, draw=True, **kwargs):
    """
    This formats the figure in a compact way with (hopefully) enough useful
    information for printing large data sets. Used mostly for line and scatter
    plots with long, information-filled titles.

    Chances are somewhat slim this will be ideal for you but it very well might
    and is at least a good starting point.
    """

    for k in kwargs.keys(): print "NOTE: '"+k+"' is not an option used by spinmob.tweaks.format_figure()"

    if figure == 'gcf': figure = _pylab.gcf()

    # get the window of the figure
    figure_window = get_figure_window(figure)
    #figure_window.SetPosition([0,0])

    # assume two axes means twinx
    window_width=figure_window.GetSize()[0]
    legend_position=1.01

    # set the size of the window
    if(tall): figure_window.SetSize([window_width,window_width*680./645.])
    else:     figure_window.SetSize([window_width,window_width*520./645.])

    # first, find overall bounds of all axes.
    ymin = 1.0
    ymax = 0.0
    xmin = 1.0
    xmax = 0.0
    for axes in figure.get_axes():
        (x,y,dx,dy) = axes.get_position().bounds
        if y    < ymin: ymin = y
        if y+dy > ymax: ymax = y+dy
        if x    < xmin: xmin = x
        if x+dx > xmax: xmax = x+dx

    # Fraction of the figure's width and height to use for all the plots.
    w  = 0.55
    if tall: h = 0.77
    else:    h = 0.75
    
    # buffers on left and bottom edges
    bb = 0.12
    bl = 0.12    
    
    xscale = w / (xmax-xmin)
    yscale = h / (ymax-ymin)
    
    current_axes = _pylab.gca()
    for axes in figure.get_axes():

        (x,y,dx,dy) = axes.get_position().bounds
        y  = bb + (y-ymin)*yscale
        dy = dy * yscale
            
        x  = bl + (x-xmin)*xscale
        dx = dx * xscale

        axes.set_position([x,y,dx,dy])

        # set the position of the legend
        _pylab.axes(axes) # set the current axes
        if len(axes.lines)>0:
            _pylab.legend(loc=[legend_position, 0], borderpad=0.02, prop=_FontProperties(size=7))

        # set the label spacing in the legend
        if axes.get_legend():
            if tall: axes.get_legend().labelsep = 0.007
            else:    axes.get_legend().labelsep = 0.01
            axes.get_legend().set_visible(1)

        # set up the title label
        axes.title.set_horizontalalignment('right')
        axes.title.set_size(8)
        axes.title.set_position([1.5,1.02])
        axes.title.set_visible(1)
        #axes.yaxis.label.set_horizontalalignment('center')
        #axes.xaxis.label.set_horizontalalignment('center')

    _pylab.axes(current_axes)

    # get the shell window
    if draw:
        shell_window = get_pyshell()
        figure_window.Raise()
        shell_window.Raise()

def impose_legend_limit(limit=30, axes="gca", **kwargs):
    """
    This will erase all but, say, 30 of the legend entries and remake the legend.
    You'll probably have to move it back into your favorite position at this point.
    """
    if axes=="gca": axes = _pylab.gca()

    # make these axes current
    _pylab.axes(axes)

    # loop over all the lines
    for n in range(0,len(axes.lines)):
        if n >  limit-1 and not n==len(axes.lines)-1: axes.lines[n].set_label("_nolegend_")
        if n == limit-1 and not n==len(axes.lines)-1: axes.lines[n].set_label("...")

    _pylab.legend(**kwargs)


def image_autozoom(axes="gca"):

    if axes=="gca": axes = _pylab.gca()

    # get the extent
    extent = axes.images[0].get_extent()

    # rezoom us
    axes.set_xlim(extent[0],extent[1])
    axes.set_ylim(extent[2],extent[3])

    _pylab.draw()

def image_coarsen(xlevel=0, ylevel=0, image="auto", method='average'):
    """
    This will coarsen the image data by binning each xlevel+1 along the x-axis
    and each ylevel+1 points along the y-axis
    
    type can be 'average', 'min', or 'max'
    """
    if image == "auto": image = _pylab.gca().images[0]

    Z = _numpy.array(image.get_array())

    # store this image in the undo list
    global image_undo_list
    image_undo_list.append([image, Z])
    if len(image_undo_list) > 10: image_undo_list.pop(0)

    # images have transposed data
    image.set_array(_fun.coarsen_matrix(Z, ylevel, xlevel, method))

    # update the plot
    _pylab.draw()


def image_neighbor_smooth(xlevel=0.2, ylevel=0.2, image="auto"):
    """
    This will bleed nearest neighbor pixels into each other with
    the specified weight factors.
    """
    if image == "auto": image = _pylab.gca().images[0]

    Z = _numpy.array(image.get_array())

    # store this image in the undo list
    global image_undo_list
    image_undo_list.append([image, Z])
    if len(image_undo_list) > 10: image_undo_list.pop(0)

    # get the diagonal smoothing level (eliptical, and scaled down by distance)
    dlevel = ((xlevel**2+ylevel**2)/2.0)**(0.5)

    # don't touch the first column
    new_Z = [Z[0]*1.0]

    for m in range(1,len(Z)-1):
        new_Z.append(Z[m]*1.0)
        for n in range(1,len(Z[0])-1):
            new_Z[-1][n] = (Z[m,n] + xlevel*(Z[m+1,n]+Z[m-1,n]) + ylevel*(Z[m,n+1]+Z[m,n-1])   \
                                   + dlevel*(Z[m+1,n+1]+Z[m-1,n+1]+Z[m+1,n-1]+Z[m-1,n-1])   )  \
                                   / (1.0+xlevel*2+ylevel*2 + dlevel*4)

    # don't touch the last column
    new_Z.append(Z[-1]*1.0)

    # images have transposed data
    image.set_array(_numpy.array(new_Z))

    # update the plot
    _pylab.draw()

def image_undo():
    """
    Undoes the last coarsen or smooth command.
    """
    if len(image_undo_list) <= 0:
        print "no undos in memory"
        return

    [image, Z] = image_undo_list.pop(-1)
    image.set_array(Z)
    _pylab.draw()


def image_set_aspect(aspect=1.0, axes="gca"):
    """
    sets the aspect ratio of the current zoom level of the imshow image
    """
    if axes is "gca": axes = _pylab.gca()

    # make sure it's not in "auto" mode
    if type(axes.get_aspect()) == str: axes.set_aspect(1.0)

    _pylab.draw() # this makes sure the window_extent is okay
    axes.set_aspect(aspect*axes.get_aspect()*axes.get_window_extent().width/axes.get_window_extent().height)
    _pylab.draw()


def image_set_extent(x=None, y=None, axes="gca"):
    """
    Set's the first image's extent, then redraws.

    Examples:
    x = [1,4]
    y = [33.3, 22]
    """
    if axes == "gca": axes = _pylab.gca()

    # get the current plot limits
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()

    # get the old extent
    extent = axes.images[0].get_extent()

    # calculate the fractional extents
    x0     = extent[0]
    y0     = extent[2]
    xwidth = extent[1]-x0
    ywidth = extent[3]-y0
    frac_x1 = (xlim[0]-x0)/xwidth
    frac_x2 = (xlim[1]-x0)/xwidth
    frac_y1 = (ylim[0]-y0)/ywidth
    frac_y2 = (ylim[1]-y0)/ywidth


    # set the new
    if not x == None:
        extent[0] = x[0]
        extent[1] = x[1]
    if not y == None:
        extent[2] = y[0]
        extent[3] = y[1]


    # get the new zoom window
    x0     = extent[0]
    y0     = extent[2]
    xwidth = extent[1]-x0
    ywidth = extent[3]-y0

    x1 = x0 + xwidth*frac_x1
    x2 = x0 + xwidth*frac_x2
    y1 = y0 + ywidth*frac_y1
    y2 = y0 + ywidth*frac_y2

    # set the extent
    axes.images[0].set_extent(extent)

    # rezoom us
    axes.set_xlim(x1,x2)
    axes.set_ylim(y1,y2)

    # draw
    image_set_aspect(1.0)


def image_scale(xscale=1.0, yscale=1.0, axes="gca"):
    """
    Scales the image extent.
    """
    if axes == "gca": axes = _pylab.gca()

    e = axes.images[0].get_extent()
    x1 = e[0]*xscale
    x2 = e[1]*xscale
    y1 = e[2]*yscale
    y2 = e[3]*yscale

    image_set_extent([x1,x2],[y1,y2], axes)



def image_click_xshift(axes = "gca"):
    """
    Takes a starting and ending point, then shifts the image y by this amount
    """
    if axes == "gca": axes = _pylab.gca()

    try:
        p1 = ginput()
        p2 = ginput()

        xshift = p2[0][0]-p1[0][0]

        e = axes.images[0].get_extent()

        e[0] = e[0] + xshift
        e[1] = e[1] + xshift

        axes.images[0].set_extent(e)

        _pylab.draw()
    except:
        print "whoops"

def image_click_yshift(axes = "gca"):
    """
    Takes a starting and ending point, then shifts the image y by this amount
    """
    if axes == "gca": axes = _pylab.gca()

    try:
        p1 = ginput()
        p2 = ginput()

        yshift = p2[0][1]-p1[0][1]

        e = axes.images[0].get_extent()

        e[2] = e[2] + yshift
        e[3] = e[3] + yshift

        axes.images[0].set_extent(e)

        _pylab.draw()
    except:
        print "whoops"

def image_shift(xshift=0, yshift=0, axes="gca"):
    """
    This will shift an image to a new location on x and y.
    """

    if axes=="gca": axes = _pylab.gca()

    e = axes.images[0].get_extent()

    e[0] = e[0] + xshift
    e[1] = e[1] + xshift
    e[2] = e[2] + yshift
    e[3] = e[3] + yshift

    axes.images[0].set_extent(e)

    _pylab.draw()



def image_set_clim(zmin=None, zmax=None, axes="gca"):
    """
    This will set the clim (range) of the colorbar.
    
    Setting zmin or zmax to None will not change them.
    Setting zmin or zmax to "auto" will auto-scale them to include all the data.
    """
    if axes=="gca": axes=_pylab.gca()

    image = axes.images[0]

    if zmin=='auto': zmin = _numpy.min(image.get_array())
    if zmax=='auto': zmax = _numpy.max(image.get_array())

    if zmin==None: zmin = image.get_clim()[0]
    if zmax==None: zmax = image.get_clim()[1]

    image.set_clim(zmin, zmax)

    _pylab.draw()

def image_ubertidy(figure="gcf", aspect=1.0, fontsize=18, fontweight='bold', fontname='Arial', ylabel_pad=0.007, xlabel_pad=0.010, colorlabel_pad=0.1, borderwidth=3.0, tickwidth=2.0, window_size=(550,500)):

    if figure=="gcf": figure = _pylab.gcf()

    # do this to both axes
    for a in figure.axes:
        _pylab.axes(a)

        # remove the labels
        a.set_title("")
        a.set_xlabel("")
        a.set_ylabel("")

        # thicken the border
        # we want thick axis lines
        a.spines['top'].set_linewidth(borderwidth)
        a.spines['left'].set_linewidth(borderwidth)
        a.spines['bottom'].set_linewidth(borderwidth)
        a.spines['right'].set_linewidth(borderwidth)
        a.set_frame_on(True) # adds a thick border to the colorbar

        # these two cover the main plot
        _pylab.xticks(fontsize=fontsize, fontweight=fontweight, fontname=fontname)
        _pylab.yticks(fontsize=fontsize, fontweight=fontweight, fontname=fontname)

        # thicken the tick lines
        for l in a.get_xticklines(): l.set_markeredgewidth(tickwidth)
        for l in a.get_yticklines(): l.set_markeredgewidth(tickwidth)

    # set the aspect and window size
    _pylab.axes(figure.axes[0])
    image_set_aspect(aspect)
    get_figure_window().SetSize(window_size)

    # we want to give the labels some breathing room (1% of the data range)
    for label in _pylab.xticks()[1]: label.set_y(-xlabel_pad)
    for label in _pylab.yticks()[1]: label.set_x(-ylabel_pad)

    # need to draw to commit the changes up to this point. Annoying.
    _pylab.draw()


    # get the bounds of the first axes and come up with corresponding bounds
    # for the colorbar
    a1 = _pylab.gca()
    b  = a1.get_position()
    aspect = figure.axes[1].get_aspect()

    pos = []
    pos.append(b.x0+b.width+0.02)   # lower left x
    pos.append(b.y0)                # lower left y
    pos.append(b.height/aspect)     # width
    pos.append(b.height)            # height

    # switch to the colorbar axes
    _pylab.axes(figure.axes[1])
    _pylab.gca().set_position(pos)

    for label in _pylab.yticks()[1]: label.set_x(1+colorlabel_pad)


    # switch back to the main axes
    _pylab.axes(figure.axes[0])

    _pylab.draw()




def is_a_number(s):
    try: eval(s); return 1
    except:       return 0




def manipulate_shown_data(f, input_axes="gca", output_axes=None, fxname=1, fyname=1, clear=1, pause=False, **kwargs):
    """
    Loops over the visible data on the specified axes and modifies it based on
    the function f(xdata, ydata), which must return new_xdata, new_ydata

    input_axes  which axes to pull the data from
    output_axes which axes to dump the manipulated data (None for new figure)

    fxname      the name of the function on x
    fyname      the name of the function on y
                1 means "use f.__name__"
                0 or None means no change.
                otherwise specify a string

    **kwargs are sent to axes.plot
    """

    # get the axes
    if input_axes == "gca": a1 = _pylab.gca()
    else:                   a1 = input_axes

    # get the xlimits
    xmin, xmax = a1.get_xlim()

    # get the name to stick on the x and y labels
    if fxname==1: fxname = f.__name__
    if fyname==1: fyname = f.__name__

    # get the output axes
    if output_axes == None:
        _pylab.figure(a1.figure.number+1)
        a2 = _pylab.axes()
    else:
        a2 = output_axes

    if clear: a2.clear()

    # loop over the data
    for line in a1.get_lines():
        if isinstance(line, _mpl.lines.Line2D):
            x, y = line.get_data()
            x, y, e = _fun.trim_data(x,y,None,[xmin,xmax])
            new_x, new_y = f(x,y)
            _plot.xy.data(new_x,new_y, clear=0, label=line.get_label(), draw=pause, **kwargs)
            if pause:
                format_figure()
                raise_pyshell()
                raw_input("<enter> ")

    # set the labels and title.
    if fxname in [0,None]:  a2.set_xlabel(a1.get_xlabel())
    else:                   a2.set_xlabel(fxname+"[ "+a1.get_xlabel()+" ]")

    if fyname in [0,None]:  a2.set_ylabel(a1.get_ylabel())
    else:                   a2.set_ylabel(fyname+"[ "+a1.get_ylabel()+" ]")

    if not kwargs.has_key('title'): 
        a2.title.set_text(get_pyshell_command() + "\n" + a1.title.get_text())

    _pylab.draw()

def manipulate_shown_xdata(fx, fxname=1, **kwargs):
    """
    This defines a function f(xdata,ydata) returning fx(xdata), ydata and
    runs manipulate_shown_data() with **kwargs sent to this. See
    manipulate_shown_data() for more info.
    """
    def f(x,y): return fx(x), y
    f.__name__ = fx.__name__
    manipulate_shown_data(f, fxname=fxname, fyname=None, **kwargs)

def manipulate_shown_ydata(fy, fyname=1, **kwargs):
    """
    This defines a function f(xdata,ydata) returning xdata, fy(ydata) and
    runs manipulate_shown_data() with **kwargs sent to this. See
    manipulate_shown_data() for more info.
    """
    def f(x,y): return x, fy(y)
    f.__name__ = fy.__name__
    manipulate_shown_data(f, fxname=None, fyname=fyname, **kwargs)



def shift(xshift=0, yshift=0, progressive=0, axes="gca"):
    """

    This function adds an artificial offset to the lines.

    yshift          amount to shift vertically
    xshift          amount to shift horizontally
    axes="gca"      axes to do this on, "gca" means "get current axes"
    progressive=0   progressive means each line gets more offset
                    set to 0 to shift EVERYTHING

    """

    if axes=="gca": axes = _pylab.gca()

    # get the lines from the plot
    lines = axes.get_lines()

    # loop over the lines and trim the data
    for m in range(0,len(lines)):
        if isinstance(lines[m], _mpl.lines.Line2D):
            # get the actual data values
            xdata = _numpy.array(lines[m].get_xdata())
            ydata = _numpy.array(lines[m].get_ydata())

            # add the offset
            if progressive:
                xdata += m*xshift
                ydata += m*yshift
            else:
                xdata += xshift
                ydata += yshift

            # update the data for this line
            lines[m].set_data(xdata, ydata)

    # zoom to surround the data properly

    auto_zoom()



def reverse_draw_order(axes="current"):
    """

    This function takes the graph and reverses the draw order.

    """

    if axes=="current": axes = _pylab.gca()

    # get the lines from the plot
    lines = axes.get_lines()

    # reverse the order
    lines.reverse()

    for n in range(0, len(lines)):
        if isinstance(lines[n], _mpl.lines.Line2D):
            axes.lines[n]=lines[n]

    _pylab.draw()


def scale_x(scale, axes="current"):
    """

    This function scales lines horizontally.

    """

    if axes=="current": axes = _pylab.gca()

    # get the lines from the plot
    lines = axes.get_lines()

    # loop over the lines and trim the data
    for line in lines:
        if isinstance(line, _mpl.lines.Line2D):
            line.set_xdata(_pylab.array(line.get_xdata())*scale)

    # update the title
    title = axes.title.get_text()
    title += ", x_scale="+str(scale)
    axes.title.set_text(title)

    # zoom to surround the data properly
    auto_zoom()

def scale_y(scale, axes="current", lines="all"):
    """

    This function scales lines vertically.
    You can specify a line index, such as lines=0 or lines=[1,2,4]

    """

    if axes=="current": axes = _pylab.gca()

    # get the lines from the plot
    lines = axes.get_lines()

    # loop over the lines and trim the data
    for line in lines:
        if isinstance(line, _mpl.lines.Line2D):
            line.set_ydata(_pylab.array(line.get_ydata())*scale)

    # update the title
    title = axes.title.get_text()
    if not title == "":
        title += ", y_scale="+str(scale)
        axes.title.set_text(title)


    # zoom to surround the data properly
    auto_zoom()

def scale_y_universal(average=[1,10], axes="current"):
    """

    This function scales lines vertically.

    average=[1,10]    indices of average universal point

    """

    if axes=="current": axes = _pylab.gca()

    # get the lines from the plot
    lines = axes.get_lines()

    # loop over the lines and trim the data
    for m in range(0,len(lines)):
        if isinstance(lines[m], _mpl.lines.Line2D):

            # get the actual data values
            xdata = lines[m].get_xdata()
            ydata = lines[m].get_ydata()

            # figure out the scaling factor
            s=0
            for n in range(average[0], average[1]+1): s += ydata[n]
            scale = 1.0*s/(average[1]-average[0]+1.0)

            # loop over the ydata to scale it
            for n in range(0,len(ydata)): ydata[n] = ydata[n]/scale

            # update the data for this line
            lines[m].set_data(xdata, ydata)

    # update the title
    title = axes.title.get_text()
    title += ", universal scale"
    axes.title.set_text(title)

    # zoom to surround the data properly
    auto_zoom()

def set_title(axes="current", title=""):
    if axes=="current": axes = _pylab.gca()
    axes.title.set_text(title)
    _pylab.draw()




def set_xrange(xmin="same", xmax="same", axes="gca"):
    if axes == "gca": axes = _pylab.gca()

    xlim = axes.get_xlim()

    if xmin == "same": xmin = xlim[0]
    if xmax == "same": xmax = xlim[1]

    axes.set_xlim(xmin,xmax)
    _pylab.draw()

def set_yrange(ymin="same", ymax="same", axes="gca"):
    if axes == "gca": axes = _pylab.gca()

    ylim = axes.get_ylim()

    if ymin == "same": ymin = ylim[0]
    if ymax == "same": ymax = ylim[1]

    axes.set_ylim(ymin,ymax)
    _pylab.draw()


def set_yticks(start, step, axes="gca"):
    """
    This will generate a tick array and apply said array to the axis
    """
    if axes=="gca": axes = _pylab.gca()

    # first get one of the tick label locations
    xposition = axes.yaxis.get_ticklabels()[0].get_position()[0]

    # get the bounds
    ymin, ymax = axes.get_ylim()

    # get the starting tick
    nstart = int(_pylab.floor((ymin-start)/step))
    nstop  = int(_pylab.ceil((ymax-start)/step))
    ticks = []
    for n in range(nstart,nstop+1): ticks.append(start+n*step)

    axes.set_yticks(ticks)

    # set the x-position
    for t in axes.yaxis.get_ticklabels():
        x, y = t.get_position()
        t.set_position((xposition, y))

    _pylab.draw()

def set_xticks(start, step, axes="gca"):
    """
    This will generate a tick array and apply said array to the axis
    """
    if axes=="gca": axes = _pylab.gca()

    # first get one of the tick label locations
    yposition = axes.xaxis.get_ticklabels()[0].get_position()[1]

    # get the bounds
    xmin, xmax = axes.get_xlim()

    # get the starting tick
    nstart = int(_pylab.floor((xmin-start)/step))
    nstop  = int(_pylab.ceil((xmax-start)/step))
    ticks = []
    for n in range(nstart,nstop+1): ticks.append(start+n*step)

    axes.set_xticks(ticks)

    # set the y-position
    for t in axes.xaxis.get_ticklabels():
        x, y = t.get_position()
        t.set_position((x, yposition))

    _pylab.draw()


def invert(axes="current"):
    """

    inverts the plot

    """
    if axes=="current": axes = _pylab.gca()
    scale_y(-1,axes)

def set_markers(marker="o", axes="current"):
    if axes == "current": axes = _pylab.gca()
    set_all_line_attributes("marker", marker, axes)

def set_all_line_attributes(attribute="lw", value=2, axes="current", refresh=True):
    """

    This function sets all the specified line attributes.

    """

    if axes=="current": axes = _pylab.gca()

    # get the lines from the plot
    lines = axes.get_lines()

    # loop over the lines and trim the data
    for line in lines:
        if isinstance(line, _mpl.lines.Line2D):
            _pylab.setp(line, attribute, value)

    # update the plot
    if refresh: _pylab.draw()

def set_line_attribute(line=-1, attribute="lw", value=2, axes="current", refresh=True):
    """

    This function sets all the specified line attributes.

    """

    if axes=="current": axes = _pylab.gca()

    # get the lines from the plot
    line = axes.get_lines()[-1]
    _pylab.setp(line, attribute, value)

    # update the plot
    if refresh: _pylab.draw()

def smooth_line(line, smoothing=1, trim=True, draw=True):
    """

    This takes a line instance and smooths its data with nearest neighbor averaging.

    """

    # get the actual data values
    xdata = list(line.get_xdata())
    ydata = list(line.get_ydata())

    _fun.smooth_array(ydata, smoothing)

    if trim:
        for n in range(0, smoothing):
            xdata.pop(0); xdata.pop(-1)
            ydata.pop(0); ydata.pop(-1)

    # don't do anything if we don't have any data left
    if len(ydata) == 0:
        print "There's nothing left in "+str(line)+"!"
    else:
        # otherwise set the data with the new arrays
        line.set_data(xdata, ydata)

    # we refresh in real time for giggles
    if draw: _pylab.draw()


def coarsen_line(line, coarsen=1, draw=True):
    """

    This takes a line instance and smooths its data with nearest neighbor averaging.

    """

    # get the actual data values
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    xdata = _fun.coarsen_array(xdata, coarsen)
    ydata = _fun.coarsen_array(ydata, coarsen)

    # don't do anything if we don't have any data left
    if len(ydata) == 0:
        print "There's nothing left in "+str(line)+"!"
    else:
        # otherwise set the data with the new arrays
        line.set_data(xdata, ydata)

    # we refresh in real time for giggles
    if draw: _pylab.draw()

def smooth_selected_trace(trim=True, axes="gca"):
    """

    This cycles through all the lines in a set of axes, highlighting them,
    and asking for how much you want to smooth by (0 or <enter> is valid)

    """

    if axes=="gca": axes = _pylab.gca()

    # get all the lines
    lines = axes.get_lines()

    for line in lines:

        if isinstance(line, _mpl.lines.Line2D):
            # first highlight it
            fatten_line(line)
            raise_figure_window()
            raise_pyshell()

            # get the smoothing factor
            ready = 0
            while not ready:
                response = raw_input("Smoothing Factor (<enter> to skip): ")
                try:
                    int(response)
                    ready=1
                except:
                    if response=="\n": ready = 1
                    else:              print "No!"

            if not response == "\n":
                smooth_line(line, int(response), trim)

            # return the line to normal
            unfatten_line(line)

def smooth_all_traces(smoothing=1, trim=True, axes="gca"):
    """

    This function does nearest-neighbor smoothing of the data

    """
    if axes=="gca": axes=_pylab.gca()

    # get the lines from the plot
    lines = axes.get_lines()

    # loop over the lines and trim the data
    for line in lines:
        if isinstance(line, _mpl.lines.Line2D):
            smooth_line(line, smoothing, trim, draw=False)
    _pylab.draw()

def coarsen_all_traces(coarsen=1, axes="all", figure=None):
    """

    This function does nearest-neighbor smoothing of the data

    """
    if axes=="gca": axes=_pylab.gca()
    if axes=="all":
        if not figure: f = _pylab.gcf()
        axes = f.axes
        
    if not _fun.is_iterable(axes): axes = [axes]

    for a in axes:    
        # get the lines from the plot
        lines = a.get_lines()
    
        # loop over the lines and trim the data
        for line in lines:
            if isinstance(line, _mpl.lines.Line2D):
                coarsen_line(line, coarsen, draw=False)
    _pylab.draw()

def line_math(fx=None, fy=None, axes='gca'):
    """
    applies function fx to all xdata and fy to all ydata.
    """

    if axes=='gca': axes = _pylab.gca()

    lines = axes.get_lines()

    for line in lines:
        if isinstance(line, _mpl.lines.Line2D):
            xdata, ydata = line.get_data()
            if not fx==None: xdata = fx(xdata)
            if not fy==None: ydata = fy(ydata)
            line.set_data(xdata,ydata)

    _pylab.draw()

def trim(xmin="auto", xmax="auto", ymin="auto", ymax="auto", axes="current"):
    """

    This function just removes all data from the plots that
    is outside of the [xmin,xmax,ymin,ymax] range.

    "auto" means "determine from the current axes's range"

    """

    if axes=="current": axes = _pylab.gca()

    # if trim_visible is true, use the current plot's limits
    if xmin=="auto": (xmin, dummy) = axes.get_xlim()
    if xmax=="auto": (dummy, xmax) = axes.get_xlim()
    if ymin=="auto": (ymin, dummy) = axes.get_ylim()
    if ymax=="auto": (dummy, ymax) = axes.get_ylim()


    # get the lines from the plot
    lines = axes.get_lines()

    # loop over the lines and trim the data
    for line in lines:
        # get the actual data values
        old_xdata = line.get_xdata()
        old_ydata = line.get_ydata()

        # loop over the xdata and trim if it's outside the range
        new_xdata = []
        new_ydata = []
        for n in range(0, len(old_xdata)):
            # if it's in the data range
            if  old_xdata[n] >= xmin and old_xdata[n] <= xmax \
            and old_ydata[n] >= ymin and old_ydata[n] <= ymax:
                # append it to the new x and y data set
                new_xdata.append(old_xdata[n])
                new_ydata.append(old_ydata[n])

        # don't do anything if we don't have any data left
        if len(new_xdata) == 0:
            print "There's nothing left in "+str(line)+"!"
        else:
            # otherwise set the data with the new arrays
            line.set_data(new_xdata, new_ydata)


    # loop over the collections, where the vertical parts of the error bars are stored
    for c in axes.collections:

        # loop over the paths and pop them if they're bad
        for n in range(len(c._paths)-1,-1,-1):

            # loop over the vertices
            naughty = False
            for v in c._paths[n].vertices:
                # if the path contains any vertices outside the trim box, kill it!
                if v[0] < xmin or v[0] > xmax or v[1] < ymin or v[1] > ymax:
                    naughty=True

            # BOOM
            if naughty: c._paths.pop(n)

    # zoom to surround the data properly
    auto_zoom()

def xscale(scale='log'):
    _pylab.xscale(scale)
    _pylab.draw()

def yscale(scale='log'):
    _pylab.yscale(scale)
    _pylab.draw()

def ubertidy(figure="gcf", zoom=True, width=None, height=None, fontsize=15, fontweight='normal', fontname='Arial',
             borderwidth=1.5, tickwidth=1, ticks_point="in", xlabel_pad=0.013, ylabel_pad=0.010, window_size=[550,550]):
    """

    This guy performs the ubertidy from the helper on the first window.
    Currently assumes there is only one set of axes in the window!

    """

    if figure=="gcf": f = _pylab.gcf()
    else:             f = figure

    # first set the size of the window
    f.canvas.Parent.SetSize(window_size)

    for n in range(len(f.axes)):
        # get the axes
        a = f.axes[n]

        # set the current axes
        _pylab.axes(a)

        # we want thick axis lines
        a.spines['top'].set_linewidth(borderwidth)
        a.spines['left'].set_linewidth(borderwidth)
        a.spines['bottom'].set_linewidth(borderwidth)
        a.spines['right'].set_linewidth(borderwidth)

        # get the tick lines in one big list
        xticklines = a.get_xticklines()
        yticklines = a.get_yticklines()

        # set their marker edge width
        _pylab.setp(xticklines+yticklines, mew=tickwidth)


        # set what kind of tickline they are (outside axes)
        if ticks_point=="out":
            for l in xticklines: l.set_marker(_mpl.lines.TICKDOWN)
            for l in yticklines: l.set_marker(_mpl.lines.TICKLEFT)

        # get rid of the top and right ticks
        a.xaxis.tick_bottom()
        a.yaxis.tick_left()

        # we want bold fonts
        _pylab.xticks(fontsize=fontsize, fontweight=fontweight, fontname=fontname)
        _pylab.yticks(fontsize=fontsize, fontweight=fontweight, fontname=fontname)

        # we want to give the labels some breathing room (1% of the data range)
        for label in _pylab.xticks()[1]: label.set_y(-xlabel_pad)
        for label in _pylab.yticks()[1]: label.set_x(-ylabel_pad)

        # get rid of tick label offsets
        #a.ticklabel_format(style='plain')

        # set the position/size of the axis in the window
        p = a.get_position().bounds
        if width:  a.set_position([0.15,p[1],0.15+width*0.5,p[3]])
        p = a.get_position().bounds
        if height: a.set_position([p[0],0.17,p[2],0.17+height*0.5])

        # set the axis labels to empty (so we can add them with a drawing program)
        a.set_title('')
        a.set_xlabel('')
        a.set_ylabel('')

        # kill the legend
        a.legend_ = None

        # zoom!
        if zoom: auto_zoom(axes=a)

def make_inset(figure="current", width=1, height=1):
    """

    This guy makes the figure thick and small, like an inset.
    Currently assumes there is only one set of axes in the window!

    """

    # get the current figure if we're not supplied with one
    if figure == "current": figure = _pylab.gcf()

    # get the window
    w = figure.canvas.GetParent()

    # first set the size of the window
    w.SetSize([220,300])

    # we want thick axis lines
    figure.axes[0].get_frame().set_linewidth(3.0)

    # get the tick lines in one big list
    xticklines = figure.axes[0].get_xticklines()
    yticklines = figure.axes[0].get_yticklines()

    # set their marker edge width
    _pylab.setp(xticklines+yticklines, mew=2.0)

    # set what kind of tickline they are (outside axes)
    for l in xticklines: l.set_marker(_mpl.lines.TICKDOWN)
    for l in yticklines: l.set_marker(_mpl.lines.TICKLEFT)

    # get rid of the top and right ticks
    figure.axes[0].xaxis.tick_bottom()
    figure.axes[0].yaxis.tick_left()

    # we want bold fonts
    _pylab.xticks(fontsize=20, fontweight='bold', fontname='Arial')
    _pylab.yticks(fontsize=20, fontweight='bold', fontname='Arial')

    # we want to give the labels some breathing room (1% of the data range)
    figure.axes[0].xaxis.set_ticklabels([])
    figure.axes[0].yaxis.set_ticklabels([])


    # set the position/size of the axis in the window
    figure.axes[0].set_position([0.1,0.1,0.1+0.7*width,0.1+0.7*height])

    # set the axis labels to empty (so we can add them with a drawing program)
    figure.axes[0].set_title('')
    figure.axes[0].set_xlabel('')
    figure.axes[0].set_ylabel('')

    # set the position of the legend far away
    figure.axes[0].legend=None

    # zoom!
    auto_zoom(figure.axes[0], 0.07, 0.07)


def export_figure(dpi=200, figure="gcf", path="ask"):
    """
    Saves the actual postscript data for the figure.
    """
    if figure=="gcf": figure = _pylab.gcf()

    if path=="ask": path = _dialogs.Save("*.*", default_directory="save_plot_default_directory")

    if path=="":
        print "aborted."
        return

    figure.savefig(path, dpi=dpi)

def save_plot(axes="gca", path="ask"):
    """
    Saves the figure in my own ascii format
    """

    global line_attributes

    # choose a path to save to
    if path=="ask": path = _dialogs.Save("*.plot", default_directory="save_plot_default_directory")

    if path=="":
        print "aborted."
        return

    if not path.split(".")[-1] == "plot": path = path+".plot"

    f = file(path, "w")

    # if no argument was given, get the current axes
    if axes=="gca": axes=_pylab.gca()

    # now loop over the available lines
    f.write("title="  +axes.title.get_text().replace('\n', '\\n')+'\n')
    f.write("xlabel="+axes.xaxis.label.get_text().replace('\n','\\n')+'\n')
    f.write("ylabel="+axes.yaxis.label.get_text().replace('\n','\\n')+'\n')

    for l in axes.lines:
        # write the data header
        f.write("trace=new\n")
        f.write("legend="+l.get_label().replace('\n', '\\n')+"\n")

        for a in line_attributes: f.write(a+"="+str(_pylab.getp(l, a)).replace('\n','')+"\n")

        # get the data
        x = l.get_xdata()
        y = l.get_ydata()

        # loop over the data
        for n in range(0, len(x)): f.write(str(float(x[n])) + " " + str(float(y[n])) + "\n")

    f.close()

def save_figure_raw_data(figure="gcf", **kwargs):
    """
    This will just output an ascii file for each of the traces in the shown figure.

    **kwargs are sent to dialogs.Save()    
    """
    
    # choose a path to save to
    path = _dialogs.Save(**kwargs)
    if path=="": return "aborted."

    # if no argument was given, get the current axes
    if figure=="gcf": figure = _pylab.gcf()

    for n in range(len(figure.axes)):
        a = figure.axes[n]        
        
        for m in range(len(a.lines)):
            l = a.lines[m]
            
            x = l.get_xdata()
            y = l.get_ydata()

            p = _os.path.split(path)
            p = _os.path.join(p[0], "axes" + str(n) + " line" + str(m) + " " + p[1])
            print p
            
            # loop over the data
            f = open(p, 'w')
            for j in range(0, len(x)):
                f.write(str(x[j]) + "\t" + str(y[j]) + "\n")
            f.close()


def load_plot(clear=1, offset=0, axes="gca"):
    # choose a path to load the file from
    path = _dialogs.SingleFile("*.*", default_directory="save_plot_default_directory")

    if path=="": return

    # read the file in
    lines = _fun.read_lines(path)

    # if no argument was given, get the current axes
    if axes=="gca": axes=_pylab.gca()

    # if we're supposed to, clear the plot
    if clear:
        axes.figure.clear()
        _pylab.gca()

    # split by space delimiter and see if the first element is a number
    xdata  = []
    ydata  = []
    line_stuff = []
    legend = []
    title  = 'reloaded plot with no title'
    xlabel = 'x-data with no label'
    ylabel = 'y-data with no label'
    for line in lines:

        s = line.strip().split('=')

        if len(s) > 1: # header stuff

            if s[0].strip() == 'title':
                # set the title of the plot
                title = ""
                for n in range(1,len(s)): title += " "+s[n].replace('\\n', '\n')

            elif s[0].strip() == 'xlabel':
                # set the title of the plot
                xlabel = ""
                for n in range(1,len(s)): xlabel += " "+s[n].replace('\\n', '\n')

            elif s[0].strip() == 'ylabel':
                # set the title of the plot
                ylabel = ""
                for n in range(1,len(s)): ylabel += " "+s[n].replace('\\n', '\n')

            elif s[0].strip() == 'legend':
                l=""
                for n in range(1,len(s)): l += " " + s[n].replace('\\n', '\n')
                legend.append(l)

            elif s[0].strip() == 'trace':
                # if we're on a new plot
                xdata.append([])
                ydata.append([])
                line_stuff.append({})

            elif s[0].strip() in line_attributes:
                line_stuff[-1][s[0].strip()] = s[1].strip()


        else: # data
            s = line.strip().split(' ')
            try:
                float(s[0])
                float(s[1])
                xdata[-1].append(float(s[0]))
                ydata[-1].append(float(s[1])+offset)
            except:
                print "error s=" + str(s)

    for n in range(0, len(xdata)):
        axes.plot(xdata[n], ydata[n])
        l = axes.get_lines()[-1]
        l.set_label(legend[n])
        for key in line_stuff[n]:
            try:    _pylab.setp(l, key, float(line_stuff[n][key]))
            except: _pylab.setp(l, key,       line_stuff[n][key])

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    format_figure(axes.figure)



def get_figure_window(figure='gcf'):
    """
    This will search through the wx windows and return the one containing the figure
    """

    if figure == 'gcf': figure = _pylab.gcf()

    return figure.canvas.GetParent()


def get_pyshell():
    """
    This will search through the wx windows and return the pyshell
    """

    # starting from the top, grab ALL the wx windows available
    w = _wx.GetTopLevelWindows()

    for x in w:
        if type(x) == _wx.py.shell.ShellFrame or type(x) == _wx.py.crust.CrustFrame: return x

    return False

def get_pyshell_command(n=0):
    """
    Returns a string of the n'th previous pyshell command.
    """
    if n: return str(get_pyshell().shell.history[n-1])
    else: return str(get_pyshell().shell.GetText().split('\n>>> ')[-1].split('\n')[0].strip())

def raise_figure_window(figure='gcf'):
    get_figure_window(figure).Raise()

def raise_pyshell():
    get_pyshell().Raise()

def modify_legend(axes="gca"):
    # get the axes
    if axes=="gca": axes = _pylab.gca()

    # get the lines
    lines = axes.get_lines()

    # loop over the lines
    for line in lines:
        if isinstance(line, _mpl.lines.Line2D):

            # highlight the line
            fatten_line(line)

            # get the label (from the legend)
            label = line.get_label()

            print label

            new_label = raw_input("New Label: ")
            if new_label == "q" or new_label == "quit":
                unfatten_line(line)
                return

            if not new_label == "\n": line.set_label(new_label)

            unfatten_line(line)
            format_figure()


def fatten_line(line, william_fatner=2.0):
    size  = line.get_markersize()
    width = line.get_linewidth()
    line.set_markersize(size*william_fatner)
    line.set_linewidth(width*william_fatner)
    _pylab.draw()

def unfatten_line(line, william_fatner=0.5):
    fatten_line(line, william_fatner)


def legend(location='best', fontsize=16, axes="gca"):
    if axes=="gca": axes = _pylab.gca()

    axes.legend(loc=location, prop=_mpl.font_manager.FontProperties(size=fontsize))
    _pylab.draw()


class GaelInput(object):
    """
    Class that create a callable object to retrieve mouse click in a
    blocking way, a la MatLab. Based on Gael Varoquaux's almost-working
    object. Thanks Gael! I've wanted to get this working for years!

    -Jack
    """

    debug  = False
    cid    = None   # event connection object
    clicks = []     # list of click coordinates
    n      = 1      # number of clicks we're waiting for
    lines  = False   # if we should draw the lines

    def on_click(self, event):
        """
        Event handler that will be passed to the current figure to
        retrieve clicks.
        """

        # write the debug information if we're supposed to
        if self.debug: print "button "+str(event.button)+": "+str(event.xdata)+", "+str(event.ydata)

        # if this event's a right click we're done
        if event.button == 3:
            self.done = True
            return

        # if it's a valid click (and this isn't an extra event
        # in the queue), append the coordinates to the list
        if event.inaxes and not self.done:
            self.clicks.append([event.xdata, event.ydata])

            # now if we're supposed to draw lines, do so
            if self.lines and len(self.clicks) > 1:
                event.inaxes.plot([self.clicks[-1][0], self.clicks[-2][0]],
                                  [self.clicks[-1][1], self.clicks[-2][1]],
                                  color='w', linewidth=2.0, scalex=False, scaley=False)
                event.inaxes.plot([self.clicks[-1][0], self.clicks[-2][0]],
                                  [self.clicks[-1][1], self.clicks[-2][1]],
                                  color='k', linewidth=1.0, scalex=False, scaley=False)
                _pylab.draw()

        # if we have n data points, we're done
        if len(self.clicks) >= self.n and self.n is not 0:
            self.done = True
            return


    def __call__(self, n=1, timeout=0, debug=False, lines=False):
        """
        Blocking call to retrieve n coordinate pairs through mouse clicks.

        n=1             number of clicks to collect. Set n=0 to keep collecting
                        points until you click with the right mouse button.

        timeout=30      maximum number of seconds to wait for clicks before giving up.
                        timeout=0 to disable

        debug=False     show each click event coordinates

        lines=False     draw lines between clicks
        """

        # just for printing the coordinates
        self.debug = debug

        # for drawing lines
        self.lines = lines

        # connect the click events to the on_click function call
        self.cid = _pylab.connect('button_press_event', self.on_click)

        # initialize the list of click coordinates
        self.clicks = []

        # wait for n clicks
        self.n    = n
        self.done = False
        t         = 0.0
        while not self.done:
            # key step: yield the processor to other threads
            _wx.Yield();
            _time.sleep(0.05)

            # check for a timeout
            t += 0.02
            if timeout and t > timeout: print "ginput timeout"; break;

        # All done! Disconnect the event and return what we have
        _pylab.disconnect(self.cid)
        self.cid = None

        return _numpy.array(self.clicks)



def ginput(n=1, timeout=0, show=True, lines=False):
    """
    Simple functional call for physicists. This will wait for n clicks from the user and
    return a list of the coordinates of each click.

    n=1             number of clicks to collect, n=0 for "wait until right click"
    timeout=30      maximum number of seconds to wait for clicks before giving up.
                    timeout=0 to disable
    show=True       print the clicks as they are received
    lines=False     draw lines between clicks

    This is my original implementation, and I'm leaving it here because it behaves a little
    differently than the eventual version that was added to matplotlib. I would recommend using
    the official version if you can!
    """

    x = GaelInput()
    return x(n, timeout, show, lines)

#
# Style cycle, available for use in plotting
#
class style_cycle:

    def __init__(self, linestyles=['-'], markers=['s','^','o'], colors=['k','r','b','g','m'], line_colors=None, face_colors=None, edge_colors=None):
        """
        Set up the line/marker rotation cycles.

        linestyles, markers, and colors need to be lists, and you can override
        using line_colors, and face_colors, and edge_colors (markeredgecolor) by
        setting them to a list instead of None.
        """

        # initial setup, assuming all the overrides are None
        self.linestyles     = linestyles
        self.markers        = markers
        self.line_colors    = colors
        self.face_colors    = colors
        self.edge_colors    = colors

        # Apply the override colors
        if not line_colors == None: self.line_colors = line_colors
        if not face_colors == None: self.face_colors = face_colors
        if not edge_colors == None: self.edge_colors = edge_colors

        self.line_colors_index  = 0
        self.markers_index      = 0
        self.linestyles_index   = 0
        self.face_colors_index  = 0
        self.edge_colors_index  = 0

    # binding for the user to easily re-initialize
    initialize = __init__

    def reset(self):
        self.line_colors_index  = 0
        self.markers_index      = 0
        self.linestyles_index   = 0
        self.face_colors_index  = 0
        self.edge_colors_index  = 0

    def get_line_color(self, increment=1):
        """
        Returns the current color, then increments the color by what's specified
        """

        i = self.line_colors_index

        self.line_colors_index += increment
        if self.line_colors_index >= len(self.line_colors):
            self.line_colors_index = self.line_colors_index-len(self.line_colors)
            if self.line_colors_index >= len(self.line_colors): self.line_colors_index=0 # to be safe

        return self.line_colors[i]

    def set_all_colors(self, colors=['k','k','r','r','b','b','g','g','m','m']):
        self.line_colors=colors
        self.face_colors=colors
        self.edge_colors=colors
        self.reset()

    def get_marker(self, increment=1):
        """
        Returns the current marker, then increments the marker by what's specified
        """

        i = self.markers_index

        self.markers_index += increment
        if self.markers_index >= len(self.markers):
            self.markers_index = self.markers_index-len(self.markers)
            if self.markers_index >= len(self.markers): self.markers_index=0 # to be safe

        return self.markers[i]

    def set_markers(self, markers=['o']):
        self.markers=markers
        self.reset()

    def get_linestyle(self, increment=1):
        """
        Returns the current marker, then increments the marker by what's specified
        """

        i = self.linestyles_index

        self.linestyles_index += increment
        if self.linestyles_index >= len(self.linestyles):
            self.linestyles_index = self.linestyles_index-len(self.linestyles)
            if self.linestyles_index >= len(self.linestyles): self.linestyles_index=0 # to be safe

        return self.linestyles[i]

    def set_linestyles(self, linestyles=['-']):
        self.linestyles=linestyles
        self.reset()

    def get_face_color(self, increment=1):
        """
        Returns the current face, then increments the face by what's specified
        """

        i = self.face_colors_index

        self.face_colors_index += increment
        if self.face_colors_index >= len(self.face_colors):
            self.face_colors_index = self.face_colors_index-len(self.face_colors)
            if self.face_colors_index >= len(self.face_colors): self.face_colors_index=0 # to be safe

        return self.face_colors[i]

    def set_face_colors(self, colors=['k','none','r','none','b','none','g','none','m','none']):
        self.face_colors=colors
        self.reset()

    def get_edge_color(self, increment=1):
        """
        Returns the current face, then increments the face by what's specified
        """

        i = self.edge_colors_index

        self.edge_colors_index += increment
        if self.edge_colors_index >= len(self.edge_colors):
            self.edge_colors_index = self.edge_colors_index-len(self.edge_colors)
            if self.edge_colors_index >= len(self.edge_colors): self.edge_colors_index=0 # to be safe

        return self.edge_colors[i]

    def set_edge_colors(self, colors=['k','none','r','none','b','none','g','none','m','none']):
        self.edge_colors=colors
        self.reset()


    def apply(self, axes="gca"):
        """
        Applies the style cycle to the lines in the axes specified
        """

        if axes == "gca": axes = _pylab.gca()
        self.reset()
        lines = axes.get_lines()

        for l in lines:
            l.set_color(self.get_line_color(1))
            l.set_mfc(self.get_face_color(1))
            l.set_marker(self.get_marker(1))
            l.set_mec(self.get_edge_color(1))
            l.set_linestyle(self.get_linestyle(1))

        _pylab.draw()

    def __call__(self, increment=1):
        return self.get_line_color(increment)

# this is the guy in charge of keeping track of the rotation of colors and symbols for plotting
style = style_cycle(colors     = ['k','r','b','g','m'],
                    markers    = ['o', '^', 's'],
                    linestyles = ['-'])


