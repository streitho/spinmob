import pylab as _pylab
import numpy as _numpy
import matplotlib as _mpl
import wx as _wx
import time as _time
from matplotlib.font_manager import FontProperties as _FontProperties

import _dialogs                                 ;reload(_dialogs)
import _functions as _fun                       ;reload(_fun)
import _pylab_colorslider as _pc                ;reload(_pc)


line_attributes = ["linestyle","linewidth","color","marker","markersize","markerfacecolor","markeredgewidth","markeredgecolor"]

undo_list = {}

def add_text(text, x=0.01, y=0.01, axes="gca", draw=True, **kwargs):
    """
    Adds text to the axes at the specified position.

    **kwargs go to the axes.text() function.
    """
    if axes=="gca": axes = _pylab.gca()
    axes.text(x, y, text, transform=axes.transAxes, **kwargs)
    if draw: _pylab.draw()


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
    w = _wx.GetApp().GetTopWindow().GetChildren()

    # loop over them and close all the type colorsliderframe's
    for x in w:
        # if it's of the right class
        if x.__class__ == _pc._pcf.ColorSliderFrame:
            x.Close()

def gui_colormap(image="top", colormap="_last"):
    close_sliders()
    _pc.GuiColorMap(image, colormap)

def auto_zoom(axes="gca", x_space=0.04, y_space=0.04):
    if axes=="gca": axes = _pylab.gca()

    a = axes
    f = a.figure

    # get all the lines
    lines = a.get_lines()

    xdata = []
    ydata = []
    # get all the data into one giant array
    for n in range(0,len(lines)):
        # store this line's data

        if isinstance(lines[n], _mpl.lines.Line2D):
            x = lines[n].get_xdata()
            y = lines[n].get_ydata()

            # now append it to the BIG data set
            for m in range(0,len(x)):
                xdata.append(x[m])
                ydata.append(y[m])

    xmin = min(xdata)
    xmax = max(xdata)
    ymin = min(ydata)
    ymax = max(ydata)

    # we want a 3% white space boundary surrounding the data in our plot
    # so set the range accordingly
    a.set_xlim(xmin-x_space*(xmax-xmin), xmax+x_space*(xmax-xmin))
    a.set_ylim(ymin-y_space*(ymax-ymin), ymax+y_space*(ymax-ymin))

    _pylab.draw()


def format_figure(figure='gcf', tall=False, autozoom=True):
    """

    This formats the figure in a compact way with (hopefully) enough useful
    information for printing large data sets. Used mostly for line and scatter
    plots with long, information-filled titles.

    Chances are somewhat slim this will be ideal for you but it very well might
    and is at least a good starting point.

    """

    if figure == 'gcf': figure = _pylab.gcf()

    # get the window of the figure
    figure_window = get_figure_window(figure)
    #figure_window.SetPosition([0,0])

    # assume two axes means twinx
    if len(figure.get_axes()) > 1:
        window_width=600
        legend_position=1.25
    else:
        window_width=700
        legend_position=1.01

    # set the size of the window
    if(tall): figure_window.SetSize([window_width,700])
    else:     figure_window.SetSize([window_width,550])

    for n in range(len(figure.get_axes())):
        axes = figure.get_axes()[n]

        # set the position/size of the axis in the window
        axes.set_position([0.13,0.1,0.5,0.8])

        # set the position of the legend
        _pylab.axes(axes) # set the current axes
        if len(axes.lines)>0: _pylab.legend(loc=[legend_position, 1.0*n/len(figure.get_axes())], borderpad=0.02, prop=_FontProperties(size=7))

        # set the label spacing in the legend
        if axes.get_legend():
            if tall: axes.get_legend().labelsep = 0.007
            else:    axes.get_legend().labelsep = 0.01

        # set up the title label
        axes.title.set_horizontalalignment('right')
        axes.title.set_size(8)
        axes.title.set_position([1.5,1.02])
        #axes.yaxis.label.set_horizontalalignment('center')
        #axes.xaxis.label.set_horizontalalignment('center')

        if autozoom and not len(axes.lines)==0: auto_zoom(axes)

    # get the shell window
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

def image_coarsen(xlevel=0, ylevel=0, image="auto"):
    """
    This will coarsen the image data by binning each xlevel+1 along the x-axis
    and each ylevel+1 points along the y-axis
    """
    if image == "auto": image = _pylab.gca().images[0]

    Z = image.get_array()

    # store this image in the undo list
    global undo_list
    undo_list["coarsen_colorplot"] = [image, Z]

    # images have transposed data
    image.set_array(_fun.coarsen_matrix(Z, ylevel, xlevel))

    # update the plot
    _pylab.draw()

def image_coarsen_undo():
    """
    Undoes the last coarsen_colorplot command.
    """
    [image, Z] = undo_list["coarsen_colorplot"]
    image.set_array(Z)
    _pylab.draw()


def image_set_aspect(aspect=1.0, axes="gca"):
    """
    sets the aspect ratio of the current zoom level of the imshow image
    """
    if axes is "gca": axes = _pylab.gca()

    # make sure it's not in "auto" mode
    if axes.get_aspect() == 'auto': axes.set_aspect(1.0)

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



def image_set_clim(vmin=None, vmax=None, axes="gca"):
    """
    This will set the clim (range) of the colorbar.
    """
    if axes=="gca": axes=_pylab.gca()

    image = axes.images[0]

    image.set_clim(vmin, vmax)

    _pylab.draw()

def image_ubertidy(figure="gcf"):

    if figure=="gcf": figure = _pylab.gcf()

    for a in figure.axes:

        # remove the labels
        a.set_title("")
        a.set_xlabel("")
        a.set_ylabel("")

        # thicken the border
        a.frame.set_linewidth(3.0)
        a.set_frame_on(True) # adds a thick border to the colorbar

        _pylab.xticks(fontsize=18, fontweight='bold', fontname='Arial')
        _pylab.yticks(fontsize=18, fontweight='bold', fontname='Arial')

        image_set_aspect(1.0)
        get_figure_window().SetSize((550,500))

        # thicken the tick lines
        for l in a.get_xticklines(): l.set_markeredgewidth(2.0)
        for l in a.get_yticklines(): l.set_markeredgewidth(2.0)

    _pylab.draw()






def is_a_number(s):
    try: eval(s); return 1
    except:       return 0


def shift(yshift, xshift=0, progressive=1, axes="gca"):
    """

    This function adds an artificial offset to the lines.

    yshift          amount to shift vertically
    xshift          amount to shift horizontally
    axes="gca"      axes to do this on, "gca" means "get current axes"
    progressive=1   progressive means each line gets more offset
                    set to 0 to shift EVERYTHING

    """

    if axes=="gca": axes = _pylab.gca()

    # get the lines from the plot
    lines = axes.get_lines()

    # loop over the lines and trim the data
    for m in range(0,len(lines)):
        if isinstance(lines[m], _mpl.lines.Line2D):
            # get the actual data values
            xdata = lines[m].get_xdata()
            ydata = lines[m].get_ydata()

            # loop over the ydata to add the offset
            for n in range(0,len(ydata)):
                if progressive:
                    xdata[n] += m*xshift
                    ydata[n] += m*yshift
                else:
                    xdata[n] += xshift
                    ydata[n] += yshift

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

    # zoom to surround the data properly
    auto_zoom()

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

def smooth_line(line, smoothing=1, trim=True):
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
    _pylab.draw()


def coarsen_line(line, coarsen=1):
    """

    This takes a line instance and smooths its data with nearest neighbor averaging.

    """

    # get the actual data values
    xdata = list(line.get_xdata())
    ydata = list(line.get_ydata())

    xdata = _fun.coarsen_array(xdata, coarsen)
    ydata = _fun.coarsen_array(ydata, coarsen)

    # don't do anything if we don't have any data left
    if len(ydata) == 0:
        print "There's nothing left in "+str(line)+"!"
    else:
        # otherwise set the data with the new arrays
        line.set_data(xdata, ydata)

    # we refresh in real time for giggles
    _pylab.draw()

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
            get_figure_window()
            get_pyshell()

            # get the smoothing factor
            ready = 0
            while not ready:
                response = raw_input("Smoothing Factor (<enter> to skip) ")
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
            smooth_line(line, smoothing, trim)

def coarsen_all_traces(coarsen=1, axes="gca"):
    """

    This function does nearest-neighbor smoothing of the data

    """
    if axes=="gca": axes=_pylab.gca()

    # get the lines from the plot
    lines = axes.get_lines()

    # loop over the lines and trim the data
    for line in lines:
        if isinstance(line, _mpl.lines.Line2D):
            coarsen_line(line, coarsen)

def trim(xmin="auto", xmax="auto", ymin="auto", ymax="auto", axes="current"):
    """

    This function just removes all data from the plots that
    is outside of the [xmin,xmax,ymin,ymax] range.

    "auto" means "determine from the current axes's range

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
        if isinstance(line, _mpl.lines.Line2D):
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

    # zoom to surround the data properly
    auto_zoom()

def ubertidy(figure="gcf", zoom=True, width=1, height=1):
    """

    This guy performs the ubertidy from the helper on the first window.
    Currently assumes there is only one set of axes in the window!

    """

    if figure=="gcf": f = _pylab.gcf()
    else:             f = figure

    # first set the size of the window
    f.canvas.Parent.SetSize([550,550])

    # get the axes
    a = f.axes[0]

    # now loop over all the data and get the range
    lines = a.get_lines()

    # we want thick axis lines
    a.frame.set_linewidth(3.0)

    # get the tick lines in one big list
    xticklines = a.get_xticklines()
    yticklines = a.get_yticklines()

    # set their marker edge width
    _pylab.setp(xticklines+yticklines, mew=2.0)


    # set what kind of tickline they are (outside axes)
    for l in xticklines: l.set_marker(_mpl.lines.TICKDOWN)
    for l in yticklines: l.set_marker(_mpl.lines.TICKLEFT)

    # get rid of the top and right ticks
    a.xaxis.tick_bottom()
    a.yaxis.tick_left()

    # we want bold fonts
    _pylab.xticks(fontsize=20, fontweight='bold', fontname='Arial')
    _pylab.yticks(fontsize=20, fontweight='bold', fontname='Arial')

    # we want to give the labels some breathing room (1% of the data range)
    for label in _pylab.xticks()[1]: label.set_y(-0.02)
    for label in _pylab.yticks()[1]: label.set_x(-0.01)

    # set the position/size of the axis in the window
    a.set_position([0.15,0.17,0.15+width*0.4,0.17+height*0.5])

    # set the axis labels to empty (so we can add them with a drawing program)
    a.set_title('')
    a.set_xlabel('')
    a.set_ylabel('')

    # kill the legend
    a.legend_ = None

    # zoom!
    if zoom: auto_zoom(a)

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

    # now loop over all the data and get the range
    lines = figure.axes[0].get_lines()

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

    if path=="ask": path = _dialog.Save("*.*", default_directory="save_plot_default_directory")

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
    if path=="ask": path = _dialog.Save("*.plot", default_directory="save_plot_default_directory")

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

def save_raw_data(axes="gca", trace=0):
    """
    This will just output an ascii file of the data that is plotted
    trace is the index of the line to get the data from
    """


    # choose a path to save to
    path = _dialog.Save("*.*", default_directory="save_plot_default_directory")

    if path=="":
        print "your path sucks."
        return

    # if no argument was given, get the current axes
    if axes=="gca": axes=_pylab.gca()

    l = axes.lines[trace]
    x = l.get_xdata()
    y = l.get_ydata()

    # loop over the data
    for n in range(0, len(x)):
        _fun.append_to_file(path, str(x[n]) + " " + str(y[n]) + "\n")


def load_plot(clear=1, offset=0, axes="gca"):
    # choose a path to load the file from
    path = _dialog.SingleFile("*.*", default_directory="save_plot_default_directory")

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

    # starting from the top, grab ALL the wx windows available
    w = _wx.GetTopLevelWindows()

    # find all the windows that are plot windows for wxagg
    plot_windows = []
    for x in w:
        if type(x) == _mpl.backends.backend_wxagg.FigureFrameWxAgg:
            plot_windows.append(x)

    # look for the one with the same figure
    for z in plot_windows:
        if z.canvas.figure == figure: return z

    return False


def get_pyshell():
    """
    This will search through the wx windows and return the pyshell
    """

    # starting from the top, grab ALL the wx windows available
    w = _wx.GetTopLevelWindows()

    # find all the windows that are plot windows for wxagg
    plot_windows = []
    for x in w:
        if type(x) == _wx.py.shell.ShellFrame: return x

    return False


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
        if isinstance(lines[n], _mpl.lines.Line2D):

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
    def __init__(self, line_colors=['k','r','b','g','m'], face_colors=['k','r','b','g','m'], edge_colors=['k','r','b','g','m'], markers=['o']):
        self.line_colors   = line_colors
        self.face_colors   = face_colors
        self.markers       = markers
        self.edge_colors   = edge_colors
        self.line_colors_index  = 0
        self.markers_index      = 0
        self.face_colors_index  = 0
        self.edge_colors_index  = 0

    def reset(self):
        self.line_colors_index  = 0
        self.markers_index      = 0
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

        _pylab.draw()

    def __call__(self, increment=1):
        return self.get_line_color(increment)

# this is the guy in charge of keeping track of the rotation of colors and symbols for plotting
style = style_cycle(['k','r','b','g','m'],['k','r','b','g','m'], ['k','r','b','g','m'], ['o', '^', 's'])


