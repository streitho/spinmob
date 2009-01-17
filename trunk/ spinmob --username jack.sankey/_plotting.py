import os           as _os
import glob         as _glob
import wx           as _wx
import thread       as _thread
import pylab        as _pylab
from matplotlib.font_manager import FontProperties as _FontProperties


import _functions as _fun  
import _data_types as _data
import _pylab_tweaks as _pt 
import _dialogs                      




#
# General plotting routines
#
def plot_image(data=_data.standard(), xaxis=None, yaxis=None, sliders=True):
    """
    This asks for a file and plots a color image of the data grid.
    """
    data.load_file()
    data.get_XYZ(None,None)
    data.plot_image(map='_last')
    _pt.gui_colormap()
    return data


def plot_files(data=_data.standard(), xscript=0, yscript=1, yerror=None, yshift=0.0, yshift_every=1, clear=1, yaxis='left', linestyle='auto', legend_max="auto", paths="ask", coarsen=0, debug=0):
    """

    This selects a bunch of files, and plots them.

    data                instance of data class used to extract plot from the files
    xscript, yscript    the scripts supplied to the data
    yshift=0.0          artificial yshift
    clear=1             clear existing plot first
    line=''             line specifier for pylab
    marker=''           symbol specifier for pylab
    yaxis='left'        if 'right', this will make an overlay axis on the right (also doesn't clear)
    paths='ask'         list of full paths to data files (or we'll ask for a list)
    style=None          matplotlib plotting style, such as '-o'

    """

    # have the user select a file
    if paths=="ask":
        paths = _dialogs.MultipleFiles(data.file_extension, default_directory=data.directory)

    if paths == []: return

    if not isinstance(paths, type([])): paths = [paths]


    # get and clear the figure and axes
    f = _pylab.gcf()
    if clear and yaxis=='left': f.clf()

    # setup the right-hand axis
    if yaxis=='right':  a = _pylab.twinx()
    else:               a = _pylab.gca()

    # determine the max legend entries
    if legend_max == "auto":
        if yshift: legend_max=40
        else:      legend_max=30


    # for each path, open the file, get the data, and plot it
    for m in range(0, len(paths)):

        # fill up the xdata, ydata, and key
        if debug: print "FILE: "+paths[m]
        data.load_file(paths[m])
        data.plot(axes=a, yshift=(m/yshift_every)*yshift, clear=0, format=0, coarsen=coarsen, xscript=xscript, yscript=yscript, yerror=yerror, linestyle=linestyle)

        # now fix the legend up nice like
        if m > legend_max-2 and m != len(paths)-1:
            a.get_lines()[-1].set_label('_nolegend_')
        elif m == legend_max-2:
            a.get_lines()[-1].set_label('...')

    # fix up the title if there's an yshift
    if yshift: data.title += ', progressive y-yshift='+str(yshift)
    a.set_title(data.title)

    if yshift: format_figure(f, tall=True)
    else:      format_figure(f, tall=False)

    _pylab.draw()
    _pt.get_figure_window()
    _pt.get_pyshell()

    # return the axes
    return data





def plot_files_as_points(datax, datay, clear=True):
    """

    This is antiquated and will be updated when I need it.

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



def plot_massive(data, offset=0.0, print_plots=False, arguments="-color", pause=True, threaded=False, f=plot_files):
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



#
def format_figure(figure='gcf', tall=False):
    """

    This formats the figure with (hopefully) enough useful information for printing

    """

    if figure == 'gcf': figure = _pylab.gcf()

    # get the window of the figure
    figure_window = _pt.get_figure_window(figure)
    #figure_window.SetPosition([0,0])

    # set the size of the window
    if(tall): figure_window.SetSize([700,700])
    else:     figure_window.SetSize([700,550])

    for axes in figure.get_axes():

        # set the position/size of the axis in the window
        axes.set_position([0.13,0.1,0.5,0.8])

        # set the position of the legend
        axes.legend(loc=[1.01,0], pad=0.02, prop=_FontProperties(size=7))

        # set the label spacing in the legend
        if axes.get_legend():
            if tall: axes.get_legend().labelsep = 0.007
            else:    axes.get_legend().labelsep = 0.01

        # set up the title label
        axes.title.set_horizontalalignment('right')
        axes.title.set_size(8)
        axes.title.set_position([1.67,1.02])
        #axes.yaxis.label.set_horizontalalignment('center')
        #axes.xaxis.label.set_horizontalalignment('center')

        _pt.auto_zoom(axes)

    # get the shell window
    shell_window = _pt.get_pyshell()
    figure_window.Raise()
    shell_window.Raise()


















