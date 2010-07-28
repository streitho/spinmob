import numpy as _numpy
import pylab as _pylab
import matplotlib as _mpl
import spinmob as _s

from matplotlib.font_manager import FontProperties as _FontProperties
from scipy import optimize as _optimize

import _functions as _fun           ; reload(_fun)
import _pylab_tweaks as _tweaks     ; reload(_tweaks)
import _models                      ; reload(_models)
import _dialogs                     ; reload(_dialogs)
import _data_types                  ; reload(_data_types)

from numpy import *

#
# Fit function based on the model class
#
def fit_files(f='a*sin(x)+b', p='a=1.5, b', bg=None, command="", settings={}, **kwargs):
    """
    Load a bunch of data files and fit them. kwargs are sent to "data.load_multiple()" which
    are then sent to "data.standard()". Useful ones to keep in mind:

    for loading:    paths, default_directory
    for data class: xscript, yscript, eyscript

    See the above mentioned functions for more information.

    f is a string of the curve to fit, p is a comma-delimited string of
    parameters (with default values if you're into that), and bg is the
    background function should you want to use it (leaving it as None
    sets it equal to f).

    This function f will be able to see all the mathematical funcions of numpy.
    """

    # generate the model
    model = _s.models.curve(f, p, bg, globals())
    fit_files_model(model, command,    settings,    **kwargs)
    settings = {}

def fit_files_model(model, command="", settings={}, **kwargs):
    """
    Load a bunch of data files and fit them. kwargs are sent to "data.load_multiple()" which
    are then sent to "data.standard()". Useful ones to keep in mind:

    for loading:    paths, default_directory
    for data class: xscript, yscript, eyscript

    See the above mentioned functions for more information.
    """

    # Have the user select a bunch of files.
    ds = _s.data.load_multiple(**kwargs)

    for d in ds:
        print '\n\n\nFILE:', ds.index(d)+1, '/', len(ds)
        print str(d.path)
        model.fit_parameters = None
        result = model.fit(d, command, settings)

        # make sure we didn't quit.
        if result['command'] == 'q':
            settings = {}
            return

        # prepare for the next file.
        command=''
        if result.has_key('settings'): settings = result['settings']

    # clean up
    settings = {}


def fit_shown_data(f='a*sin(x)+b', p='a=1.5, b', bg=None, command="", settings={}, axes="gca", **kwargs):
    """
    Loops over the shown data, performing a fit in a separate figure.
    ***kwargs are sent to fit()
    """

    # get the axes
    if axes == "gca": axes = _pylab.gca()

    xlabel=axes.get_xlabel()
    ylabel=axes.get_ylabel()

    # get the xlimits
    xmin, xmax = axes.get_xlim()

    # get the output axes
    fig = _pylab.figure(axes.figure.number+1)

    # create the data object for fitting
    d = _s.data.standard(xlabel,ylabel,None)

    # generate the model
    model = _s.models.curve(f, p, bg, globals())

    # loop over the data
    lines = axes.get_lines()
    for n in range(len(lines)):
        line = lines[n]
        if isinstance(line, _mpl.lines.Line2D):
            # get the trimmed data from the line
            x, y    = line.get_data()
            x, y, e = _fun.trim_data(x,y,None,[xmin,xmax])

            # put together a data object with the right parameters
            d.path = line.get_label()
            d[xlabel] = x
            d[ylabel] = y

            # do the fit
            print '\n\n\nLINE:', n+1, '/', len(lines)
            model.fit_parameters = None
            settings['autopath'] = False
            settings['figure']   = axes.figure.number+1
            result = model.fit(d, command, settings)

            # make sure we didn't quit.
            if result['command'] == 'q': return

            # prepare for the next file.
            command=''
            if result.has_key('settings'): settings = result['settings']

    # clean up
    settings = {}














