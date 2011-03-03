#######################################################################
# Set up the wx and matplotlib environment for use with pyshell/pycrust
#######################################################################

import wx as _wx

# setup matplotlib and wx so it works well with pyshell/pycrust
try:
    import matplotlib as _mpl
    _mpl.use('wxAgg') # set the backend (must do this first)

except:
    print "Note: wxAgg was not set as the backend, meaning some wx-specific functions may not work."


import pylab
pylab.ion()          # turn on interactive mode
from pylab import gca, gcf, figure, axes, draw, subplot

import scipy
import numpy

# now get the global application
_app = _wx.GetApp()
if _app == None: _app = _wx.PySimpleApp()

#############################
# Spinmob stuff
#############################

# create the user preferences object (sets up prefs directory and stuff)
import _prefs
prefs = _prefs.Prefs()

import _dialogs as dialogs             ;dialogs._prefs            = prefs
import _functions as fun               ;fun._prefs                = prefs
import _pylab_colorslider              ;_pylab_colorslider._prefs = prefs
import _plotting as plot               ;plot._prefs               = prefs
import _models as models               ;models._prefs             = prefs
import _constants as constants
import _fitting as fit                 ;fit._prefs                = prefs
import _data as data                   ;data._prefs               = prefs
data._data_types._prefs = prefs

# pull some of the common functions to the top
printer                 = fun.printer
array                   = numpy.array


xscale = plot.tweaks.xscale
yscale = plot.tweaks.yscale
