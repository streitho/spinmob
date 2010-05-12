#######################################################################
# Set up the wx and matplotlib environment for use with pyshell/pycrust
#######################################################################

import wx as _wx

# setup matplotlib and wx so it works well with pyshell/pycrust
try:
    # first see if we've loaded pylab. If we have, we've already done this.
    _pylab
except:
    # pylab is not around. Set the backend!
    import matplotlib as _mpl
    _mpl.use('WXAgg') # set the backend (must do this first)

import pylab
pylab.ion()          # turn on interactive mode
from pylab import gca, gcf, figure, axes, draw

import scipy
import numpy


# now get the global application
_app = _wx.GetApp()
if _app == None: app = _wx.App()



#############################
# Spinmob stuff
#############################

# create the user preferences object (sets up prefs directory and stuff)
import _prefs
prefs = _prefs.Prefs()

import _functions as fun               ;fun._prefs               = prefs
import _plotting as plot               ;plot._prefs              = prefs
import _dialogs as dialogs             ;dialogs._prefs           = prefs
import _data_types as data             ;data._prefs              = prefs
import _models as models               ;models.prefs             = prefs
import _constants as constants
import _pylab_colorslider              ;_pylab_colorslider.prefs = prefs
import _fitting                        ;_fitting._prefs          = prefs
import _common_math as math


# pull some of the common functions to the top
printer                 = fun.printer
fit                     = _fitting.fit
fit_model               = _fitting.fit_model

print "\nSpinmob Analysis Kit X-TREEEME\n"
