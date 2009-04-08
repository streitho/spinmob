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

import pylab as _pylab
_pylab.ion()          # turn on interactive mode

# import some common functions to the global namespace
from pylab import gca, gcf, figure

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
import _pylab_tweaks as tweaks
import _dialogs as dialogs             ;dialogs._prefs           = prefs
import _data_types as data             ;data._prefs              = prefs
import _pylab_colorslider              ;_pylab_colorslider.prefs = prefs
import _fitting                        ;_fitting._prefs          = prefs
import _models as models
import _constants as constants
import _plotting as _plotting          ;_plotting.style          = tweaks.style
import _common_math as math

# pull some of the common functions to the top
plot_data               = _plotting.plot_data
plot_files              = _plotting.plot_files
plot_function           = _plotting.plot_function
plot_image              = _plotting.plot_image
plot_surface_data       = _plotting.plot_surface_data
plot_surface_function   = _plotting.plot_surface_function
printer                 = fun.printer
fit                     = _fitting.fit

# define a fast reload function (mostly for the developers)
def _r():
    reload(fun)                 ;printer = fun.printer
    reload(tweaks)
    reload(dialogs)
    reload(data)
    reload(_pylab_colorslider)
    reload(_fitting)            ;fit = _fitting.fit
    reload(_models)
    reload(_constants)
    reload(_plotting)
    reload(math)

    plot_data               = _plotting.plot_data
    plot_files              = _plotting.plot_files
    plot_function           = _plotting.plot_function
    plot_image              = _plotting.plot_image
    plot_surface_data       = _plotting.plot_surface_data
    plot_surface_function   = _plotting.plot_surface_function
    printer                 = fun.printer
    fit                     = _fitting.fit

print "\nSpinmob Analysis Kit X-TREEEME\n"
