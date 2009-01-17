#######################################################################
# Set up the wx and matplotlib environment for use with pyshell/pycrust
#######################################################################

import wx as _wx

# setup matplotlib and wx so it works well with pyshell/pycrust
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

import _functions as fun
import _pylab_tweaks as tweaks
import _dialogs as dialogs; dialogs._prefs = prefs
import _data_types as data
import _pylab_colorslider;  _pylab_colorslider.prefs = prefs

plot  = data.plot_files




def r():
    reload(fun)
    reload(tweaks)
    reload(data)
    reload(dialogs)

    plot = data.arccos

print "\nSpinmob Analysis Kit X-TREEEME\n"
