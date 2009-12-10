#######################################################################
# Set up the wx and matplotlib environment for use with pyshell/pycrust
#######################################################################

import wx as _wx

# setup matplotlib and wx so it works well with pyshell/pycrust
# pylab is not around. Set the backend!
import matplotlib as _mpl
_mpl.use('WXAgg') # set the backend (must do this first)

import pylab as _pylab
_pylab.ion()          # turn on interactive mode



# now get the global application
_app = _wx.GetApp()
if _app == None: app = _wx.App()



#############################
# Spinmob stuff
#############################

# create the user preferences object (sets up prefs directory and stuff)
import _prefs
prefs = _prefs.Prefs(None)


# import the rest of the sub-modules
import _data as data;           reload(data)