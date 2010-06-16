import wx as _wx
import os as _os

import _functions as _fun             ;reload(_fun)
import _pylab_tweaks as _pt           ;reload(_pt)
import _dialogs                       ;reload(_dialogs)
import _data_types                    ;reload(_data_types)

# make the standard data class visible
standard = _data_types.standard
databox  = standard

def load(path="ask", first_data_line="auto", filters="*.*", text="Select a file, FACEHEAD.", default_directory="default_directory", **kwargs):
    """
    Loads a data file into the standard data class. Returns the data object.

    **kwargs are sent to standard()
    """
    d = standard(**kwargs)
    d.load_file(path, first_data_line, filters, text, default_directory)
    print "loaded", d.path
    _wx.Yield()
    return d

def load_multiple(paths="ask", first_data_line="auto", filters="*.*", text="Select some files, FACEHEAD.", default_directory="default_directory", **kwargs):
    """
    Loads a list of data files into a list of standard data objects.
    Returns said list.

    **kwargs are sent to standard()
    """
    if paths=="ask": paths = _dialogs.MultipleFiles(filters, text, default_directory)

    if paths==None: return

    datas = []
    for path in paths:
        datas.append(load(path, first_data_line, **kwargs))

    return datas

