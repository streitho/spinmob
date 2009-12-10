import wx as _wx

from spinmob2 import prefs


#
# Dialogs
#
def Save(filters="*.*", text='Save THIS.', default_directory='default_directory'):

    global prefs

    # if this type of pref doesn't exist, we need to make a new one
    if prefs.has_key(default_directory): default = prefs[default_directory]
    else:                                default = ""

    # define the dialog object.  Doesn't opent he window
    dialog = _wx.FileDialog(None,
                           message = text,
                           defaultDir = default,
                           wildcard = filters,
                           style = _wx.SAVE|_wx.OVERWRITE_PROMPT)

    # This is the command that pops up the dialog for the user
    if dialog.ShowModal() == _wx.ID_OK:

       # update the default path so you don't have to keep navigating
       prefs[default_directory] = dialog.GetDirectory()

       # not sure if you need to, but destroy the object
       dialog.Destroy()

       return(dialog.GetPath())

    else:   return(None)






def SingleFile(filters="*.*", text='Select a file, facehead.', default_directory='default_directory'):

    global prefs

    # if this type of pref doesn't exist, we need to make a new one
    if prefs.has_key(default_directory): default = prefs[default_directory]
    else:                                 default = ""

    # define the dialog object.  Doesn't opent he window
    dialog = _wx.FileDialog(None,
                           message = text,
                           defaultDir = default,
                           wildcard = filters,
                           style = _wx.OPEN)

    # This is the command that pops up the dialog for the user
    if dialog.ShowModal() == _wx.ID_OK:

	    # get the paths for returning
	    path = dialog.GetPath()

	    # update the default path so you don't have to keep navigating
	    prefs[default_directory] = dialog.GetDirectory()

	    # not sure if you need to, but destroy the object
	    dialog.Destroy()

	    return(path)

    else:   return(None)




def Directory(text='Select a directory, facehead.', default_directory='default_directory'):

    global prefs

    # if this type of pref doesn't exist, we need to make a new one
    if prefs.has_key(default_directory): default = prefs[default_directory]
    else:                                default = ""

    # define the dialog object.  Doesn't opent he window
    dialog = _wx.DirDialog(None,
                           message = text,
                           defaultPath = default,
                           style = _wx.DD_DEFAULT_STYLE)

    # This is the command that pops up the dialog for the user
    if not dialog.ShowModal() == _wx.ID_OK: return None

    # update the default path so you don't have to keep navigating
    prefs[default_directory] = dialog.GetPath()

    # not sure if you need to, but destroy the object
    dialog.Destroy()

    return(dialog.GetPath())






def MultipleFiles(filters="*.*", text='Select some files, facehead.', default_directory='default_directory'):

    global prefs

    # if this type of pref doesn't exist, we need to make a new one
    if prefs.has_key(default_directory): default = prefs[default_directory]
    else:                                default = ""

    # define the dialog object.  Doesn't opent he window
    dialog = _wx.FileDialog(None,
                           message = text,
                           defaultDir = default,
                           wildcard = filters,
                           style = _wx.OPEN | _wx.MULTIPLE)

    # This is the command that pops up the dialog for the user
    if not dialog.ShowModal() == _wx.ID_OK: return None

    # get the paths for returning
    path_list = dialog.GetPaths()

    # update the default path so you don't have to keep navigating
    prefs[default_directory] = dialog.GetDirectory()

    # not sure if you need to, but destroy the object
    dialog.Destroy()

    return(dialog.GetPaths())


