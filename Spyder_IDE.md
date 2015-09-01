# Spyder #

[Spyder](http://code.google.com/p/spyderlib/) is a great IDE.  It comes with the [Python(x,y)](http://code.google.com/p/pythonxy/) distribution which is not fully compatible with Enthought.

## Windows 7 ##

A hack installation follows.

After installing Enthought Python to get Pyshell running (this is the priority for spinmob) then download and install Python(x,y).  This installation breaks Enthought.  Repair Enthought by executing the Enthought .msi again and following the prompt.  This breaks Spyder.  Download the [latest version](http://code.google.com/p/spyderlib/downloads/list) of Spyder and install.  Pyshell and Spyder now both work.

### Mac ###

Installing on Mac has been painful in the past.  Putting it in a Windows virtual machine on a mac (virtualbox is free) is worth it.  Upgrades can break Spyder on mac, forcing a second painful install.  Hopefully this is worked out in the future.