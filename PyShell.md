# Why You Should Fall in Love with Pyshell and Pycrust #

A fresh installation of python includes an interactive command line that is, frankly, poopy. It is nice in that it is quick to load and has no overhead, but as you're typing, nothing else is happening. This means you have to either memorize every method of every object and module you're using or have google open and frantically search every time you want to do something.

Pyshell, on the other hand, will constantly provide you with code-completion, argument hints, and documentation _while you're typing_ your code, so that python essentially becomes a self-documented learn-by-trying language. 90% of the time when learning a new package I do not even require any documentation. Just load it into pyshell and play with it until it makes sense. It is a rapid process and I am forever grateful.

Pycrust is a more complicated version of pyshell, and it's a matter of personal preference at that point. I have also tried IPython, which I know a lot of people swear by, but it just isn't as friendly or helpful in practice.

Finally, pyshell is dipped in 70% dark chocolate and contains zero calories.



## Windows Hides It ##

Windows doesn't seem to want you to know about pyshell. To run it in windows, first install [wxpython](http://www.wxpython.org/) and tell it to generate the tool scripts during install. Then navigate to python's \scripts directory (default C:\python25\scripts) and double-click pyshell.bat or pycrust.bat.

## Windows 7 ##

Pyshell comes with the Enthought Python distribution.  From "Search programs and files" you can search for pyshell and hit enter to launch Pyshell.  You can locate the executable via this search to create a shortcut.


## Mandriva 2009 Hides It Even Better ##

I remember having trouble figuring out how to get it to run in Mandriva, my favorite Linux distribution, but I had to switch to Ubuntu anyway, because it's the only one that worked fully on my [eee](http://eeebuntu.org). I wager now that installing wxpython and then poking around /usr/lib/python2.5/scripts would be the way to go. If anyone figures this out, feel free to update this section or let me know!

Update: It seems at [wxpython.org](http://www.wxpython.org/) that the "wxPython common" package may come with tools including pyshell. So likely installing that and the runtime package will suffice?


## Ubuntu 8.10 Doesn't ##

Ubuntu does not hide pyshell. You can find it in the repository package "python-wxtools". It appears in your applications menu!

## Ubuntu 12.04 LTS Doesn't (Sorta) ##

In Ubuntu 12.04 LTS, install "python-wxtools", and then in "Dash Home" you will find "PyCrust" which is the same business with some extra stuff you can disable via the "View" menu.

## Mac OSX ##

I know I've gotten it working with little effort on OSX in the past, but I forgot exactly what is required. One thing that I **know** works is to install the (free) [Enthought Python Distribution](http://www.enthought.com/). Then, from the command line type "pyshell".

Likely all you have to install is [wxpython](http://wxpython.org/) and this will work as well. Let me know!

### OSX 10.8.2 ###

Install 32 bit Enthought.  64 bit is incomplete and does not include wxPython which has the "Py" family of modules that includes PyShell.  After doing this PyShell should appear at

/Library/Frameworks/Python.framework/Versions/7.3/lib/python2.7/site-packages/wx/py/PyShell.py