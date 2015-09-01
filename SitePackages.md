# Python site-packages Directory #

The site-packages directory is where modules are installed. When calling the command

>>> import whatever

python will search this directory (and the current working directory or PYTHONPATH) for the package named "whatever". When you install numpy or scipy, you should see them appear in this directory.

## Default Locations ##

I had a heck of a time finding the site-packages directory when I switched to Linux. Linux is great these days, but it really doesn't tell you what it's doing most of the time. Well, it doesn't tell n00bs, because we don't know where the /etc/quogfdmx.rs/fjgm.log files are. Anyway, here's a growing list of locations I've found.

| Windows  | C:\python25\Lib\site-packages |
|:---------|:------------------------------|
| Ubuntu 8.10 | /usr/lib/python2.5/site-packages |
| Ubuntu 12.04 LTS | /usr/lib/python2.7/dist-packages |
| osx 10.8.2 (Enthought dist.) | /Library/Frameworks/EPD64.framework/Versions/7.3/lib/python2.7/site-packages |

The easiest way to find this is to start python and type

```
import sys
sys.path
```

which will display a list of python search paths. One of them ends with "site-packages". Look there!