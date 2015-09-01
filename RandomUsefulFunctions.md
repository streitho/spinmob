Once you have [explored spinmob](GettingStarted.md) a little, it is helpful to use the pyshell auto-complete to see what functions are available and just try them out. They aren't perfect; some are extremely useful and some of them are total hacks, but at one point or another I've needed them, sometimes out of complete ignorance. :)

There are a few commonly-used-by-me functions in the global spinmob namespace, along with some categories (assuming, in [GettingStarted](GettingStarted.md) we've already typed "import spinmob as s") and extra functions:

| s.constants   | list of physical constants, in mks       |
|:--------------|:-----------------------------------------|
| s.data        | databox data handler (see [GettingStarted](GettingStarted.md)) |
| s.dialogs     | file dialogs for selecting one or many files |
| s.fit()       | interactive data fitting using string functions (easier) |
| s.fit\_model() | interactive data fitting using the model class (harder but more flexible) |
| s.fun         | a hodgepodge of miscellaneous functions (string manipulation, array handling, etc) |
| s.models      | location of some example models I use to fit data |
| s.prefs       | global preferences object, for settings like the last used directory etc... |
| s.printer()   | prints the current figure (provided you have a valid s.prefs['print\_command'] string defined. I use gsprint.exe that came with ghostscript/gsview |

more to come.