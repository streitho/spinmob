# Printing #

Quickly printing formatted data with adequate blank space to take notes, draw pictures, etc. is a big strength of spinmob.  Setting up printing can be a bit tricky.  The information below will hopefully be of use to setup spinmob printing.


## Windows 7 ##

On Windows 7 first download and install [Ghostscript](http://www.ghostscript.com/) and [gsview](http://pages.cs.wisc.edu/~ghost/gsview/).

Second, generate a Windows ".bat" file, with contents similar to:
```
C:\"Program Files"\Ghostgum\gsview\gsprint.exe -color %1
if errorlevel 1 pause
```
This obviously depends on the location of gsprint.exe.  This .bat file is good for default printing.  Label it "Win7\_Print\_To\_Default.bat"

Save this file somewhere, for example:
```
"C:Lab\Programming\Python\Win7_Print_to_Default.bat"
```
Finally, in python save the key value pair to your spinmob preferences:
```
>>> s.prefs['print_command']='C:Lab\\Programming\\Python\\Win7_Print_to_Default.bat'
```
To deal with spaces use the following, don't forget the 'r' out front.
```
>>> s.prefs['print_command']=r'C:Lab\"printing with spaces directory"\Programming\Python\Win7_Print_to_Default.bat'
```
Now you should have printing enabled.  Make a figure and then run:
```
>>> s.printer()
```
to print and generate hard copies of nicely formatted figures.

## Mac OSX 10.8 ##

This applies to previous versions of OSX.

Set a default mac printer using System Preferences.

Then
```
>>> s.prefs['print_command']='lpr'
```

and you can now print with

```
>>> s.printer()
```