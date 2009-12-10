import numpy    as _numpy
import wx       as _wx
import csv      as _csv

# import some of the more common numpy functions (this is for the scripting!)
from _common_math import *

import _dialogs                       ;reload(_dialogs)



#
# This is the base class, which currently rocks.
#
class Worksheet():

    _strings        = []
    _attributes     = {}

    def __init__(self):
        """
        This object is in charge of handling a single 2-D array of data, similar
        to a spreadsheet.
        """
        return

    def load_file(self, path=None, delimiter=' \t', **kwargs):
        """
        This function loads and splits the lines of a file, storing everything
        in the master list.

        path            path to the file (path=None opens a file dialog)
        delimiter       delimiter (or list/string of delimiters) for elements of each line
        newline         delimiter separating lines
        **kwargs        sent to csv.reader()
        """

        if path==None: path=_dialogs.SingleFile()
        if path==None: return

        # get the master string from the file
        f = open(path, 'rb')
        s = f.read()
        f.close()

        # if we have a list of delimiters, do the search/replace
        if len(delimiter) > 1:
            for n in range(1,len(delimiter)): s=s.replace(delimiter[n], delimiter[0])
            delimiter = delimiter[0]

        # split the lines using whatever newline character is available
        lines = s.splitlines(False)

        # do the fast csv splitting loop
        r = _csv.reader(lines, delimiter=delimiter, skipinitialspace=True)

        # store the result in a big array of strings
        self._strings = list(r)
