import pylab as _pylab
import time
import wx as _wx
import os as _os
from mpl_toolkits.mplot3d import Axes3D

import _functions as _fun             ;reload(_fun)
import _pylab_tweaks as _pt           ;reload(_pt)
import _dialogs                       ;reload(_dialogs)

# do this so all the scripts will work with all the numpy functions
from numpy import *

#
# This is the base class, which currently rocks.
#
class standard:

    # this is used by the load_file to rename some of the annoying
    # column names that aren't consistent between different types of data files (and older data files)
    # or to just rename columns with difficult-to-remember ckeys.

    obnoxious_ckeys = {}
    #obnoxious_ckeys = {"example_annoying1" : "unified_name1",
    #                   "example_annoying2" : "unified_name2"}




    # this is just a data class with some inherited features

    # These are the current working data sets used by plotting functions.
    # The raw data from the file (after load_file()) are stored in columns and header
    ydata          = None
    xdata          = None
    eydata         = None
    X              = None
    Y              = None
    Z              = None

    xscript        = None
    yscript        = None
    eyscript       = None

    directory      = "default_directory"
    xlabel         = "xlabel"
    ylabel         = "ylabel"
    legend_string  = "(no legend_string set)"
    title          = "title"
    path           = "path"

    debug          = False  # Use this to print debug info in various places
    delimiter      = None   # delimiter of the ascii file. If "None" this will just use any whitespace
    file_extension = "*"    # when asking the user for a file, use this as the filter

    headers = {}            # this dictionary will hold the header information
    columns = {}            # this dictionary will hold the data columns
    ckeys   = []            # we need a special list of column keys to keep track of their order during data assembly
    hkeys   = []            # ordered list of header keys
    extra_globals = {}

    def __call__(self, column):
        """
        Returns the specified column. Can be an index or a label.
        """
        return self.c(column)

    def __getitem__(self, n):
        try:
            return self.columns[self.ckeys[n]]
        except:
            return self(n)

    def __setitem__(self, n, x):
        """
        set's the n'th column to x (n can be a column name too)
        """
        if type(n) == str:
            self.insert_column(data_array=x, ckey=str(n), index='end')
        elif type(n) in [int, long] and n > len(self.ckeys)-1:
            self.insert_column(data_array=x, ckey='_column'+str(len(self.ckeys)), index='end')
        else:
            self.columns[self.ckeys[n]] = array(x)


    def __len__(self):
        return len(self.ckeys)



    #
    # functions that are often overwritten in modified data classes
    #
    def __init__(self, xscript=0, yscript=1, eyscript=None, delimiter=None, file_extension="*", debug=False, **kwargs):
        """
        xscript, yscript, eyscript  Default scripts to generate xdata, ydata, eydata
                                    if the get_data() method is called
        delimiter                   The delimiter the file uses. None means "white space"
        file_extension              Default file extension when navigating files
        debug                       Displays some partial debug information while running
        """

        # update with the user-supplied/default values with kwargs
        plot_kwargs = {}
        for key in kwargs:
            try:    eval(key + "=" + kwargs[key])
            except: plot_kwargs[key] = kwargs[key]

        # this keeps the dictionaries from getting all jumbled with each other
        self.clear_columns()
        self.clear_headers()
        self.obnoxious_ckeys = {}
        self.extra_globals   = {}

        self.xscript   = xscript
        self.yscript   = yscript
        self.eyscript  = eyscript
        self.debug     = debug
        self.delimiter = delimiter
        self.file_extension = file_extension


    def assemble_title(self):
        # get the pieces of the string for the plot title
        pathparts = self.path.split(_prefs.path_delimiter)

        # now add the path to the title (either the full path or back a few steps
        x = min([7, len(pathparts)-1])
        self.title = "Last file:  ..."
        for n in range(0, x): self.title += _prefs.path_delimiter+pathparts[n-x]

        return self.title

    #
    # really useful functions
    #
    def load_file(self, path="ask", first_data_line="auto", filters="*.*", text="Select a file, FACEPANTS.", default_directory=None):
        """
        This will load a file, storing the header info in self.headers, and the data in
        self.columns

        If first_data_line="auto", then the first data line is assumed to be the first line
        where all the elements are numbers.

        If you specify a first_data_line (index, starting at 0), the columns need not be
        numbers. Everything above will be considered header information and below will be
        data columns.

        In both cases, the line used to label the columns will always be the last
        header line with the same (or more) number of elements as the first data line.

        """

        if default_directory==None: default_directory = self.directory

        # this loads the file, getting the header and the column values,
        if self.debug: print "resetting all the file-specific stuff, path =", path

        self.clear_columns()
        self.clear_headers()

        self.xdata   = None
        self.ydata   = None
        self.eydata  = None

        if path=="ask": path = _dialogs.SingleFile(self.file_extension, default_directory=self.directory)
        self.path = path

        if path==None:
            print "Aborted."
            return False

        # open said file for reading, read in all the lines and close
        t0 = time.time()
        if self.debug: print time.time()-t0, "seconds: starting read_lines()"
        self.lines = _fun.read_lines(path)
        if self.debug: print time.time()-t0, "seconds: done."

        # break up the path into parts and take the last bit (and take a stab at the legend string)
        self.legend_string = path.split(_os.path.sep)[-1]





        # read in the header information
        if self.debug: print time.time()-t0, "seconds: start reading headers"
        ckeys_line = -2
        for n in range(len(self.lines)):
            # split the line by the delimiter
            s = self.lines[n].strip().split(self.delimiter)

            # first check and see if this is a data line (all elements are numbers)
            if first_data_line=="auto" and _fun.elements_are_numbers(s):
                # we've reached the first data line
                first_data_line = n
                if self.debug: print "first data line =", n

                # quit the header loop
                break;

            # first thing to try is simply evaluating the remaining string

            try:
                remainder = list(s)
                hkey = remainder.pop(0)
                remainder = _fun.join(remainder).strip()
                self.insert_header(hkey, eval(remainder))


            # if that didn't work, try all the other complicated/flexible stuff
            except:

                # if this isn't an empty line and has strings for elements, assume it's a column key line for now
                # (we keep overwriting this until we get to the first data line)
                if len(s) > 0:
                    # overwrite the ckeys, and note the line number
                    self.ckeys = list(s) # this makes a new instance of the list so it doesn't lose the first element!
                    ckeys_line = n

                    # if it's length 1, it's just some word. Store a dummy string in there.
                    if len(s) == 1: s.append('')

                    # Also assume it is a header line. Here should be at least two elements in a header element
                    if len(s) == 2:
                        # If there are exactly two elemenents, just store the header constant
                        try:    self.headers[s[0]] = float(s[1]) # this one is a number
                        except: self.headers[s[0]] = s[1]        # this one is a string

                        # store the key in a variable like the other cases
                        l = s[0]


                    else:
                        # if there are more than 2 elements, then this is an array or a phrase

                        # if all the elements after the first are numbers, this is an array row
                        if _fun.elements_are_numbers(s, 1):
                            # just add this to the headers as an array
                            for n in range(1,len(s)): s[n] = float(s[n])

                        # pop off the first element, this is the string used to access the array
                        l = s.pop(0)
                        self.headers[l] = s


                    # in either case, we now have a header key in the variable l.
                    # now add it to the ordered list, but only if it doesn't exist
                    if _fun.index(l, self.hkeys) < 0:
                        self.hkeys.append(l)
                    else:
                        print "Duplicate header:", l

                    if self.debug: print "header '"+l+"' = "+str(self.headers[l])[0:20]+" ..."



        # Make sure first_data_line isn't None (which happens if there's no data)
        if first_data_line == "auto":
            print "Could not find a line of pure data!"
            return



        # at this point we've found the first_data_line, and ckeys_line is correct or -2


        # count the number of data columns
        column_count = len(self.lines[first_data_line].strip().split(self.delimiter))

        # check to see if ckeys line is first_data_line-1, and that it is equal in length to the
        # number of data columns. If it isn't, it's a false ckeys line
        if ckeys_line == first_data_line-1 and len(self.ckeys) >= column_count:
            # it is valid.
            # if we have too many column keys, mention it
            if len(self.ckeys) > column_count:
                print "Note: more ckeys than columns (stripping extras)"

            # remove this line from the header
            try:    self.pop_header(self.ckeys[0])
            except: print "Couldn't pop column labels from header. Weird."

        else:
            # it is an invalid ckeys line. Generate our own!
            self.ckeys = []
            for m in range(0, column_count): self.ckeys.append("column_"+str(m))


        # for good measure, make sure to trim down the ckeys array to the size of the data columns
        for n in range(column_count, len(self.ckeys)): self.ckeys.pop(-1)



        # now we have a valid set of column ckeys one way or another, and we know first_data_line.



        # initialize the columns arrays
        for label in self.ckeys: self.columns[label] = []

        # start grabbing the data
        if self.debug: print time.time()-t0, "seconds: starting to read data"

        for n in range(first_data_line, len(self.lines)):
            # split the line up
            s = self.lines[n].strip().split(self.delimiter)

            # now start filling the column, ignoring the empty or bad data lines
            for m in range(len(s)):
                try:    self.columns[self.ckeys[m]].append(float(s[m]))
                except: pass

        if self.debug: print time.time()-t0, "seconds: yeah."

        # now loop over the columns and make them all hard-core numpy columns!
        for k in self.ckeys: self.columns[k] = array(self.columns[k])

        if self.debug: print time.time()-t0, "seconds: totally."

        # now, as an added bonus, rename some of the obnoxious headers
        for k in self.obnoxious_ckeys:
            if self.columns.has_key(k):
                if self.debug: print "renaming column",k,self.obnoxious_ckeys[k]
                self.columns[self.obnoxious_ckeys[k]] = self.columns[k]


    def save_file(self, path="ask"):
        """
        This will save all the header info and columns to an ascii file.
        """

        if path=="ask": path = _dialogs.SingleFile(self.file_extension, default_directory=self.directory)
        if path in ["", None]:
            print "Aborted."
            return False

        self.path=path

        # if the path exists, make a backup
        if _os.path.exists(path):
            _os.rename(path,path+".backup")

        # get the delimiter
        if self.delimiter==None: delimiter = "\t"
        else:                    delimiter = self.delimiter

        # open the file and write the header
        f = open(path, 'w')
        for k in self.hkeys:
            # if this is a numpy array, turn it into a list
            if type(self.headers[k]) == type(array([])):
                self.headers[k] = self.headers[k].tolist()

            f.write(k + delimiter)

            # just write it
            f.write(str(self.headers[k]) + "\n")

        # now write the ckeys line
        f.write("\n")
        elements = []
        for ckey in self.ckeys: elements.append(str(ckey))
        f.write(_fun.join(elements,delimiter)+"\n")

        # now loop over the data
        for n in range(0, len(self[0])):
            # loop over each column
            elements = []
            for m in range(0, len(self)):
                # write the data if there is any, otherwise, placeholder ("x")
                if n < len(self[m]):
                    elements.append(str(self[m][n]))
                else:
                    elements.append('_')
            f.write(_fun.join(elements, delimiter)+"\n")


        f.close()



    def pop_data_point(self, n, ckeys=[]):
        """
        This will remove and return the n'th data point (starting at 0)
        in the supplied list of columns.

        n       index of data point to pop
        ckeys   which columns to do this to, specified by index or key
                empty list means "every column"
        """

        # if it's empty, it's everything
        if ckeys == []: ckeys = self.ckeys

        # loop over the columns of interest and pop the data
        popped = []
        for k in ckeys:
            if not k == None:
                # first convert to a list
                data = list(self.c(k))

                # pop the data
                popped.append(data.pop(n))

                # now set this column again
                self.insert_column(data, k)

        return popped



    def plot_and_pop_data_points(self, xkey=0, ykey=1, ekey=None, ckeys=[], **kwargs):
        """
        This will plot the columns specified by the scripts and then wait for clicks
        from the user, popping data points nearest the clicks. Right-click quits.

        xkey,ykey,ekey      column keys to plot
        ckeys               list of columns to pop, using pop_data_point()

        Set ckeys=[] to pop from all columns, and ckey="these" to pop only from the
        plotted columns, or a list of ckeys from which to pop.
        """

        if ckeys == "these": ckeys = [xkey, ykey, ekey]

        # plot the data. This should generate self.xdata and self.ydata
        self.plot(xkey, ykey, ekey, **kwargs)
        a = _pylab.gca()

        # start the loop to remove data points
        raw_input("Zoom in on the region of interest. <enter>")
        print "Now click near the data points you want to pop. Right-click to finish."
        poppies = []
        while True:
            # get a click
            clicks = _pt.ginput()
            if len(clicks)==0: return poppies
            [cx,cy] = clicks[0]

            # search through x and y for the closest point to this click
            diff = (self.xdata-cx)**2 + (self.ydata-cy)**2
            i    = _fun.index(min(diff), diff)

            # now pop!
            poppies.append(self.pop_data_point(i, ckeys))

            # now get the current zoom so we can replot
            xlim = a.get_xlim()
            ylim = a.get_ylim()

            # replot and rezoom
            _pylab.hold(True)
            self.plot(xkey, ykey, ekey, **kwargs)
            a.set_xlim(xlim)
            a.set_ylim(ylim)
            _pylab.hold(False)
            _pylab.draw()




    def generate_column(self, script, name=None):
        """
        Generates a new column of your specification. If name=None it will just
        return the data without inserting it into the data object.

        Scripts are of the form:

        "3.0 + x/y - self[0] where x=3.0*c('my_column')+h('setting'); y=c(1)"

        "self" refers to the data object, giving access to everything, enabling
        complete control over the universe. c() and h() give quick reference
        to self.c() and self.h() to get columns and header lines

        You can also access globals in this module, such as _numpy and for
        convenience, many common functions like sin() and sqrt() are imported
        explicitly. If you would like access to additional globals, set
        self.extra_globals to the appropriate globals dictionary.

        Another acceptable script is simply "F", if there's a column labeled "F".
        However, I only added this functionality as a shortcut, and something like
        "2.0*a where a=F" will not work unless F is defined somehow. I figure
        since you're already writing a complicated script, you don't want to
        accidentally shortcut your way into using a column instead of a constant!
        Use "2.0*a where a=c(F)" instead.

        NOTE: You shouldn't try to use variables like 'c=...' or 'h=...' because
        they are already column and header functions!

        """
        if self.debug: print "Generating column '"+str(name)+"' = "+str(script)+"..."

        # get the expression and variables
        [expression, v] = self.parse_script(script)
        if v == None:
            return None

        # generate the new column
        if self.debug: print expression
        new_column = eval(expression, v)

        # only insert it if someone gave it a name!
        if name==None: return new_column
        else:
            self.columns[name] = new_column

            # if we don't already have a column of this label, add it to the ckeys
            if not name in self.ckeys:
                self.ckeys.append(name)

            return self.columns[name]

    def get_data(self):
        """
        This function is mostly used for the fitting routine, whose only
        restriction on the data class is that it can load_file(), and get_data()
        storing the results in self.xdata, self.ydata, self.eydata.

        It has no parameters because the fit function doesn't need to know.
        Uses self.xscript, self.yscript, and self.eyscript.
        """

        self.xdata  = self.generate_column(self.xscript, name=None)
        self.ydata  = self.generate_column(self.yscript, name=None)
        if self.eyscript:   self.eydata = self.generate_column(self.eyscript)
        else:               self.eydata = None
        self.xlabel = self.xscript
        self.ylabel = self.yscript


    def insert_column(self, data_array, ckey='temp', index='end'):
        """
        This will insert/overwrite a new column and fill it with the data from the
        the supplied array.

        If ckey is an integer, use self.ckeys[ckey]
        """

        # if it's an integer, use the ckey from the list
        if type(ckey) in [int, long]: ckey = self.ckeys[ckey]

        # append/overwrite the column value
        self.columns[ckey] = array(data_array)
        if not ckey in self.ckeys:
            if index=='end':
                self.ckeys.append(ckey)
            else:
                self.ckeys.insert(index, ckey)


    def insert_header(self, hkey, value, index='end'):
        """
        This will insert/overwrite a value to the header and hkeys.

        If hkey is an integer, use self.hkeys[hkey]
        """

        # if it's an integer, use the hkey from the list
        if type(hkey) in [int, long]: hkey = self.hkeys[hkey]

        # set the data
        self.headers[str(hkey)] = value
        if not hkey in self.hkeys:
            if index=='end':
                self.hkeys.insert(-1,str(hkey))
            else:
                self.hkeys.insert(index, str(hkey))

    def insert_global(self, thing, name=None):
        """
        Appends or overwrites the supplied object in the self.extra_globals.

        Use this to expose generate_column() or parse_script() etc... to external
        objects and functions.

        If name=None, use thing.__name__
        """

        if name==None: name=thing.__name__
        self.extra_globals[name] = thing



    def pop_header(self, hkey):
        """
        This will remove and return the specified header value.

        You can specify either a key string or an index.
        """

        # try the integer approach first to allow negative values
        if not type(hkey) == str:
            return self.headers.pop(self.hkeys.pop(hkey))
        else:
            # find the key integer and pop it
            hkey = self.hkeys.index(hkey)

            # if we didn't find the column, quit
            if hkey < 0:
                print "Column does not exist (yes, we looked)."
                return

            # pop it!
            return self.headers.pop(self.hkeys.pop(hkey))

    def pop_column(self, ckey):
        """
        This will remove and return the data in the specified column.

        You can specify either a key string or an index.
        """

        # try the integer approach first to allow negative values
        if not type(ckey) == str:
            return self.columns.pop(self.ckeys.pop(ckey))
        else:
            # find the key integer and pop it
            ckey = self.ckeys.index(ckey)

            # if we didn't find the column, quit
            if ckey < 0:
                print "Column does not exist (yes, we looked)."
                return

            # pop it!
            return self.columns.pop(self.ckeys.pop(ckey))


    def clear_columns(self):
        """
        This will remove all the ckeys and columns.
        """
        self.ckeys   = []
        self.columns = {}


    def clear_headers(self):
        """
        This will remove all the hkeys and headers
        """
        self.hkeys    = []
        self.headers  = {}


    def parse_script(self, script, n=0):
        """
        This takes a script such as "a/b where a=c('current'), b=3.3" and returns
        ["a/b", {"a":self.columns["current"], "b":3.3}]

        You can also just use an integer for script to reference columns by number
        or use the column label as the script.

        n is for internal use. Don't use it. In fact, don't use this function, user.
        """

        if n > 1000:
            print "This script ran recursively 1000 times!"
            a = raw_input("<enter> or (q)uit: ")
            if a.strip().lower() in ['q', 'quit']:
                script = None

        if script==None: return [None, None]

        # check if the script is simply an integer
        if type(script)==int:
            if script<0: script = script+len(self.ckeys)
            return ["___"+str(script), {"___"+str(script):self[script]}]



        # the scripts would like to use calls like "h('this')/3.0*c('that')",
        # so to make eval() work we should add these functions to a local list

        globbies = {'h':self.h, 'c':self.c, 'self':self}

        # add in the module globals
        globbies.update(globals())

        # add in the supplied globals
        globbies.update(self.extra_globals)

        # first split up by "where"
        split_script = script.split(" where ")




        # #######################################
        # Scripts without a "where" statement:
        # #######################################

        # if it's a simple script, like "column0" or "c(3)/2.0"
        if len(split_script) == 1:
            if self.debug: print "script of length 1"

            # try to evaluate the script

            # first try to evaluate it as a simple column label
            if n==0 and script in self.ckeys:
                # only try this on the zero'th attempt
                # if this is a recursive call, there can be ambiguities if the
                # column names are number strings
                return ['___', {'___':self[script]}]


            # Otherwise, evaluate it.
            try:
                b = eval(script, globbies)
                return ['___', {'___':b}]
            except:
                print
                print "ERROR: Could not evaluate '"+str(script)+"'"
                _wx.Yield()
                return [None, None]


        # ######################################
        # Full-on fancy scripts
        # ######################################

        # otherwise it's a complicated script like "c(1)-a/2 where a=h('this')"

        # tidy up the expression
        expression = split_script[0].strip()

        # now split the variables list up by ,
        varsplit = split_script[1].split(';')

        # loop over the entries in the list of variables, storing the results
        # of evaluation in the "stuff" dictionary
        stuff = {}
        for var in varsplit:

            # split each entry by the "=" sign
            s = var.split("=")
            if len(s) == 1:
                print s, "has no '=' in it"
                return [None, None]

            # tidy up into "variable" and "column label"
            v = s[0].strip()
            c = s[1].strip()

            # now try to evaluate c, given our current globbies

            # recursively call this sub-script. At the end of all this mess
            # we want the final return value to be the first expression
            # and a full dictionary of variables to fill it
            [x,y] = self.parse_script(c, n+1)

            # if it's not working, just quit out.
            if y==None: return [None, None]

            stuff[v] = y[x]

        # incorporate the globbies so other functions can eval() with things
        # like c('this')
        stuff.update(globbies)

        # at this point we've found or generated the list
        return [expression, stuff]



    def rename_header(self, old_name, new_name):
        """
        This will rename the header. The supplied names need to be strings.
        """
        self.hkeys[self.hkeys.index(old_name)] = new_name
        self.headers[new_name] = self.headers.pop(old_name)

    def rename_column(self, old_name, new_name):
        """
        This will rename the column. The supplied names need to be strings.
        """
        self.ckeys[self.ckeys.index(old_name)] = new_name
        self.columns[new_name] = self.columns.pop(old_name)


    def plot(self, xscript=0, yscript=1, eyscript=None, clear=True, autoformat=True, axes="gca", coarsen=0, yshift=0, linestyle='auto', marker='auto', label=None, **kwargs):
        """

        KEYWORDS (can set as arguments or kwargs):

        xscript, yscript, eyscript    These are the scripts to generate the three columns of data

        axes="gca"                  Which set of axes to use. "gca" means use the current axes.
        clear=True                  Clear the axes first?
        autoformat=True             Format the axes/labels/legend/title when done plotting?
        coarsen=0                   Should we coarsen the data?
        yshift=0                    How much vertical artificial offset should we add?

        linestyle="auto"            What type of line should we plot?
                                    "auto" means lines for data with no error and symbols
                                    for data with error (using spinmob style object).
                                    "style" means always use lines from spinmob style cycle

        marker="auto"               What type of markers should we use?
                                    "auto" means markers only for data with error, using spinmob style
                                    "style" means definitely use markers from spinmob style
                                    otherwise just specify a marker

        label=None                  None means use self.legend_string (usually file name)

        kwargs

        """

        # update with the user-supplied/default values with kwargs
        plot_kwargs = {}
        for key in kwargs:
            try:    eval(key + "=" + kwargs[key])
            except: plot_kwargs[key] = kwargs[key]


        # if we're doing a no-script plot
        if xscript==None or yscript==None:
            if self.ydata == None or len(self.ydata) <= 0:
                print "No data to plot! Generate data with xscript and yscript."
                return False

            xdata = self.xdata
            ydata = self.ydata
            eydata= self.eydata

        # if we're doing a scripted plot
        else:

            # use the expected error column if we're supposed to
            if eyscript == "auto": eyscript = yscript+"_error"

            # if eydata doesn't exist and we haven't specified no error
            # try to generate the column
            if  not eyscript in self.columns.keys() \
            and not eyscript==None                  \
            and not type(eyscript) in [int,long]:
                if self.debug: print eyscript, "is not a column"
                eydata = self.generate_column(eyscript, None)


            [xpression, xvars] = self.parse_script(xscript)
            if xvars == None: return
            [ypression, yvars] = self.parse_script(yscript)
            if yvars == None: return

            if not eyscript == None:
                [spression, svars] = self.parse_script(eyscript)
                if svars == None: eydata = None

            # try to evaluate the data
            self.xdata  = eval(xpression, xvars)
            self.ydata  = eval(ypression, yvars)
            if eyscript == None:  self.eydata = None
            else:                 self.eydata = eval(spression, svars)

            xdata  = self.xdata
            ydata  = self.ydata
            eydata = self.eydata

            self.xlabel = xscript
            self.ylabel = yscript
            self.title  = self.assemble_title()


        # coarsen the data if we're supposed to
        if coarsen: [xdata, ydata, eydata]=_fun.coarsen_data(xdata, ydata, eydata, coarsen)

        # assumes we've gotten data already
        if axes=="gca": axes = _pylab.gca()
        if clear: axes.clear()

        # modify the legend string
        if label == None: label=self.legend_string
        if yshift: label = label + " ("+str(yshift)+")"




        # Now figure out the list of arguments and plotting function



        # default line and marker values
        mec = None
        mfc = None
        line_color = None


        # no eydata.
        if eydata == None:
            # if we're to use the style object to get the line attributes
            if linestyle in ['auto', 'style']:
                # get the linestyle from the style cycle
                linestyle  = _pt.style.get_linestyle(1)
                line_color = _pt.style.get_line_color(1)

            # only make markers without eydata if we're not in auto mode
            if marker in ['auto']:
                marker = ''

            # if we're forcing the use of style
            elif marker in ['style']:
                # get the marker attributes from the style cycle
                marker = _pt.style.get_marker(1)
                mfc    = _pt.style.get_face_color(1)
                mec    = _pt.style.get_edge_color(1)

            # otherwise, marker is already defined. Hopefully **plot_kwargs will override these values

            # handle to plotting function
            plotter = axes.plot

        # we have error bars
        else:
            # if we're in auto mode, NO LINES!
            if linestyle in ['auto']:
                linestyle  = ''
                line_color = 'k'

            # if we're forcing the style object
            elif linestyle in ['style']:
                linestyle  = _pt.style.get_linestyle(1)
                line_color = _pt.style.get_line_color(1)

            # otherwise it's specified. Default to blue and let **plot_kwargs override

            # similarly for markers
            if marker in ['auto', 'style']:
                # get the marker attributes from the style cycle
                marker = _pt.style.get_marker(1)
                mfc    = _pt.style.get_face_color(1)
                mec    = _pt.style.get_edge_color(1)

            # otherwise it's specified

            # handle to plotter and error argument
            plotter = axes.errorbar
            plot_kwargs['yerr']   = eydata
            plot_kwargs['ecolor'] = mec

        # only add these new arguments to plot_kwargs if they don't already exist
        # we want to be able to supercede the style cycle
        if  not plot_kwargs.has_key('color')           \
        and not line_color == None:                     plot_kwargs['color']        = line_color

        if  not plot_kwargs.has_key('linestyle')       \
        and not plot_kwargs.has_key('ls'):              plot_kwargs['linestyle']    = linestyle

        if  not plot_kwargs.has_key('marker'):          plot_kwargs['marker']       = marker

        if  not plot_kwargs.has_key('mec')             \
        and not plot_kwargs.has_key('markeredgecolor') \
        and not mec==None:                              plot_kwargs['mec']          = mec

        if  not plot_kwargs.has_key('mfc')             \
        and not plot_kwargs.has_key('markerfacecolor') \
        and not mfc==None:                              plot_kwargs['mfc']          = mfc

        if  not plot_kwargs.has_key('markeredgewidth') \
        and not plot_kwargs.has_key('mew'):             plot_kwargs['mew']          = 1.0

        # actually do the plotting
        plotter(xdata, ydata + yshift, label=label, **plot_kwargs)
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_title(self.title)

        if autoformat: _pt.format_figure()
        return axes



    def plot_columns(self, start=1, end=-1, yshift=0.0, yshift_every=1, xcolumn=0, legend=None, clear=1, axes="gca", legend_max=30, autoformat=True, tall="auto", **kwargs):
        """
        This does a line plot of a range of columns.

        start=1         Index of the starting column.
        end=-1          Index of the end column, with -1 meaning "all the way"
        yshift=0.0      How much vertical artificial offset to apply
        yshift_every=1  How many traces should sit at the same offset
        xcolumn=0       Index of the x-data column
        legend=None     What header row to use as the legend values. If set to None,
                        use the column ckeys

        legend_max=40   Maximum number of legend entries
        tall=True       When formatting the figure, make it tall.

        **kwargs        Arguments to be sent to "plot". See "plot" for more details!
        """

        # get the axes
        if axes=="gca": axes=_pylab.gca()
        if clear:       axes.clear()

        # set the xdata and ckeys
        self.xdata  = self.c(xcolumn)
        self.xlabel = self.ckeys[xcolumn]
        self.ylabel = self.ckeys[start]

        # get the last index if necessary
        if end < start: end = len(self.columns)-1

        # now loop over the columns
        for n in range(start, end+1):
            # store this trace
            self.ydata = self.c(n)

            self.eydata = None
            if legend == None:  self.legend_string = self.ckeys[n].replace("_","")
            else:               self.legend_string = str(self.h(legend)[n-1]).replace("_","")

            # now plot it
            self.plot(yshift=((n-start)/yshift_every)*yshift, axes=axes, clear=0, autoformat=False, **kwargs)

            # now fix the legend up real nice like
            if   n-start >  legend_max-2 and n != end: axes.get_lines()[-1].set_label('_nolegend_')
            elif n-start == legend_max-2:              axes.get_lines()[-1].set_label('...')


        # fix up the title if there's an offset
        if yshift: self.title = self.path + '\nprogressive y-shift='+str(yshift)+" every "+str(yshift_every)
        axes.set_title(self.title)

        # make it look nice
        if tall=="auto": tall = yshift
        if autoformat: _pt.format_figure(axes.figure, tall=tall)

        # bring it to the front, but keep the command line up too
        _pt.get_figure_window()
        _pt.get_pyshell()



    def get_XYZ(self, xaxis=None, yaxis=None, xlabel=None, ylabel=None, xcoarsen=0, ycoarsen=0):

        """
        This will assemble the X, Y, Z data for a 2d colorplot or surface.

        yaxis=None          What values to use for the y-axis data. "first" means take the first column
                            yaxis=None means just use bin number
        xaxis=None          What values to use for the x-axis data, can be a header array
                            xaxis="first" means pop off the first row of the data
        xcoarsen, ycoarsen  How much to coarsen the columns or rows

        """


        # next we assemble the 2-d array for the colorplot
        Z=[]
        for c in self.ckeys:
            # transform so the image plotting has the same orientation as the data
            # in the file itself.
            col = list(self.columns[c])
            col.reverse()
            Z.append(col)

        # initialize the axis labels
        X=[]
        for n in range(len(Z)): X.append(n)
        self.xlabel = "x-step number"

        Y=[]
        for n in range(len(Z[0])): Y.append(n)
        self.ylabel = "y-step number"
        Y.reverse()

        # now if we're supposed to, pop off the first column as Y labels
        if yaxis=="first":

            # just pop off the first column (Z columns are already reversed)
            Y = Z.pop(0)
            self.ylabel = "y-values"

            # pop the first element of the X-data
            X.pop(0)


        # otherwise, it's a column value
        elif not yaxis==None:
            Y = list(self.c(yaxis))
            Y.reverse()
            self.ylabel = yaxis



        # if we're supposed to, pop off the top row for the x-axis values
        if xaxis == "first":
            X = []
            for n in range(len(Z)):
                X.append(Z[n].pop(-1))

            self.xlabel = "x-values"

            # pop the first element of the Y-data
            Y.pop(-1)

        # otherwise, if we specified a row from the header, use that
        elif not xaxis==None:
            X = array(self.h(xaxis))

            # trim X down to the length of the Zd.ZX row
            X.resize(len(Z[:])-1)

            self.xlabel = xaxis



        # now if we're supposed to coarsen, do so (produces a numpy array)
        self.X = _fun.coarsen_array(X, xcoarsen)
        self.Y = _fun.coarsen_array(Y, ycoarsen)

        # Z has to be transposed to make the data file look like the plot
        self.Z = _fun.coarsen_matrix(Z, xcoarsen, ycoarsen).transpose()

        # if we specified labels, they trump everything
        if xlabel: self.xlabel = xlabel
        if ylabel: self.ylabel = ylabel

        return


    def get_columns_from_XYZ(self, corner="x"):
        """
        Assuming you have arryas self.X and self.Y along with the matrix
        self.Z, clear out the current column data and regenerate it from XYZ
        using X values for column labels, and the corner argument for the first
        column label.
        """
        self.clear_columns()

        # do the necessary transforms to make the saved data in the file
        # arranged as plotted
        Y = list(self.Y); Y.reverse()
        self.insert_column(Y, corner)

        # same for the Z's
        for n in range(len(self.X)):
            Zn = list(self.Z[:,n]); Zn.reverse()
            self.insert_column(Zn, str(self.X[n]))



    def plot_XYZ(self, cmap="Blues", plot="image", **kwargs):
        """
        This is 8 million times faster than pseudocolor I guess, but it won't handle unevenly spaced stuff.

        You need to generate X, Y and Z first, probably using get_XYZ.

        cmap    Name of the matplotlib cmap to use
        plot    Type of plot, "image" for fast colorplot, "mountains" for slow 3d plot
        """

        # if we don't have the data, tell the user
        if self.X == None or self.Y == None or self.Z == None:
            print "You haven't assembled the surface data yet. Use get_XYZ first!"
            return

        # try the user's colormap
        try:
            colormap = eval("_pylab.cm."+cmap)
        except:
            print "ERROR: Invalid colormap, using default."
            colormap = _pylab.cm.Blues

        # at this point we have X, Y, Z and a colormap, so plot the mf.
        f=_pylab.gcf()
        f.clear()

        if plot.lower() == "mountains":
            X, Y = meshgrid(self.X, self.Y)
            a = Axes3D(f)
            a.plot_surface(X, Y, self.Z, rstride=2, cstride=2, cmap=colormap, **kwargs)

        else:
            # assume X and Y are the bin centers and figure out the bin widths
            x_width = abs(float(self.X[-1] - self.X[0])/(len(self.X)-1))
            y_width = abs(float(self.Y[-1] - self.Y[0])/(len(self.Y)-1))

            # do whatever transformation is required
            X = self.X
            Y = self.Y
            Z = self.Z

            # reverse the Z's
            Z = list(Z); Z.reverse(); Z = array(Z)

            _pylab.imshow(Z, cmap=colormap,
                      extent=[X[0]-x_width/2.0, X[-1]+x_width/2.0,
                      Y[0]+y_width/2.0, Y[-1]-y_width/2.0], **kwargs)
            _pylab.colorbar()
            _pt.image_set_aspect(1.0)

        # set the title and labels
        self.title = self.path
        a = _pylab.gca()
        a.set_title(self.title)
        a.set_xlabel(self.xlabel)
        a.set_ylabel(self.ylabel)



    def c(self, ckey):
        """
        Returns the n'th column if it's an integer, returns the column based on key
        """
        if type(ckey) in [int, long]: return self.columns[self.ckeys[ckey]]

        # if it's not an integer, search through the columns for a matching string
        return self.columns[ckey]


    def h(self, hkey):
        """
        This function searches through hkeys for one *containing* the supplied key string,
        and returns that header value. It's mostly for shortening coding.

        Also can take integers, returning the key'th header value.
        """
        # if this is an index
        if type(hkey) in [int, long]: return self.headers[self.hkeys[hkey]]

        # if this is an exact match
        if hkey in self.hkeys: return self.headers[hkey]

        # Look for a fragment.
        for k in self.hkeys:
            if k.find(hkey) >= 0:
                return self.headers[k]
        print "Couldn't find",hkey,"in headers."
        return None



