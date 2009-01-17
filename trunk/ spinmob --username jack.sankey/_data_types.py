import numpy as _numpy
import pylab as _pylab
import time
import wx as _wx

# import some of the more common numpy functions
from numpy import sin, cos, tan, arcsin, arccos, arctan, sqrt, exp


import _functions as _fun             ;reload(_fun)
import _pylab_tweaks as _pt           ;reload(_pt)
import _dialogs                       ;reload(_dialogs)




pi   = 3.1415926535
u0   = 1.25663706e-6
uB   = 9.27400949e-24
e    = 1.60217e-19
h    = 6.626068e-34
hbar = h/(2*pi)
gamma0 = 2*uB/hbar


#
# This is the base class, which currently rocks.
#
class standard:

    # this is used by the load_file to rename some of the annoying
    # column names that aren't consistent between different types of data files (and older data files)
    # or to just rename columns with difficult-to-remember labels.
    obnoxious_column_labels = {"example_annoying1" : "unified_name1",
                               "example_annoying2" : "unified_name2"}




    # this is just a data class with some inherited features

    # These are the current working data sets used by plotting functions.
    # The raw data from the file (after load_file()) are stored in columns and header
    ydata          = None
    xdata          = None
    yerror         = None
    X              = None
    Y              = None
    Z              = None

    directory      = "default_directory"
    xlabel         = "xlabel"
    ylabel         = "ylabel"
    legend_string  = "legend_string"
    title          = "title"
    path           = "path"

    debug          = False  # Use this to print debug info in various places
    delimiter      = None   # delimiter of the ascii file. If "None" this will just use any whitespace
    file_extension = "*"    # when asking the user for a file, use this as the filter

    constants = {"file":0}  # This is not used much, but holds random info, like the file name or whatever else you like
    header  = {}            # this dictionary will hold the header information
    columns = {}            # this dictionary will hold the data columns
    labels  = []            # we need a special list of column labels to keep track of their order during data assembly

    def __call__(self, column):
        """
        Returns the specified column. Can be an index or a label.
        """
        return self.c(column)

    def __getitem__(self, n):
        return self.columns[self.labels[n]]

    def __setitem__(self, n, x):
        """
        set's the n'th column to x
        """
        self.columns[self.labels[n]] = x


    def __len__(self):
        return len(self.labels)



    #
    # functions that are often overwritten in modified data classes
    #
    def __init__(self, delimiter=None, file_extension="*", debug=False):
        """
        delimiter       The delimiter the file uses. None means "white space"
        file_extension  Default file extension when navigating files
        debug           Displays some partial debug information while running
        """
        self.debug=debug
        self.delimiter=delimiter
        self.file_extension=file_extension


    def assemble_title(self):
        # get the pieces of the string for the plot title
        pathparts = self.path.split('\\')

        # now add the path to the title (either the full path or back a few steps
        x = min([7, len(pathparts)-1])
        self.title = "Last file:  ..."
        for n in range(0, x): self.title += '/'+pathparts[n-x]

        return self.title

    #
    # really useful functions
    #
    def load_file(self, path="ask", first_data_line="auto"):
        """
        This will load a file, storing the header info in self.header, and the data in
        self.columns

        If first_data_line="auto", then the first data line is assumed to be the first line
        where all the elements are numbers.

        If you specify a first_data_line (index, starting at 0), the columns need not be
        numbers. Everything above will be considered header information and below will be
        data columns.

        In both cases, the line used to label the columns will always be the last
        header line with the same (or more) number of elements as the first data line.

        """


        # this loads the file, getting the header and the column values,
        if self.debug: print "resetting all the file-specific stuff, path =", path
        self.constants = {"file":0}
        self.columns = {}
        self.labels  = []
        self.header  = {}
        self.xdata   = None
        self.ydata   = None
        self.yerror  = None

        if path=="ask": path = _dialogs.SingleFile(self.file_extension, default_directory=self.directory)
        if path=="":
            print "Aborted."
            return False

        self.path = path

        # open said file for reading, read in all the lines and close
        t0 = time.time()
        if self.debug: print time.time()-t0, "seconds: starting read_lines()"
        self.lines = _fun.read_lines(path)
        if self.debug: print time.time()-t0, "seconds: done."

        # break up the path into parts and take the last bit (and take a stab at the legend string)
        self.constants["file"] = path.split("\\")[-1]
        self.legend_string     = "./"+str(self.constants["file"])


        # read in the header information
        if self.debug: print time.time()-t0, "seconds: start reading header"
        labels_line = -1
        for n in range(len(self.lines)):
            # split the line by the delimiter
            s = self.lines[n].strip().split(self.delimiter)

            # first check and see if this is a data line (all elements are numbers)
            if first_data_line=="auto" and _fun.elements_are_numbers(s):
                # we've reached the first data line
                first_data_line = n
                if self.debug: print "first data line =", n

            # stop the header loop if we're here already
            if first_data_line == n: break

            # if this isn't an empty line and is all strings, assume it's a column label line
            # (we keep overwriting this until we get to the first data line)
            if len(s) > 0 and _fun.elements_are_strings(s):
                # overwrite the labels, and note the line number
                self.labels = list(s) # this makes a new instance of the list so it doesn't lose the first element!
                labels_line = n
                if self.debug: print "new labels_line:",n,s

            # otherwise, there should be at least two elements in a header element
            if len(s) == 2:
                # If there are exactly two elemenents, just store the header constant
                try:    self.header[s[0]] = float(s[1]) # this one is a number
                except: self.header[s[0]] = s[1]        # this one is a string
                if self.debug: print "header '"+s[0]+"' = "+s[1]

            elif len(s) > 2:
                # if there are more than 2 elements, then this is the column labels, or column summary values, or the date

                # if all the elements after the first are numbers, this is a "column values" row
                if _fun.elements_are_numbers(s, 1):
                    # just add this to the headers as an array
                    for n in range(1,len(s)): s[n] = float(s[n])

                    # pop off the first element, this is the string used to access the array
                    l = s.pop(0)
                    self.header[l] = _numpy.array(s)

                # otherwise it could be a list of column labels or the date or a string or something
                # treat it like a date/string or something for starters
                else:
                    # now store the string in the header
                    l=s.pop(0)
                    self.header[l] = _fun.join(s, " ")

                if self.debug: print "header '"+l+"' = "+str(self.header[l])



        # Make sure first_data_line isn't None (which happens if there's no data)
        if first_data_line == "auto":
            print "Could not find a line of pure data!"
            return



        # at this point we've found the first_data_line, and labels_line is correct or -1


        # count the number of data columns
        column_count = len(self.lines[first_data_line].strip().split(self.delimiter))

        # check to see if labels line is first_data_line-1. If it isn't, it's a false labels line (and it must be equal length to the data!)
        if not labels_line == first_data_line-1 or not len(self.labels) >= column_count:
            # it is an invalid labels line. Generate our own!
            self.labels = []
            for m in range(0, column_count): self.labels.append("column"+str(m))

        # for good measure, make sure to trim down the labels array to the size of the data columns
        for n in range(column_count, len(self.labels)): self.labels.pop(-1)

        # now we have a valid set of column labels one way or another, and we know first_data_line
        # initialize the columns arrays
        for label in self.labels: self.columns[label] = []


        # start grabbing the data
        if self.debug: print time.time()-t0, "seconds: starting to read data"

        for n in range(first_data_line, len(self.lines)):
            # split the line up
            s = self.lines[n].split(self.delimiter)

            # now start filling the column, ignoring the bad data lines
            if len(s) == len(self.labels):
                for m in range(len(s)):
                    try:    self.columns[self.labels[m]].append(float(s[m]))
                    except: self.columns[self.labels[m]].append(s[m])

            # most data files have a trailing new line, so one bad data line is ordinary
            elif not n==len(self.lines)-1: print "bad data at line,", n, "-", self.lines[n]

        if self.debug: print time.time()-t0, "seconds: yeah."

        # now loop over the columns and make them all hard-core numpy columns!
        for k in self.labels: self.columns[k] = _numpy.array(self.columns[k])

        if self.debug: print time.time()-t0, "seconds: totally."

        # now, as an added bonus, rename some of the obnoxious headers
        for k in self.obnoxious_column_labels:
            if self.columns.has_key(k):
                if self.debug: print "renaming column",k,self.obnoxious_column_labels[k]
                self.columns[self.obnoxious_column_labels[k]] = self.columns[k]


    def save_file(self, path="ask"):
        """
        This will save all the header info and columns to an ascii file.
        """

        if path=="ask": path = _dialogs.SingleFile(self.file_extension, default_directory=self.directory)
        if path=="":
            print "Aborted."
            return False

        self.path=path

        # get the delimiter
        if self.delimiter==None: delimiter = "\t"
        else:                    delimiter = self.delimiter

        # open the file and write the header
        f = open(path, 'w')
        for k in self.header.keys():
            f.write(k + delimiter)

            # if this element is a string, float, or int, just write it
            if type(self.header[k]) in [str, float, int]: f.write(str(self.header[k]) + "\n")

            # pretend it's an array and try to write it as such
            else:
                try:
                    for n in range(len(self.header[k])): f.write(str(self.header[k][n]) + delimiter)
                    f.write("\n")
                except:
                    print "header element '"+k+"' is an unknown type"

        # now write the column headers
        for l in self.labels: f.write(l + delimiter)
        f.write("\n")

        # now loop over the data
        for n in range(0, len(self[0])):
            # loop over each column
            for m in range(0, len(self)): f.write(str(self[m][n])+delimiter)
            f.write("\n")


        f.close()



    def get_new_column(self, script, name="temp"):
        """
        Generates a new column of your specification.

        Scripts are of the form:

        "3.0 + x/y - self[0] where x=3.0*c('my_column')+h('setting'), y=c(1)"

        "self" refers to the data object, giving access to everything, enabling
        complete control over the universe. c() and h() give quick reference
        to self.c() and self.h() to get columns and header lines

        You can also access globals in this module, such as _numpy and for
        convenience, common functions like sin() and sqrt() are imported
        explicitly.

        """
        if self.debug: print "Generating column '"+name+"' = "+script+"..."

        # get the expression and variables
        [expression, vars] = self.parse_script(script)
        if vars == None:
            print "invalid script"
            return None

        # generate the new column
        self.columns[name] = eval(expression, vars)

        # if we don't already have a column of this label, add it to the labels
        if not name in self.labels:
            self.labels.append(name)

        return self.columns[name]

    def get_integrated_column(self, name="integrated_dvdi", xscript="-I where I=c('current')", yscript="c('dvdi')"):
        """
        This makes a new column with trapezoid integration,
        setting the center value to zero.
        """

        self.get_new_column(xscript, "x")
        self.get_new_column(yscript, "y")

        # now numerically (trapezoid) integrate to get the other torques
        y = self.columns["y"]
        x = self.columns["x"]
        iy = [0]

        # do the trapezoid integration
        for n in range(1, len(y)):
            iy.append( iy[-1] + (0.5*y[n]+0.5*y[n-1])*(x[n]-x[n-1]) )
        iy = _numpy.array(iy)

        # now set the center value to zero
        self.columns[name] = iy-iy[len(iy)/2]


    def scale_column_by_center(self, col="dvdi"):
        """
        This will produce column_scaled, the column divided by its center value
        """

        self.columns[col+"_scaled"] = self.columns[col]/self.columns[col][len(self.columns[col])/2]
        if self.columns.has_key(col+"_error"):
            print "scaling error by the same amount"
            self.columns[col+"_error_scaled"] = self.columns[col+"_error"]/self.columns[col][len(self.columns[col])/2]




    def coarsen_columns(self, level=1):
        """
        This just coarsens the data in all the columns.
        """
        for n in range(len(self)): self[n] = _fun.coarsen_array(self[n], level)

    def parse_script(self, script):
        """
        This takes a script such as "a/b where a=current, b=3.3" and returns
        ["a/b", {"a":self.columns["current"], "b":3.3}]

        use "... where a=h(current)" for the header data

        use "... where a=c(n)" for column number n"

        You can also just use an integer for script to reference columns by number
        or use the column label as the script.
        """

        if script=="None" or script==None: return [None, None]

        # check if the script is simply an integer
        if type(script)==int:
            return ["column"+str(script), {"column"+str(script):self[script]}]



        # the scripts would like to use calls like "h('this')/3.0*c('that')",
        # so to make eval() work we should add these functions to a local list

        globbies = {'h':self.h, 'c':self.c, 'self':self}
        globbies.update(globals())

        # first split up by "where"
        split_script = script.split(" where ")


        # if it's a simple script, like "column0" or "c(3)/2.0"
        if len(split_script) == 1:
            # try to evaluate the script
            try:
                # just return a simple script so we can evaluate it later
                return ["a", {"a":eval(script, globbies)}]

            except:
                print "could not execute script:",script
                return [None, None]


        # otherwise it's a complicated script like "c(1)-a/2 where a=h('this')"

        # tidy up the expression
        expression = split_script[0].strip()

        # now split the variables list up by ,
        varsplit = split_script[1].split(',')

        # loop over the entries in the list of variables, storing the results
        # of eval in the "stuff" dictionary
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
            try:
                eval(c, globbies)
                stuff[v] = eval(c, globbies)

            except:
                print "could not evaluate variable",v,"=",script
                return [None, None]

        # incorporate the globbies so other functions can eval() with things
        # like c('this')
        stuff.update(globbies)

        # at this point we've found or generated the list
        return [expression, stuff]





    def plot(self, xscript=0, yscript=1, yerror=None, axes="gca", clear=True, format=True, coarsen=0, yshift=0, linestyle='auto', **kwargs):
        """

        KEYWORDS (can set as arguments or kwargs):

        xscript, yscript, yerror    These are the scripts to generate the three columns of data

        axes="gca"                  Which set of axes to use. "gca" means use the current axes.
        clear=True                  Clear the axes first?
        format=True                 Format the axes/labels/legend/title when done plotting?
        coarsen=0                   Should we coarsen the data?
        yshift=0                    How much vertical artificial offset should we add?
        linestyle="auto"            What type of line should we plot? "auto" means lines for data with no error
                                    and symbols for data with error.

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
            yerror= self.yerror

        # if we're doing a scripted plot
        else:

            # use the expected error column if we're supposed to
            if yerror == "auto": yerror = yscript+"_error"

            # if yerror doesn't exist and we haven't specified no error
            # set the error to none
            if not yerror in self.columns.keys() and not yerror==None:
                if self.debug: print yerror, "is not a column"
                yerror=None

            [xpression, xvars] = self.parse_script(xscript)
            if xvars == None: return
            [ypression, yvars] = self.parse_script(yscript)
            if yvars == None: return

            if not yerror == None:
                [spression, svars] = self.parse_script(yerror)
                if svars == None: yerror = None

            # try to evaluate the data
            self.xdata  = eval(xpression, xvars)
            self.ydata  = eval(ypression, yvars)
            if yerror == None:  self.yerror = None
            else:               self.yerror = eval(spression, svars)

            xdata  = self.xdata
            ydata  = self.ydata
            yerror = self.yerror

            self.xlabel = xscript
            self.ylabel = yscript
            self.title  = self.assemble_title()


        # coarsen the data if we're supposed to
        if coarsen: [xdata, ydata, yerror]=_fun.coarsen_data(xdata, ydata, yerror, coarsen)

        # assumes we've gotten data already
        if axes=="gca": axes = _pylab.gca()
        if clear:       axes.clear()

        if yshift: self.legend_string = self.legend_string + " ("+str(yshift)+")"

        if yerror == None:
            if linestyle=='auto':
                axes.plot(xdata, ydata + yshift, color = _pt.style.get_line_color(1), label=self.legend_string, linestyle='-', **plot_kwargs)
                _pylab.draw()
                axes.legend()

                # just to ease use, cycle through the markers and stuff too
                _pt.style.get_marker(1)
                _pt.style.get_face_color(1)
                _pt.style.get_edge_color(1)

            else:
                axes.plot(xdata, ydata + yshift, color   = _pt.style.get_line_color(1), label=self.legend_string,
                                             marker = _pt.style.get_marker(1),
                                             mfc    = _pt.style.get_face_color(1),
                                             mec    = _pt.style.get_edge_color(1),
                                             mew    = 1.0,
                                             linestyle = linestyle,
                                             **plot_kwargs)
                _pylab.draw()

        else:
            axes.errorbar(xdata, ydata + yshift, color  = _pt.style.get_line_color(1), label=self.legend_string,
                                        yerr   = yerror,
                                        marker = _pt.style.get_marker(1),
                                        mfc    = _pt.style.get_face_color(1),
                                        mec    = _pt.style.get_edge_color(1),
                                        mew    = 1.0, linestyle='',
                                        **plot_kwargs)
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_title(self.title)

        if format: format_figure()
        return axes



    def plot_columns(self, start=1, end=-1, yshift=0.0, yshift_every=1, xcolumn=0, legend=None, clear=1, axes="gca", legend_max=30, tall="auto", **kwargs):
        """
        This does a line plot of a range of columns.

        start=1         Index of the starting column.
        end=-1          Index of the end column, with -1 meaning "all the way"
        yshift=0.0      How much vertical artificial offset to apply
        yshift_every=1  How many traces should sit at the same offset
        xcolumn=0       Index of the x-data column
        legend=None     What header row to use as the legend values. If set to None,
                        use the column labels

        legend_max=40   Maximum number of legend entries
        tall=True       When formatting the figure, make it tall.

        **kwargs        Arguments to be sent to "plot". See "plot" for more details!
        """

        # get the axes
        if axes=="gca": axes=_pylab.gca()
        if clear:       axes.clear()

        # set the xdata and labels
        self.xdata  = self.c(xcolumn)
        self.xlabel = self.labels[xcolumn]
        self.ylabel = self.labels[start]

        # get the last index if necessary
        if end < start: end = len(self.columns)-1

        # now loop over the columns
        for n in range(start, end+1):
            # store this trace
            self.ydata = self.c(n)

            self.yerror = None
            if legend == None:  self.legend_string = self.labels[n].replace("_","")
            else:               self.legend_string = str(self.h(legend)[n-1]).replace("_","")

            # now plot it
            self.plot(yshift=((n-start)/yshift_every)*yshift, axes=axes, clear=0, format=False, **kwargs)

            # now fix the legend up real nice like
            if   n-start >  legend_max-2 and n != end: axes.get_lines()[-1].set_label('_nolegend_')
            elif n-start == legend_max-2:              axes.get_lines()[-1].set_label('...')


        # fix up the title if there's an offset
        if yshift: self.title = self.path + '\nprogressive y-shift='+str(yshift)+" every "+str(yshift_every)
        axes.set_title(self.title)

        # make it look nice
        if tall=="auto": tall = yshift
        format_figure(axes.figure, tall=tall)

        # bring it to the front, but keep the command line up too
        _pt.get_figure_window()
        _pt.get_pyshell()



    def get_XYZ(self, xaxis=None, yaxis="first", xlabel=None, ylabel=None, xcoarsen=0, ycoarsen=0):

        """
        This will assemble the X, Y, Z data for a 2d colorplot or surface.

        yaxis="first"       What values to use for the y-axis data. "first" means take the first column
                            yaxis=None means just use bin number
        xaxis=None          What values to use for the x-axis data, can be a header array
                            xaxis="first" means pop off the first row of the data
        xcoarsen, ycoarsen  How much to coarsen the columns or rows

        """


        # next we assemble the 2-d array for the colorplot
        Z=[]
        for c in self.labels: Z.append(list(self.columns[c]))

        # initialize the axis labels
        X=[]
        for n in range(len(Z)): X.append(n)
        self.xlabel = "x-step number"

        Y=[]
        for n in range(len(Z[0])): Y.append(n)
        self.ylabel = "y-step number"

        # if we're supposed to, pop off the column (X-axis) labels
        if xaxis == "first":
            X = []
            for n in range(len(Z)):
                X.append(Z[n][0])
                Z[n] = Z[n].take(range(1,len(Z[n])))
            self.xlabel = "x-values"

        # otherwise, if we specified a row from the header, use that
        elif not xaxis==None:
            X = _numpy.array(self.h(xaxis))

            # trim X down to the length of the Zd.ZX row
            X.resize(len(Z[:])-1)

            self.xlabel = xaxis

        # now if we're supposed to, pop off the row (X-axis) labels
        if yaxis=="first":

            # just pop off the first column
            Y = Z.pop(0)
            self.ylabel = "y-values"

            # if we took off the top row, we must also remove the first element of X
            if xaxis=="first": X.pop(0)

        # otherwise, it's a column value
        elif not yaxis==None:
            Y = _numpy.array(self.c(yaxis))
            self.ylabel = yaxis

        # now if we're supposed to coarsen, do so (produces a numpy array)
        self.X = _fun.coarsen_array(X, xcoarsen)
        self.Y = _fun.coarsen_array(Y, ycoarsen)
        self.Z = _fun.coarsen_matrix(Z, xcoarsen, ycoarsen)

        # if we specified labels, they trump everything
        if xlabel: self.xlabel = xlabel
        if ylabel: self.ylabel = ylabel

        return

    def plot_pseudocolor(self, map="Blues"):
        """
        This is ridiculously slow, but it is useful if you have oddly-spaced data!
        """

        # if we don't have the data, tell the user
        if self.X == None or self.Y == None or self.Z == None:
            print "You haven't assembled the surface data yet. Use get_XYZ first!"
            return

        # try the user's colormap
        try:
            colormap = eval("_pylab.cm."+map)
        except:
            print "ERROR: Invalid colormap, using default."
            colormap = _pylab.cm.Blues

        # at this point we have X, Y, Z and a colormap, so plot the bitch.
        a = _pylab.gca()
        a.clear()
        _pylab.pcolor(self.X,self.Y, self.Z.transpose(), cmap=colormap)

    def plot_image(self, map="Blues", aspect=1.0, **kwargs):
        """
        This is 8 million times faster than pseudocolor I guess, but it won't handle unevenly spaced stuff.

        You need to generate X, Y and Z first, probably using get_XYZ.
        """

        # if we don't have the data, tell the user
        if self.X == None or self.Y == None or self.Z == None:
            print "You haven't assembled the surface data yet. Use get_XYZ first!"
            return

        # try the user's colormap
        try:
            colormap = eval("_pylab.cm."+map)
        except:
            print "ERROR: Invalid colormap, using default."
            colormap = _pylab.cm.Blues

        # assume X and Y are the bin centers and figure out the bin widths
        x_width = float(self.X[-1] - self.X[0])/(len(self.X)-1)
        y_width = float(self.Y[-1] - self.Y[0])/(len(self.Y)-1)

        # at this point we have X, Y, Z and a colormap, so plot the mf.
        f=_pylab.gcf()
        f.clear()
        _pylab.imshow(self.Z.transpose(), cmap=colormap,
                      aspect=abs(aspect*float(self.X[len(self.Z)-1]-self.X[0])/float(self.Y[len(self.Z[0])-1]-self.Y[0])),
                      extent=[self.X[0]-x_width/2.0, self.X[len(self.Z   )-1]+x_width/2.0,
                              self.Y[len(self.Z[0])-1]+y_width/2.0, self.Y[0]-y_width/2.0], **kwargs)

        # set the title and labels
        self.title = self.path

        a = _pylab.gca()
        a.set_title(self.title)
        a.set_xlabel(self.xlabel)
        a.set_ylabel(self.ylabel)
        _pylab.colorbar()



    def c(self, column):
        """
        Returns the n'th column if it's an integer, returns the column based on key
        """
        if type(column)==int: return self.columns[self.labels[column]]

        # if it's not an integer, search through the columns for a matching string
        for key in self.columns.keys():
            if key.find(column) >= 0:
                return self.columns[key]
        print "Couldn't find",column,"in columns."
        return None


    def h(self, search_string):
        """
        This function searches through the header keys for one containing search_string,
        and returns that value. It is just to shorten coding.
        """
        for key in self.header.keys():
            if key.find(search_string) >= 0:
                return self.header[key]
        print "Couldn't find",search_string,"in header."
        return None









import _plotting		      ;reload(_plotting)

plot_files    = _plotting.plot_files
format_figure = _plotting.format_figure

