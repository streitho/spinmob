import os
import scipy as _scipy
import numpy as _numpy
import pylab as _pylab
from matplotlib.font_manager import FontProperties as _FontProperties
import spinmob as _spinmob
import _dialogs

_tweaks = _spinmob.plot.tweaks

import _functions as _fun
import wx as _wx





#
# Classes
#
class model_base:

    # this is something the derived classes must do, to define
    # their fit variables
    pnames          = []
    function_string = None
    D               = None
    output_columns  = []
    output_path     = None

    # this function just creates a p0 array based on the size of the pnames array
    def __init__(self):
        # get a numpy array and then resize it
        self.p0     = _numpy.array([])
        self.p0.resize(len(self.pnames))

    def __call__(self, p, x):
        self.evaluate(p,x)

    # This function is to be overridden.
    # this is where you define the shape of your function
    # in terms of parameters p at value x
    def evaluate(self, p, x): return(p[0]*0.0*x) # example

    # this is another overridden function
    # used to get just the bacgkround, given p and value x
    def background(self, p, x): return(p[0]*0.0*x) # example

    # this is another overridden function
    # use this to guess the intial values p0 based on data
    # xbi1 and 2 are the indices used to estimate the background
    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0
        p[0] = xdata[xbi2] # example
        self.write_to_p0(p)
        return


    #
    #
    #  These functions are generally not overwritten
    #
    #

    def optimize(self, xdata, ydata, eydata=None, p0="internal"):
        """
        This actually performs the optimization on xdata and ydata.
        p0="internal"   is the initial parameter guess, such as [1,2,44.3].
                        "internal" specifies to use the model's internal guess result
                        but you better have guessed!
        returns the fit p from scipy.optimize.leastsq()
        """
        if p0 == "internal": p0 = self.p0
        if self.D == None: return _scipy.optimize.leastsq(self.residuals, p0, args=(xdata,ydata,eydata,), full_output=1)
        else:              return _scipy.optimize.leastsq(self.residuals, p0, args=(xdata,ydata,eydata,), full_output=1, Dfun=self.jacobian, col_deriv=1)

    def residuals(self, p, xdata, ydata, eydata=None):
        """
        This function returns a vector of the differences between the model and ydata, scaled by the error
        (eydata could be a scalar or a vector of length equal to ydata)
        """

        if eydata==None: eydata=1
        return (ydata - self.evaluate(p,xdata))/_numpy.absolute(eydata)

    def residuals_variance(self, p, xdata, ydata, eydata=None):
        """
        This returns the variance of the residuals, or chi^2/DOF.
        """

        return self.chi_squared(p, xdata, ydata, eydata)/(len(xdata)-len(p))

    def chi_squared(self, p, xdata, ydata, eydata=None):
        """
        This returns a single number that is the chi squared for a given set of parameters p

        This is currently not in use for the optimization.  That uses residuals.
        """

        return sum( self.residuals(p,xdata,ydata,eydata)**2 )


    # this returns the jacobian given the xdata.  Derivatives across rows, data down columns.
    # (so jacobian[len(xdata)-1] is len(p) wide)
    def jacobian(self, p, xdata, ydata):
        """
        This returns the jacobian of the system, provided self.D is defined
        """

        if not type(p)     == type(_numpy.array([0])): p     = _numpy.array(p)
        if not type(xdata) == type(_numpy.array([0])): xdata = _numpy.array(xdata)

        return self.D(p,xdata)


    def set_parameter(self, name, value):
        """
        This functions sets a parameter named "name" to a value.
        """

        try:
            # if we enter something like "min=x", get a click from the user
            if value in ['x','y']:
                # get the click
                print "Please click somewhere to get the "+value+" value."
                _tweaks.raise_figure_window()
                click = _pylab.ginput()

                # use the click value.
                if    len(click)>0 and value=='x': value = click[0][0]
                elif  len(click)>0 and value=='y': value = click[0][1]
                else:
                    print "\nCLICK ABORTED.\n"
                    return

            elif value in ['dx', 'dy', 'slope']:
                # get two clicks
                print "Please click twice to use the "+value+" value."
                _tweaks.raise_figure_window()
                clicks = _pylab.ginput(2)

                # make sure we got two clicks
                if len(clicks) == 2:
                    dx = clicks[1][0]-clicks[0][0]
                    dy = clicks[1][1]-clicks[0][1]
                    if value=='dx': value = dx
                    if value=='dy': value = dy
                    if value=='slope': value = dy/dx

                else:
                    print "\nCLICKS ABORTED.\n"
                    return


            i = self.pnames.index(name)
            self.p0[i] = float(value)
            return True

        except:
            print "ERROR:", name, "is not a valid variable or", value, "is not a valid value."
            return False

    def write_to_p0(self, p):
        """
        This function checks p's against a possible
        variable self.already_guessed and stores only those
        not in already_guessed.
        """

        try:
            # loop over all the p0's, storing p's if necessary
            for n in range(0, len(self.p0)):
                # if the pname of this p0 is not on the guessed list
                print self.pnames[n]+"POOP"+str(self.guessed_list.index(self.pnames[n]))
                if self.guessed_list.index(self.pnames[n])<0:
                    self.p0[n] = p[n]
        except: # an error occurred, likely due to no guessed_list
            self.p0 = p






    ######################################
    ## Interactive fitting routine
    ######################################

    def fit(self, data, command="", settings={}):
        """
        This generates xdata, ydata, and eydata from the three scripts
        (or auto-sets the error and updates it depending on the fit),
        fits the data, stores the results (and scripts) in the data file's header
        and saves the data in a new file.

        data            instance of a data class
        command         initial interactive fit command
        interactive     set to False to automatically fit without confirmation
        """
        d = data # for ease of coding.

        # dictionary of settings like "min" and "skip"
        default_settings = {"min"               : None,
                            "max"               : None,
                            "skip"              : True,
                            "subtract"          : False,
                            "xb1"               : 0,
                            "xb2"               : -1,
                            "show_guess"        : False,
                            "show_error"        : True,
                            "show_background"   : True,
                            "plot_all"          : False,
                            "smooth"            : 0,
                            "coarsen"           : 0,
                            "auto_error"        : False,
                            "guess"             : None,
                            "save_file"         : True,
                            "file_tag"          : 'fit_',
                            "figure"            : 1,
                            "autofit"           : False,
                            "fullsave"          : False}
        if d.eyscript == None: default_settings["auto_error"] = True

        # fill in the non-supplied settings with defaults
        for k in default_settings.keys():
            if not k in settings.keys():
                settings[k] = default_settings[k]

        # Initialize the fit_parameters (we haven't any yet!)
        fit_parameters = None
        fit_errors     = None

        # set up the figure
        fig = _pylab.figure(settings["figure"])
        fig.clear()
        axes1 = fig.add_axes([0.10, 0.08, 0.73, 0.70])
        axes2 = fig.add_axes([0.10, 0.79, 0.73, 0.13])
        axes2.xaxis.set_ticklabels([])


        # Now keep trying to fit until the user says its okay or gives up.
        hold_plot=False
        while True:

            # Plot everything.
            if hold_plot:
                hold_plot=False
            else:
                if settings["skip"]: print "Plotting but not optimizing..."
                else:                print "Beginning fit routine..."

                # Get the data.
                d.get_data()

                # if we're doing auto error, start with an array of 1's,
                # and plot the data with no error
                if d.eydata==None: d.eydata = d.xdata*0.0 + (max(d.ydata)-min(d.ydata))/20.0

                # now sort the data in case it's jaggy!
                matrix_to_sort = _numpy.array([d.xdata, d.ydata, d.eydata])
                sorted_matrix  = _fun.sort_matrix(matrix_to_sort, 0)
                d.xdata  = sorted_matrix[0]
                d.ydata  = sorted_matrix[1]
                d.eydata = sorted_matrix[2]

                # now trim all the data based on xmin and xmax
                xmin = settings["min"]
                xmax = settings["max"]
                if xmin==None: xmin = min(d.xdata)-1
                if xmax==None: xmax = max(d.xdata)+1
                [x, y, ye] = _fun.trim_data(d.xdata, d.ydata, d.eydata, [xmin, xmax])

                # smooth and coarsen
                [x,y,ye] = _fun.smooth_data( x,y,ye,settings["smooth"])
                [x,y,ye] = _fun.coarsen_data(x,y,ye,settings["coarsen"])

                # now do the first optimization. Start by guessing parameters from
                # the data's shape. This writes self.p0
                if settings["guess"]==None:
                    self.guess(x, y, settings["xb1"], settings["xb2"])
                else:
                    self.write_to_p0(settings['guess'])

                print "\n  FUNCTION:", self.function_string
                print "  GUESS:"
                for n in range(len(self.pnames)):
                    print "    "+self.pnames[n]+" = "+str(self.p0[n])
                print

                # now do the first optimization
                if not settings["skip"]:

                    fit_output = self.optimize(x, y, ye, self.p0)
                    # optimize puts out a float if there's only one parameter. Annoying.
                    if getattr(fit_output[0], '__iter__', False) == False:
                            fit_parameters = _numpy.array([fit_output[0]])
                    else:   fit_parameters = fit_output[0]

                    # If we're doing auto error, now we should scale the error so that
                    # the reduced xi^2 is 1
                    sigma_y = 1.0
                    if settings["auto_error"]:

                        # guess the correction to the y-error we're fitting (sets the reduced chi^2 to 1)
                        sigma_y = _numpy.sqrt(self.residuals_variance(fit_parameters,x,y,ye))
                        print "  initial reduced chi^2 =", sigma_y**2
                        print "  scaling error by", sigma_y, "and re-optimizing..."
                        ye       = sigma_y*ye
                        d.eydata = sigma_y*d.eydata

                        # optimize with new improved errors, using the old fit to start
                        fit_output = self.optimize(x,y,ye,p0=fit_parameters)
                        # optimize puts out a float if there's only one parameter. Annoying.
                        if getattr(fit_output[0], '__iter__', False) == False:
                                fit_parameters = _numpy.array([fit_output[0]])
                        else:   fit_parameters = fit_output[0]

                    # Now that the fitting is done, show the output

                    # grab all the information from fit_output
                    fit_covariance = fit_output[1]
                    fit_reduced_chi_squared = self.residuals_variance(fit_parameters,x,y,ye)
                    if fit_covariance is not None:
                        # get the error vector and correlation matrix from (scaled) covariance
                        [fit_errors, fit_correlation] = _fun.decompose_covariance(fit_covariance)
                    else:
                        print "  WARNING: No covariance matrix popped out of model.optimize()"
                        fit_errors      = fit_parameters
                        fit_correlation = None

                    print "  reduced chi^2 is now", fit_reduced_chi_squared

                    # print the parameters
                    print "\n  FUNCTION:", self.function_string
                    print "  FIT:"
                    for n in range(0,len(self.pnames)): print "    "+self.pnames[n]+" =", fit_parameters[n], "+/-", fit_errors[n]
                    print

                # get the data to plot
                if settings["plot_all"]:
                    x_plot  = d.xdata
                    y_plot  = d.ydata
                    ye_plot = d.eydata*sigma_y

                    # smooth and coarsen
                    [x_plot, y_plot, ye_plot] = _fun.smooth_data( x_plot, y_plot, ye_plot, settings["smooth"])
                    [x_plot, y_plot, ye_plot] = _fun.coarsen_data(x_plot, y_plot, ye_plot, settings["coarsen"])
                else:
                    # this data is already smoothed and coarsened before the fit.
                    x_plot  = x
                    y_plot  = y
                    ye_plot = ye

                # by default, don't subtract any backgrounds or anything.
                thing_to_subtract = 0.0*x_plot


                # now plot everything

                # don't draw anything until the end.
                _pylab.hold(True)
                axes1.clear()
                axes2.clear()

                # get the fit data if we're supposed to so we can know the thing to subtract
                if not fit_parameters==None:
                    # get the fit and fit background for plotting (so we can subtract it!)
                    y_fit            = self.evaluate(fit_parameters, x_plot)
                    y_fit_background = self.background(fit_parameters, x_plot)
                    if settings["subtract"]: thing_to_subtract = y_fit_background

                # plot the guess
                if settings["show_guess"]:
                    y_guess = self.evaluate(self.p0, x_plot)
                    axes1.plot(x_plot, y_guess-thing_to_subtract, color='gray', label='guess')
                    if settings["show_background"]:
                        y_guess_background = self.background(self.p0, x_plot)
                        axes1.plot(x_plot, y_guess_background-thing_to_subtract, color='gray', linestyle='--', label='guess background')

                # Plot the data
                if settings["show_error"]: # and not fit_parameters==None:
                    axes1.errorbar(x_plot, y_plot-thing_to_subtract, ye_plot, linestyle='', marker='D', mfc='blue', mec='w', ecolor='b', label='data')
                else:
                    axes1.plot(    x_plot, y_plot-thing_to_subtract,          linestyle='', marker='D', mfc='blue', mec='w', label='data')

                # plot the fit
                if not fit_parameters == None and not settings["skip"]:
                    axes1.plot(x_plot, y_fit-thing_to_subtract, color='red', label='fit')
                    if settings["show_background"]:
                        axes1.plot(x_plot, y_fit_background-thing_to_subtract, color='red', linestyle='--', label='fit background')

                    # plot the residuals in the upper graph
                    axes2.plot    (x_plot, 0*x_plot, linestyle='-', color='k')
                    axes2.errorbar(x_plot, (y_plot-y_fit)/ye_plot, ye_plot*0+1.0, linestyle='', marker='o', mfc='blue', mec='w', ecolor='b')

                # come up with a title
                title1 = d.path

                # second line of the title is the model
                title2 = "eyscript="+str(d.eyscript)+", model:"+str(self.__class__).split()[0][0:] + ", " + str(self.function_string)

                # third line is the fit parameters
                title3 = ""
                if not settings["skip"] and not fit_parameters==None:
                    t = []
                    for i in range(0,len(self.pnames)):
                        t.append(self.pnames[i]+"=%.4g+/-%.2g" % (fit_parameters[i], fit_errors[i]))
                    title3 = title3+_fun.join(t,", ")
                else:
                    title3 = title3+"(no fit performed)"

                # Start by formatting the previous plot
                axes2.xaxis.set_ticklabels([])
                axes2.set_title(title1+"\n"+title2+"\nFit: "+title3)
                axes1.set_xlabel(d.xscript)
                axes1.set_ylabel(d.yscript)

                # set the position of the legend
                axes1.legend(loc=[1.01,0], borderpad=0.02, prop=_FontProperties(size=7))

                # set the label spacing in the legend
                axes1.get_legend().labelsep = 0.01

                # set up the title label
                axes2.title.set_horizontalalignment('right')
                axes2.title.set_size(8)
                axes2.title.set_position([1.0,1.010])

                _tweaks.auto_zoom(axes1)
                _pylab.draw()

                _tweaks.raise_figure_window()
                _wx.Yield()
                _tweaks.raise_pyshell()

            # the only way we optimize is if we hit enter.
            if settings["autofit"]: settings["skip"] = False
            else:                   settings["skip"] = True















            # If last command is None, this is the first time. Parse the initial
            # command but don't ask for one.
            if command == "":
                print "min=" + str(settings['min']) + ", max="+str(settings['max'])
                if settings["autofit"]:
                    if fit_parameters==None:    command = ""
                    else:                       command = "y"
                else:
                    command = raw_input("-------> ").strip()

            # first check and make sure the command isn't one of the simple ones
            if command.lower() in ['h', 'help']:
                print
                print "COMMANDS"
                print "  <enter>    Do more iterations."
                print "  g          Guess and show the guess."
                print "  z          Use current zoom to set xmin and xmax."
                print "  o          Choose and output summary file."
                print "  n          No, this is not a good fit. Move on."
                print "  y          Yes, this is a good fit. Move on."
                print "  u          Same as 'y' but use fit as the next guess."
                print "  p          Call the printer() command."
                print "  q          Quit."
                print
                print "SETTINGS"
                for key in settings.keys(): print "  "+key+" =", settings[key]
                print
                print "SETTING PARAMETER GUESS VALUES"
                print "  <parameter>=<value>"
                print "              sets the parameter guess value."
                print
                print "  <parameter>=x|y|dx|dy|slope"
                print "              sets the parameter guess value to the"
                print "              clicked x, y, dx, dy, or slope value."

                command=""
                hold_plot=True
                continue

            elif command.lower() in ['q', 'quit', 'exit']:
                return {'command':'q'}

            elif command.lower() in ['g', 'guess']:
                settings['guess'] = None
                settings['show_guess'] = True

            elif command.lower() in ['z', 'zoom']:
                settings['min'] = axes1.get_xlim()[0]
                settings['max'] = axes1.get_xlim()[1]

            elif command.lower() in ['o', 'output']:
                # print all the header elements of the current databox
                # and have the user choose as many as they want.
                print "\n\nChoose which header elements to include as columns in the summary file:"
                for n in range(len(d.hkeys)):
                    print "  "+str(n)+": "+str(d.hkeys[n])

                # get a list of numbers from the user
                key_list = raw_input("pick headers by number: ").split(',')
                try:
                    # get the list of keys.
                    self.output_columns = []
                    for n in key_list: self.output_columns.append(d.hkeys[int(n.strip())])

                    # now have the user select a file
                    self.output_path = _dialogs.Save()
                    if not self.output_path==None:
                        # write the column names
                        f = open(self.output_path, 'w')
                        f.write('function_string\t'+self.function_string+
                                '\nmodel\t'+str(self.__class__)+
                                '\nxscript\t'+str(d.xscript)+
                                '\nyscript\t'+str(d.yscript)+
                                '\neyscript\t'+str(d.eyscript)+'\n\n')
                        for k in self.output_columns: f.write(k+'\t')
                        for n in self.pnames: f.write(n+'\t'+n+'_error\t')
                        f.write('reduced_chi_squared\n')
                        f.close()

                    # all set. It will now start appending to this file.

                except:
                    print "\nOOPS. OOPS."

            elif command.lower() in ['y', 'yes','u','use']:

                if fit_parameters==None or fit_errors==None:
                    print "Can't say a fit is good with no fit!"

                elif settings['save_file']:
                    # If this is a good fit. Add relevant information to the header then save
                    d.insert_header("fit_model", str(self.__class__).split()[0][0:])
                    d.insert_header("fit_function", str(self.function_string))
                    for n in range(len(self.pnames)):
                        d.insert_header("fit_"+self.pnames[n], [fit_parameters[n], fit_errors[n]])
                    d.insert_header("fit_reduced_chi_squared",fit_reduced_chi_squared)

                    # build the correlations array (not a 2-d array)
                    d.insert_header("fit_correlations",       fit_correlation)

                    d.insert_header("fit_min",      settings['min'])
                    d.insert_header("fit_max",      settings['max'])
                    d.insert_header("fit_smooth",   settings['smooth'])
                    d.insert_header("fit_coarsen",  settings['coarsen'])

                    # auto-generate the new file name
                    if settings['fullsave'] in [1, True, 'auto']:
                        directory, filename = os.path.split(d.path)
                        new_path = directory + os.sep + settings['file_tag'] + filename
                        if new_path: d.save_file(new_path)

                    elif settings['fullsave'] in [2, 'ask']:
                        new_path = _dialogs.SingleFile()
                        if new_path: d.save_file(new_path)

                    # append to the summary file
                    if self.output_path:
                        f = open(self.output_path,'a')
                        for k in self.output_columns:
                            f.write(str(d.h(k))+'\t')
                        for n in range(len(fit_parameters)):
                            f.write(str(fit_parameters[n])+'\t'+str(fit_errors[n])+'\t')
                        f.write(str(fit_reduced_chi_squared)+'\n')
                        f.close()


                # Return the information
                return_value = {"command"                   :'y',
                                "fit_parameters"            :fit_parameters,
                                "fit_errors"                :fit_errors,
                                "fit_reduced_chi_squared"   :fit_reduced_chi_squared,
                                "fit_covariance"            :fit_covariance,
                                "settings"                  :settings}
                if command.lower() in ['u', 'use']:
                    return_value['command'] = 'u'
                    return_value['settings']['guess'] = fit_parameters
                return return_value

            elif command.lower() in ['n', 'no', 'next']:
                return {'command':'n'}

            elif command.lower() in ['p', 'print']:
                _spinmob.printer()
                hold_plot = True

            elif command.lower() in ['']:
                settings["skip"] = False

            else:
                # now parse it (it has the form "min=2, max=4, plot_all=True")
                s = command.split(',')
                for c in s:
                    try:
                        [key, value] = c.split('=')
                        key   = key.strip()
                        value = value.strip()
                        if settings.has_key(key):
                            settings[key] = eval(value)
                        else:
                            self.set_parameter(key, value)
                            settings['guess'] = self.p0

                    except:
                        print "ERROR: '"+str(c)+"' is an invalid command."

            # make sure we don't keep doing the same command over and over!
            command = ""
            print



class curve(model_base):

    globs={} # globals such as sin and cos...

    def __init__(self, f='a+b*x+c*x**2', p='a=1.5, b, c=1.5', bg=None, globs={}):
        """
        This class takes the function string you specify and generates
        a model based on it.

        f is a string of the curve to fit, p is a comma-delimited string of
        parameters (with default values if you're into that), and bg is the
        background function should you want to use it (leaving it as None
        sets it equal to f).

        globs is a list of globals should you wish to have these visible to f.

        If you want to do something a little more fancy with a guessing algorithm,
        it's relatively straightforward to write one of the model classes similar
        to the examples given in spinmob.models
        """

        # get the function
        self.function_string                = f
        if bg==None: self.background_string = f
        else:        self.background_string = bg

        # start by parsing the f string
        p_split = p.split(',')

        # Loop over the parameters, get their names and possible default values
        self.pnames     = []
        self.defaults   = []
        for parameter in p_split:
            parameter_split = parameter.split('=')

            self.pnames.append(parameter_split[0].strip())

            if len(parameter_split)==2: self.defaults.append(float(parameter_split[1]))
            else:                       self.defaults.append(1.0)

        # set up the guess
        self.p0 = _numpy.array(self.defaults)

        # store the globals
        self.globs = globs

        # override the function and background
        args = 'x,'+_fun.join(self.pnames,',')
        self.f  = eval('lambda ' + args + ': '+self.function_string,   self.globs)
        self.bg = eval('lambda ' + args + ': '+self.background_string, self.globs)


    def evaluate(self,   p, x):
        return self.f( x, *p)
    def background(self, p, x):
        return self.bg(x, *p)

    # You can override this if you want the guess to be something fancier.
    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        """
        This function takes the supplied data (and two indices from which to
        estimate the background should you want them) and returns a best guess
        of the parameters, then stores this guess in p0.
        """

        self.write_to_p0(self.defaults)
        return









class complete_lorentz_flat_background(model_base):

    function_string = "p[0]+p[3]*((x-p[1])/p[2]))/(1.0+((x-p[1])/p[2])**2)+p[4]"
    pnames = ["height", "center", "width", "height2", "offset"]

    # define the function
    def background(self, p, x):
        return(p[4]) # the 0.0*x is so you get an array from an array

    # main function
    def evaluate(self, p, x):
        return (p[0]+p[3]*((x-p[1])/p[2]))/(1.0+((x-p[1])/p[2])**2)+p[4]

    # come up with a routine for guessing p0
    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # guess the background
        p[4] = (ydata[xbi1]+ydata[xbi2])/2.0

        # guess the height and center from the max
        p[0] = max(ydata - p[4])
        p[1] = xdata[(ydata-p[4]).index(max(ydata-p[4]))] # center

        # guess the asymmetric part from the minimum and background
        p[3] = 0

        # guess the halfwidth
        p[2] = (max(xdata)-min(xdata))/12.0

        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)




class exponential_offset(model_base):

    function_string = "p[0]*_numpy.exp(-x/p[1])+p[2]"
    pnames = ["amplitude", "tau", "offset"]

    # define the function
    def background(self, p, x):
        return(p[2]+0.0*x)

    def evaluate(self, p, x):
        return(p[0]*_numpy.exp(-x/p[1])+p[2])

    # come up with a routine for guessing p0
    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # amplitude = zero value?
        p[0] = ydata[0]
        p[1] = xdata[_fun.index_nearest(min(ydata) + 0.368*(max(ydata)-min(ydata)), ydata)]
        p[2] = ydata[-1]

        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)






class linear(model_base):

    function_string = "p[0]*x + p[1]"
    pnames = ["slope", "offset"]

    # this must return an array!
    def background(self, p, x):
        return p[0]*x + p[1] # the 0.0*x is so you get an array from an array

    def evaluate(self, p, x):
        return p[0]*x + p[1]

    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        x1 = xdata[xbi1]
        y1 = ydata[xbi1]
        x2 = xdata[xbi2]
        y2 = ydata[xbi2]

        # guess the slope and intercept
        p[0] = (y1-y2)/(x1-x2)
        p[1] = y1 - p[0]*x1

        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)


class noisy_lorentzian_with_offset(model_base):
    function_string = "h/(1.0+((x-x0)/w)**2)+b"
    pnames = ["h", "x0", "w", "b"]

    # this function just creates a p0 array based on the size of the pnames array
    def __init__(self):
        """
        This model assumes the error on the Lorentzian is proportional to the
        height at x (including the offset). It's a rough guess, I know but it's
        much better than constant error bars for FFT's fluctuating resonators.
        """
        model_base.__init__(self)


    # define the function
    def background(self, p, x):
        return(p[3])
    def evaluate(self, p, x):
        return(p[0]/(1.0+((x-p[1])/p[2])**2)+p[3])

    # come up with a routine for guessing p0
    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # guess the slope and background
        p[3] = ydata[xbi1]

        # guess the height and center from the max
        p[0] = max(ydata - p[3])
        p[1] = xdata[list(ydata-p[3]).index(max(ydata-p[3]))] # center

        # guess the halfwidth
        p[2] = (max(xdata)-min(xdata))/12.0

        # write these values to self.p0
        self.write_to_p0(p)



    def residuals(self, p, xdata, ydata, eydata=None):
        """
        This function returns a vector of the differences between the model and ydata, scaled by the error
        (eydata could be a scalar or a vector of length equal to ydata)
        """

        # the eydata can be a number or None
        if eydata == None: eydata = 1.0

        # get the model y-values
        ymodel = self.evaluate(p,xdata)

        # scale the error by the ymodel values
        yscale = ymodel/max(ymodel)
        eydata = eydata*yscale

        return (ydata - ymodel)/_numpy.absolute(eydata)






class lorentz_linear_background(model_base):

    function_string = "p[0]/(1.0+((x-p[1])/p[2])**2)+p[3]*x+p[4]"
    pnames = ["height", "center", "width", "slope", "offset"]

    # define the function
    def background(self, p, x):
        return(p[3]*x+p[4])
    def evaluate(self, p, x):
        return(p[0]/(1.0+((x-p[1])/p[2])**2)+p[3]*x+p[4])

    # come up with a routine for guessing p0
    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # guess the slope and background
        p[3] = (ydata[xbi1]-ydata[xbi2])/(xdata[xbi1]-xdata[xbi2])
        p[4] = ydata[xbi1]-p[3]*xdata[xbi1]

        # guess the height and center from the max
        p[0] = max(ydata - p[4])
        p[1] = xdata[(ydata-p[4]).index(max(ydata-p[4]))] # center

        # guess the halfwidth
        p[2] = (max(xdata)-min(xdata))/12.0

        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)



class parabola(model_base):

    function_string = "p[0]*(x-p[1])**2 + p[2]"
    pnames = ["A", "x0", "y0"]

    # this must return an array!
    def background(self, p, x):
        return p[2]+0*x # must return an array

    def evaluate(self, p, x):
        return p[0]*(x-p[1])**2 + p[2]

    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # guess the slope and intercept
        p[0] = 1.0
        p[1] = xdata[len(xdata)/2]
        p[2] = min(ydata)

        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)

class sine(model_base):

    function_string = "p[0]*sin(2*pi*x/p[1]+p[2]) + p[3]"
    pnames = ["A", "lambda", "phi","offset"]

    # this must return an array!
    def background(self, p, x):
        return p[3]+0*x # must return an array

    def evaluate(self, p, x):
        return p[0]*_numpy.sin(2*pi*x/p[1]+p[2]) + p[3]

    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # guess the offset
        p[3] = (max(ydata)+min(ydata))/2.0

        # guess the amplitude
        p[0] = (max(ydata)-min(ydata))/2.0

        # guess the wavelength and phase
        n1 = _fun.index_next_crossing(p[3],ydata,0)
        n2 = _fun.index_next_crossing(p[3],ydata,n1+3)
        if n1<0 or n2<0:
            p[1] = (xdata[-1]-xdata[0])/3
            p[2] = 0
        else:
            p[1] = (xdata[n2]-xdata[n1])
            p[2] = -2*pi*xdata[n1]/p[1]



        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)

class sine_no_phase(model_base):

    function_string = "p[0]*sin(2*pi*x/p[1]) + p[2]"
    pnames = ["A", "lambda", "offset"]

    # this must return an array!
    def background(self, p, x):
        return p[-1]+0*x # must return an array

    def evaluate(self, p, x):
        return p[0]*_numpy.sin(2*pi*x/p[1]) + p[2]

    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # guess the offset
        p[2] = (max(ydata)+min(ydata))/2.0

        # guess the amplitude
        p[0] = (max(ydata)-min(ydata))/2.0

        # guess the wavelength and phase
        n1 = _fun.index_next_crossing(p[-1],ydata,0)
        n2 = _fun.index_next_crossing(p[-1],ydata,n1+3)
        if n1<0 or n2<0:
            p[1] = (xdata[-1]-xdata[0])/3
        else:
            p[1] = (xdata[n2]-xdata[n1])



        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)


class sine_stretched_3(model_base):

    function_string = "p[0]*_numpy.sin(2*pi*(p[1]*(x-p[4]) + p[2]*(x-p[4])**2 + p[3]*(x-p[4])**3)) + p[5]"
    pnames = ["A", "a1", "a2", "a3", "x0","y0"]

    # this must return an array!
    def background(self, p, x):
        return p[-1]+0*x # must return an array

    def evaluate(self, p, x):
        return p[0]*_numpy.sin(2*pi*(p[1]*(x-p[4]) + p[2]*(x-p[4])**2 + p[3]*(x-p[4])**3)) + p[5]

    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # guess the offset
        p[5] = (max(ydata)+min(ydata))/2.0

        # guess the amplitude
        p[0] = (max(ydata)-min(ydata))/2.0

        # guess the wavelength and phase
        n1 = _fun.index_next_crossing(p[-1],ydata,0)
        n2 = _fun.index_next_crossing(p[-1],ydata,n1+3)
        if n1<0 or n2<0:
            p[1] = 3./(xdata[-1]-xdata[0])
            p[4] = 0
        else:
            p[1] = 1.0/(xdata[n2]-xdata[n1])
            p[4] = -2*pi*xdata[n1]/p[1]

        # assume not stretched much initially
        p[2] = 0
        p[3] = 0

        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)


class sine_stretched_4(model_base):

    function_string = "p[0]*_numpy.sin(2*pi*(p[1]*(x-p[5]) + p[2]*(x-p[5])**2 + p[3]*(x-p[5])**3 + p[4]*(x-p[5])**4)) + p[6]"
    pnames = ["A", "a1", "a2", "a3", "a4", "x0","y0"]

    # this must return an array!
    def background(self, p, x):
        return p[-1]+0*x # must return an array

    def evaluate(self, p, x):
        return p[0]*_numpy.sin(2*pi*(p[1]*(x-p[5]) + p[2]*(x-p[5])**2 + p[3]*(x-p[5])**3 + p[4]*(x-p[5])**4)) + p[6]

    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # guess the offset
        p[5] = (max(ydata)+min(ydata))/2.0

        # guess the amplitude
        p[0] = (max(ydata)-min(ydata))/2.0

        # guess the wavelength and phase
        n1 = _fun.index_next_crossing(p[-1],ydata,0)
        n2 = _fun.index_next_crossing(p[-1],ydata,n1+3)
        if n1<0 or n2<0:
            p[1] = 3./(xdata[-1]-xdata[0])
            p[4] = 0
        else:
            p[1] = 1.0/(xdata[n2]-xdata[n1])
            p[4] = -2*pi*xdata[n1]/p[1]

        # assume not stretched much initially
        p[2] = 0
        p[3] = 0

        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)


class sine_plus_linear(model_base):

    function_string = "p[0]*_numpy.sin(2*pi*x/p[1]+p[2]) + p[3] + p[4]*x"
    pnames = ["A", "lambda", "phi","offset","slope"]

    # this must return an array!
    def background(self, p, x):
        return p[3]+p[4]*x # must return an array

    def evaluate(self, p, x):
        return p[0]*_numpy.sin(2*pi*x/p[1]+p[2]) + p[3] + p[4]*x

    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # guess the offset
        p[3] = (max(ydata)+min(ydata))/2.0

        # guess the amplitude
        p[0] = (max(ydata)-min(ydata))/2.0

        p[4] = 0

        # guess the wavelength and phase
        n1 = _fun.index_next_crossing(p[3],ydata,0)
        n2 = _fun.index_next_crossing(p[3],ydata,n1+3)
        if n1<0 or n2<0:
            p[1] = (xdata[-1]-xdata[0])/3
            p[2] = 0
        else:
            p[1] = (xdata[n2]-xdata[n1])
            p[2] = -2*pi*xdata[n1]/p[1]



        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)






class quadratic(model_base):

    function_string = "p[0] + p[1]*x + p[2]*x*x"
    pnames = ["a0", "a1", "a2"]

    # this must return an array!
    def background(self, p, x):
        return self.evaluate(p,x)

    def evaluate(self, p, x):
        return p[0] + p[1]*x + p[2]*x*x

    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # guess the slope and intercept
        p[0] = ydata[len(xdata)/2]
        p[1] = (ydata[xbi2]-ydata[xbi1])/(xdata[xbi2]-xdata[xbi1])
        p[2] = 0.0

        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)

class cubic(model_base):

    function_string = "p[0] + p[1]*x + p[2]*x*x + p[3]*x*x*x"
    pnames = ["a0", "a1", "a2", "a3"]

    # this must return an array!
    def background(self, p, x):
        return self.evaluate(p,x)

    def evaluate(self, p, x):
        return p[0] + p[1]*x + p[2]*x*x + p[3]*x*x*x

    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # guess the slope and intercept
        p[0] = ydata[len(xdata)/2]
        p[1] = (ydata[xbi2]-ydata[xbi1])/(xdata[xbi2]-xdata[xbi1])
        p[2] = 0.0
        p[3] = 0.0

        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)


class quartic(model_base):

    function_string = "p[0] + p[1]*x + p[2]*x*x + p[3]*x*x*x + p[4]*x*x*x*x"
    pnames = ["a0", "a1", "a2", "a3", "a4"]

    # this must return an array!
    def background(self, p, x):
        return self.evaluate(p,x)

    def evaluate(self, p, x):
        return p[0] + p[1]*x + p[2]*x*x + p[3]*x*x*x + p[4]*x*x*x*x

    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # guess the slope and intercept
        p[0] = ydata[len(xdata)/2]
        p[1] = (ydata[xbi2]-ydata[xbi1])/(xdata[xbi2]-xdata[xbi1])
        p[2] = 0.0
        p[3] = 0.0
        p[4] = 0.0

        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)





