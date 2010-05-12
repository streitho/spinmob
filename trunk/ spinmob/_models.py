import os
import scipy as _scipy
import numpy as _numpy
import pylab as _pylab
from matplotlib.font_manager import FontProperties as _FontProperties
import spinmob as _spinmob

_tweaks = _spinmob.plot.tweaks

import _functions as _fun


pi   = 3.1415926535
u0   = 1.25663706e-6
uB   = 9.27400949e-24
e    = 1.60217e-19
h    = 6.626068e-34
hbar = h/(2*pi)

#
# Fit function based on the model class
#
def fit(model, command="", settings={}, **kwargs):
    """
    Load a bunch of data files and fit them. kwargs are sent to "data.load_multiple()" which
    are then sent to "data.standard()". Useful ones to keep in mind:

    for loading:    paths, default_directory
    for data class: xscript, yscript, eyscript

    See the above mentioned functions for more information.
    """

    # Have the user select a bunch of files.
    ds = _spinmob.data.load_multiple(**kwargs)

    for d in ds:
        print '\n\n\nFILE:', ds.index(d)+1, '/', len(ds)
        result = model.fit(d, command, settings)

        # make sure we didn't quit.
        if result['command'] == 'q': return

        # prepare for the next file.
        command=''
        if result.has_key('settings'):
            settings = result['settings']



#
# Classes
#
class model_base:

    # this is something the derived classes must do, to define
    # their fit variables
    pnames = ["height", "center", "width", "height2", "offset"]

    D = None

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

    # this can be another overridden function, but it's not necessary
    # all THIS version does is try to guess the offset and then
    # tries to write the offset guess parameter (if "offset" is defined)
    def guess_background(self, ydata, xbi1=0, xbi2=-1):
        self.set_parameter("offset", (ydata[xbi1]+ydata[xbi2])/2.0)

    # You can override these functions if you like (not fully implemented yet)
    def append_header_to_file(self, path, data):
        # write the header to the fit peaks fill
        append_to_file(fit_file, "\n\nfit_data: model "+str(self))
        if invert: append_to_file(fit_file, " (DATA INVERTED)")
        append_to_file(fit_file, "\n"+"data: "+str(data)+"\n")

        # print the key names and the parameters
        key_names = []
        for k in data.constants.keys():
            append_to_file(fit_file, k+" ")
            key_names.append(k) # make sure we remember the order of keys
        for n in model.pnames:
            append_to_file(fit_file, "fit-" + n + " ")
        append_to_file(fit_file, "\n")


    def append_pfit_to_file(self, path):
        return

    #
    #
    #  These functions are generally not overwritten
    #
    #
    def chi_squared(self, p, xdata, ydata, yerror=None):
        """
        This returns a single number that is the chi squared for a given set of parameters p

        This is currently not in use for the optimization.  That uses residuals.
        """

        if yerror==None: yerror=1
        return sum( (ydata - self.evaluate(p,xdata))**2 / yerror**2)

    def optimize(self, xdata, ydata, yerror=None, p0="internal"):
        """
        This actually performs the optimization on xdata and ydata.
        p0="internal"   is the initial parameter guess, such as [1,2,44.3].
                        "internal" specifies to use the model's internal guess result
                        but you better have guessed!
        returns the fit p from scipy.optimize.leastsq()
        """
        if p0 == "internal": p0 = self.p0
        if self.D == None: return _scipy.optimize.leastsq(self.residuals, p0, args=(xdata,ydata,yerror,), full_output=1)
        else:              return _scipy.optimize.leastsq(self.residuals, p0, args=(xdata,ydata,yerror,), full_output=1, Dfun=self.jacobian, col_deriv=1)

    def residuals(self, p, xdata, ydata, yerror=None):
        """
        This function returns a vector of the differences between the model and ydata, scaled by the error
        (yerror could be a scalar or a vector of length equal to ydata)
        """

        if yerror==None: yerror=1
        return (ydata - self.evaluate(p,xdata))/_numpy.absolute(yerror)

    def residuals_variance(self, p, xdata, ydata, yerror=None):
        """
        This returns the variance of the residuals, or chi^2/DOF.
        """

        return self.chi_squared(p, xdata, ydata, yerror)/(len(xdata)-len(p))

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
            i = self.pnames.index(name)
            self.p0[i] = float(value)
            return True

        except:
            print name, "is not a valid variable"
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
        This generates xdata, ydata, and yerror from the three scripts
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
                            "figure"            : 1}
        if d.eyscript == None: default_settings["auto_error"] = True

        # fill in the non-supplied settings with defaults
        for k in default_settings.keys():
            if not k in settings.keys():
                settings[k] = default_settings[k]

        # Initialize the fit_parameters (we haven't any yet!)
        fit_parameters = None

        # set up the figure
        fig = _pylab.figure(settings["figure"])
        fig.clear()
        axes1 = fig.add_axes([0.10, 0.08, 0.73, 0.70])
        axes2 = fig.add_axes([0.10, 0.79, 0.73, 0.13])
        axes2.xaxis.set_ticklabels([])

        # start by plotting the data, no error bars.
        d.get_data()
        axes1.plot(d.xdata, d.ydata, linestyle='', marker='D', mfc='blue', mec='w', label='data')

        # Now keep trying to fit until the user says its okay or gives up.
        while True:

            # Start by formatting the previous plot

            axes2.xaxis.set_ticklabels([])

            # come up with a title
            title1 = d.path

            # second line of the title is the model
            title2 = "eyscript="+str(d.eyscript)+", model:"+str(self.__class__).split()[0][0:]

            title3 = "(no fit performed)"
            if not settings["skip"]:
                title3 = []
                for i in range(0,len(self.pnames)):
                    title3.append(self.pnames[i]+"=%.4g+/-%.2g" % (fit_parameters[i], fit_errors[i]))
                title3 = _fun.join(title3,", ")

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
            if not settings["skip"]: _tweaks.auto_zoom(axes2)
            _pylab.draw()

            _tweaks.raise_figure_window()
            _tweaks.raise_pyshell()





            # the only way we optimize is if we hit enter.
            settings["skip"] = True

            # If last command is None, this is the first time. Parse the initial
            # command but don't ask for one.
            if command == "":
                print
                print "min=" + str(settings['min']) + ", max="+str(settings['max'])
                command = raw_input("Command (<enter> to fit, 'h' for help): ").strip()

            # first check and make sure the command isn't one of the simple ones
            if command.lower() in ['h', 'help']:
                print
                print "COMMANDS"
                print "  <enter>    Do more iterations."
                print "  g          Guess and show the guess."
                print "  n          No, this is not a good fit. Move on."
                print "  y          Yes, this is a good fit. Move on."
                print "  u          Same as 'y' but use fit as the next guess."
                print "  p          Call the printer() command."
                print "  q          Quit."
                print
                print "SETTINGS"
                for key in settings.keys(): print "  "+key+" =", settings[key]
                print

                command=""
                continue

            elif command.lower() in ['q', 'quit', 'exit']:
                return {'command':'q'}

            elif command.lower() in ['g', 'guess']:
                settings['guess'] = None
                settings['show_guess'] = True

            elif command.lower() in ['y', 'yes','u','use']:

                if fit_parameters==None:
                    print "Can't say a fit is good with no fit!"

                elif settings['save_file']:
                    # If this is a good fit. Add relevant information to the header then save
                    d.insert_header("fit_model", str(self.__class__).split()[0][0:])
                    for n in range(len(self.pnames)):
                        d.insert_header("fit_"+self.pnames[n], fit_parameters[n])

                    d.insert_header("fit_reduced_chi_squared",fit_reduced_chi_squared)
                    d.insert_header("fit_errors",             fit_errors)

                    # build the correlations array (not a 2-d array)
                    d.insert_header("fit_correlations",       fit_correlation)

                    d.insert_header("fit_min",      settings['min'])
                    d.insert_header("fit_max",      settings['max'])
                    d.insert_header("fit_smooth",   settings['smooth'])
                    d.insert_header("fit_coarsen",  settings['coarsen'])

                    # auto-generate the new file name
                    directory, filename = os.path.split(d.path)
                    new_path = directory + os.sep + settings['file_tag'] + filename

                    # save the file
                    d.save_file(new_path)

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

            elif command.lower() in ['']:
                settings["skip"] = False

            else:
                # now parse it (it has the form "min=2, max=4, plot_all=True")
                s = command.split(',')
                for c in s:
                    try:
                        [key, value] = c.split('=')
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

            # now that that's out of the way, we're ready to fit.
            if settings["skip"]: print "Plotting but not optimizing..."
            else:                print "Beginning fit routine..."

            # if we're doing auto error, start with an array of 1's
            if d.yerror==None: d.yerror = d.xdata*0.0 + 1.0

            # now sort the data in case it's jaggy!
            matrix_to_sort = _numpy.array([d.xdata, d.ydata, d.yerror])
            sorted_matrix  = _fun.sort_matrix(matrix_to_sort, 0)
            d.xdata  = sorted_matrix[0]
            d.ydata  = sorted_matrix[1]
            d.yerror = sorted_matrix[2]

            # now trim all the data based on xmin and xmax
            xmin = settings["min"]
            xmax = settings["max"]
            if xmin==None: xmin = min(d.xdata)+1
            if xmax==None: xmax = max(d.xdata)+1
            [x, y, ye] = _fun.trim_data(d.xdata, d.ydata, d.yerror, [xmin, xmax])

            # smooth and coarsen
            [x,y,ye] = _fun.smooth_data(x,y,ye,settings["smooth"])
            [x,y,ye] = _fun.coarsen_data(x,y,ye,settings["coarsen"])

            # now do the first optimization. Start by guessing parameters from
            # the data's shape. This writes self.p0
            if settings["guess"]==None:
                self.guess(x, y, settings["xb1"], settings["xb2"])
            else:
                self.write_to_p0(settings['guess'])
            print "  GUESS:", self.p0


            # now do the first optimization
            if not settings["skip"]:
                fit_output = self.optimize(x, y, ye, self.p0)

                # If we're doing auto error, now we should scale the error so that
                # the reduced xi^2 is 1
                sigma_y = 1.0
                if settings["auto_error"]:

                    # guess the correction to the y-error we're fitting (sets the reduced chi^2 to 1)
                    sigma_y = _numpy.sqrt(self.residuals_variance(fit_output[0],x,y,ye))
                    print "    initial reduced chi^2 =", sigma_y**2
                    print "    scaling error by", sigma_y, "and re-optimizing..."
                    ye       = sigma_y*ye
                    d.yerror = sigma_y*d.yerror

                    # optimize with new improved errors, using the old fit to start
                    fit_output = self.optimize(x,y,ye,p0=fit_output[0])

                # Now that the fitting is done, show the output

                # grab all the information from fit_output
                fit_parameters = fit_output[0]
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
                for n in range(0,len(self.pnames)): print "  "+self.pnames[n]+" =", fit_parameters[n], "+/-", fit_errors[n]




            # get the data to plot
            if settings["plot_all"]:
                x_plot  = d.xdata
                y_plot  = d.ydata
                ye_plot = d.yerror*sigma_y

                # smooth and coarsen
                [x_plot, y_plot, ye_plot] = _fun.smooth_data(x_plot, y_plot, ye_plot, settings["smooth"])
                [x_plot, y_plot, ye_plot] = _fun.coarsen_data(x_plot,y_plot, ye_plot, settings["coarsen"])
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
            if settings["show_error"] and not fit_parameters==None:
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












class complete_lorentz_flat_background(model_base):

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
        p[1] = xdata[index(max(ydata-p[4]), ydata-p[4])] # center

        # guess the asymmetric part from the minimum and background
        p[3] = 0

        # guess the halfwidth
        p[2] = (max(xdata)-min(xdata))/12.0

        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)




class even_polynomial(model_base):

    # set up the parameter structure here
    def __init__(self, order=4):
        self.order = order

        # create the pnames based on the supplied order
        self.pnames = []
        self.p0     = []
        for n in range(0, self.order/2+1):
            self.pnames.append("p"+str(n*2))
            self.p0.append(0)

    # define the function
    def background(self, p, x):
        return(0.0*x + p[0])

    def evaluate(self, p, x):
        y = 0.0
        for n in range(0,len(p)):
            y += p[n] * x**(2*n)
        return(y)

    # come up with a routine for guessing p0
    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # this one's easy to guess.
        p[0] = ydata[_fun.index_nearest(0, xdata)]

        # make the rest 0
        for n in range(1, len(p)):
            p[n] = 0.0

        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)


class exponential_offset(model_base):

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



class kittel_medium_axis(model_base):

    global hbar, pi, uB
    pnames = ["Bzy", "Byx"]

    # define the function
    def background(self, p, x): return 0*x # the 0.0*x is so you get an array from an array

    # main function
    def evaluate(self, p, x):
        xa = _numpy.absolute(x)
        return 2*1e-9*(uB/hbar)*((xa+p[0])*(xa-p[1]))**0.5/(2*pi)

    # come up with a routine for guessing p0
    def guess(self, xdata, ydata, xbi1=0, xbi2=-1):
        # first get the appropriate size array
        p=self.p0

        # guess the background
        p[0] = 1.0
        p[1] = 0.05

        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)




class linear(model_base):

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


class lorentz_linear_background(model_base):

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
        p[1] = xdata[index(max(ydata-p[4]), ydata-p[4])] # center

        # guess the halfwidth
        p[2] = (max(xdata)-min(xdata))/12.0

        # write these values to self.p0, but avoid the guessed_list
        self.write_to_p0(p)



class parabola(model_base):

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






#
# Handy Functions
#
def index(value, array):
    # simply returns the index of the first
    # we need this to deal with numpy arrays too
    for n in range(0,len(array)):
        if value == array[n]:
            return(n)
    return(-1)

