import numpy as _numpy
import pylab as _pylab
import spinmob as _s

from matplotlib.font_manager import FontProperties as _FontProperties
from scipy import optimize as _optimize

import _functions as _fun           ; reload(_fun)
import _pylab_tweaks as _tweaks     ; reload(_tweaks)
import _models                      ; reload(_models)
import _dialogs                     ; reload(_dialogs)
import _data_types                  ; reload(_data_types)

from numpy import *

#
# Fit function based on the model class
#
def fit(f='a*x+b; a; b', bg='0.0', command="", settings={}, **kwargs):
    """
    Load a bunch of data files and fit them. kwargs are sent to "data.load_multiple()" which
    are then sent to "data.standard()". Useful ones to keep in mind:

    for loading:    paths, default_directory
    for data class: xscript, yscript, eyscript

    See the above mentioned functions for more information.

    f (the function) must be a semicolon-delimted string with the first element being the function
    and the remainder being the variables to fit.

    bg (the background) is also a function string like the first element of f.
    """

    # generate the model
    model = _s.models.curve(f, bg, globals())
    fit_model(model, command,    settings,    **kwargs)

def fit_model(model, command="", settings={}, **kwargs):
    """
    Load a bunch of data files and fit them. kwargs are sent to "data.load_multiple()" which
    are then sent to "data.standard()". Useful ones to keep in mind:

    for loading:    paths, default_directory
    for data class: xscript, yscript, eyscript

    See the above mentioned functions for more information.
    """

    # Have the user select a bunch of files.
    ds = _s.data.load_multiple(**kwargs)

    for d in ds:
        print '\n\n\nFILE:', ds.index(d)+1, '/', len(ds)
        model.fit_parameters = None
        result = model.fit(d, command, settings)

        # make sure we didn't quit.
        if result['command'] == 'q': return

        # prepare for the next file.
        command=''
        if result.has_key('settings'):
            settings = result['settings']




























def _fit(data=_data_types.standard(), model=_models.parabola(), auto_error=1, show_guess=0, show_error=1, plot_all=0, skip_first_try=0, subtract=0, clear_plot=1, invert=0, autofile=1, paths="ask"):
    """

    This is antiquated and ugly.

    data                (instance of class) data to extract from the files
    model               (instance of class) fitting class
    show_guess          show the guess on the fit plots
    skip_first_try      skip the first optimization attempt
    clear_plot=1        should we clear the plot each time?
    invert=0            invert the data?
    autofile=1          generate fit files automatically based on fit directory

    """

    # initialize some model variables
    model.plot_all       = plot_all
    model.auto_error     = auto_error
    model.subtract       = subtract
    model.show_guess     = show_guess
    model.show_error     = show_error
    model.skip_first_try = skip_first_try
    model.hold           = 0
    model.last_command   = ''
    model.have_a_guess   = 0
    model.clear_plot     = clear_plot
    model.smoothing      = 0
    model.coarsen        = 0
    model.auto_trim      = 0



    # keep looping until cancel or "quit" happens
    time_to_quit = 0
    while not time_to_quit:

        # have the user select a file
        print paths
        if paths=="ask": paths = _dialogs.MultipleFiles('DIS AND DAT|*.*', default_directory=data.directory)
        else:            time_to_quit = True

        # quit if we cancelled
        if paths==None: return

        # if not set to manual fit file saving, assemble the location
        if autofile:
            # figure out the fit file based on the directory
            # first get the directory from the first path
            s = paths[0].split(_prefs.path_delimiter)

            # generate the file name from the directory
            s.pop(-1) # rid ourselves of the file name
            fit_folder = s.pop(-1)

            # now generate the fit file path
            fit_file = _fun.join(s, _prefs.path_delimiter)+_prefs.path_delimiter+fit_folder+".fits"

        else: fit_file = _dialogs.Save("*.fits", text='SAVE FIT RESULTS TO WHERE?', default_directory='default_directory')

        # if we got a '' it means we hit "cancel"
        if fit_file == '': return

        if len(fit_file.split('.'))<=1 or not fit_file.split('.')[-1] == "fits":
            fit_file = fit_file + ".fits"

        # write the header to the fit peaks fill
        _fun.write_to_file(fit_file, "fit_data: model "+str(model))
        if invert: _fun.append_to_file(fit_file, " (DATA INVERTED)")
        _fun.append_to_file(fit_file, "\n"+"data: "+str(data)+"\n")

        # print the key names and the parameters
        key_names = []
        for k in data.constants.keys():
            _fun.append_to_file(fit_file, k+" ")
            key_names.append(k) # make sure we remember the order of keys

        # print the fit headers
        for n in model.pnames:
            _fun.append_to_file(fit_file, "fit_" + n + " ")

        # print the error headers
        for n in model.pnames:
            _fun.append_to_file(fit_file, "fit_" + n + "_error ")

        # now write the upper half of the covariance matrix headers
        # with names like "cov_height_height2" and stuff
        for n in range(0, len(model.pnames)-1):
            for m in range(n+1, len(model.pnames)):
                c = [model.pnames[n], model.pnames[m]]
                c.sort()
                _fun.append_to_file(fit_file, "cov_" + _fun.join(c, '_') + ' ')

        # print the reduced chi^2 header
        _fun.append_to_file(fit_file, "reduced_chi_squared")

        _fun.append_to_file(fit_file, "\n")



        #########################
        # done writing the header
        #########################



        model.last_command = ''

        # for each path, open the file, get the data, and fit it
        model.last = len(paths)-1
        for model.m in range(0, len(paths)):

            # if we're supposed to skip the first try, tell the model
            if  model.skip_first_try:  model.skip_next_optimization=1
            else:                      model.skip_next_optimization=0

            # fill up the xdata, ydata, and key
            data.load_file(paths[model.m])
            data.get_data()

            # now for each key, print the value in the fit file
            for k in key_names:  _fun.append_to_file(fit_file, str(data.constants[k]).replace(' ', '_') + ' ');

            # now start the query loop to make sure and fit or move on
            pfit = _interactive_fitting_loop(model, data)

            # now deal with the output
            if pfit == "quit":
                return "quit"
            elif pfit == "next":
                print "okay, moving on..."
                _fun.append_to_file(fit_file,'\n')

            # otherwise, pfit["parameters"] is a good set of fit parameters
            # pfit["errors"] are the (correlated!) errors
            # and pfit["covariance"] is the covariance matrix
            elif not pfit==None:
                fit_peaks = open(fit_file, 'a')

                # write the parameter values
                for a in pfit["parameters"]: fit_peaks.write(str(a) + ' ')

                # write the error values
                for a in pfit["errors"]: fit_peaks.write(str(a) + ' ')

                # now write the upper half of the covariance matrix
                # with names like "cov_height_height2" and stuff
                for n in range(0, len(pfit["errors"])-1):
                    for m in range(n+1, len(pfit["errors"])):
                        fit_peaks.write(str(pfit["covariance"][n][m]) + ' ')

                # finally print the reduced chi^2
                fit_peaks.write(str(pfit["reduced_chi_squared"]))

                # finish up
                fit_peaks.write('\n')
                fit_peaks.close()

        # end for m in files
        paths = "ask"

    return



def _interactive_fitting_loop(model, data, auto_fast=False):
    """
    This is totally antiquated and nasty.
    """


    # get a first guess at the background positions
    x_background_index1 = 1
    x_background_index2 = -2
    xmin = min(data.xdata)
    xmax = max(data.xdata)

    # get the current axes
    f = _pylab.gcf()
    if model.clear_plot: f.clear()
    axes  = f.add_axes([0.10, 0.08, 0.73, 0.70])
    axes2 = f.add_axes([0.10, 0.79, 0.73, 0.13])
    axes2.xaxis.set_ticklabels([])

    plot_all = 1
    command = model.last_command.strip()

    # while not done with a given file
    while True:
        ask_again = False

        # enter means more iterations
        if command == '\n' or command == '':
            print "Time to do some stuff."

        # y is good
        elif command == 'y':
            model.have_a_guess = False
            return pfit

        # u means "good, and also use this fit for the next guess"
        elif command == 'u':
            model.p0 = pfit["parameters"]
            model.have_a_guess = True
            return pfit

        # b means "good, and also use this to fit for the next guess, but guess the background"
        elif command == 'b':
            model.p0 = pfit["parameters"]
            model.have_a_guess = "all but background"
            return pfit

        # p means print
        elif command == 'p':
            _s.printer()
            ask_again = True

        # q means quit everything
        elif command == 'q' or command == 'quit' or command == 'exit':
            return "quit"

        # n means "next data set.  Skip this!"
        elif command == 'n':
            model.have_a_guess = False
            return "next"

        elif command == 'c':
            model.last_command=''
            ask_again=True

        elif command == 'clear':
            axes.clear()
            _pylab.draw()
            ask_again=True

        elif command == 'g' or command == "guess":
            model.guess(data.xdata,data.ydata)
            if model.clear_plot: axes.clear()
            if not data.eydata==None: axes.errorbar(data.xdata, data.ydata, "^", color="blue", mec="w", label="data", yerr=data.eydata)
            else:                     axes.plot(    data.xdata, data.ydata, "^", color="blue", mec="w", label="data")

            print "shit", data.eydata

            axes.plot(data.xdata, model.evaluate(model.p0, data.xdata), label="guess")

            # set the axes labels
            axes.set_title("guess result (no fitting)")
            axes.set_xlabel(data.xlabel)
            axes.set_ylabel(data.ylabel)
            format_figure(axes.figure)
            ask_again=True


        elif command == 'h' or command == 'help':
            print "---Standalone Commands---"
            print "y                 Yes!  Good fit!  Save it and move on."
            print "u                 Like 'y' but use the fit parameters as the next guess."
            print "b                 Like 'u' but still try to guess the background."
            print "n                 No!  Next!"
            print "q                 Quit."
            print "p                 Print this plot."
            print "c                 Clear all options."
            print "g                 Guess at parameters."
            print "clear             clear plot contents now."
            print "<enter>           Do more iterations."

            print "\n---Fitting Variables, of form <name>=<value>---"
            print "---  's' at start means 'same as before'    ---"
            print "min               data to include in the fit minimum"
            print "max               data to include in the fit maximum"
            print "xb1               first x-value of data to estimate background during guess"
            print "xb2               second x-value of data to estimate background during guess"
            print "hold              don't do a fit until this is 0"
            print "skip_first_try    should we skip the first optimization attempts?"
            print "subtract          should we subtract the background?"
            print "show_guess        should we plot the guess?"
            print "show_error        should we plot the error bars?"
            print "plot_all          should we plot the entire data set with the fit?"
            print "smoothing         how much to smooth the data by before fitting?"
            print "coarsen           how much to coarsen the data before fitting?"
            print "\n"
            ask_again = True

        # otherwise, the input should be a list of variables and values
        # like "center=4, width=2, min=3, max=5"
        else:
            try:
                # first we have to split the command by spaces to get the values
                s = command.split(',')

                # get the new and old variable list, and sub in/append
                # accordingly.  Then reassemble the string for the normal
                # processing

                # get the new variable list
                new = {}
                for n in range(0,len(s)):
                    try:
                        a = s[n].split('=')
                        if len(a)==2: new[a[0].strip()]=a[1].strip()
                    except:
                        print "Unpacking the new elements did not work.  No biggy."

                # get the old variable list
                old={}
                old_s = model.last_command.split(',')
                for n in range(0,len(old_s)):
                    try:
                        a = old_s[n].split('=')
                        if len(a)==2: old[a[0].strip()]=a[1].strip()
                    except:
                        print old_s
                        print "Unpacking the old elements did not work.  No biggy."
                        return


                # now for each new variable either swap it in or make
                # a new value
                k = new.keys()
                for n in range(0,len(k)):
                    old[k[n]] = new[k[n]]

                # now reassemble the complete command (into a string)
                k = old.keys()
                command = ''
                for n in range(0,len(k)):
                    command += k[n]+'='+old[k[n]]+', '
                command=command.strip()



                # save the command string for future use
                model.last_command = command




                # now loop over all the arguments
                xmin = min(data.xdata)
                xmax = max(data.xdata)
                model.x_background_index1=1
                model.x_background_index2=-2

                s = command.split(',')
                for n in range(0,len(s)):

                    # get the variable to be set and the value
                    try:
                        a=s[n].split('=')
                        if len(a)==2:
                            variable = a[0].strip()
                            value    = float(a[1].strip())

                            model.guessed_list = []

                            # first look for special strings
                            if variable == 'min': # min data range
                                xmin = value
                                x_background_index1 = 1
                            elif variable == 'max': # max data range
                                xmax = value
                                x_background_index2 = -2
                            elif variable == 'hold':
                                model.hold = value
                            elif variable == 'skip_first_try': # should we skip the first optimization attempt?
                                model.skip_first_try = value
                            elif variable == 'subtract': # should we subtract the background when plotting?
                                model.subtract = value
                            elif variable == 'show_guess': # draw the guess
                                model.show_guess = value
                            elif variable == 'plot_all': # show all the data
                                model.plot_all = value
                            elif variable == 'smoothing': # presmooth the data
                                model.smoothing = int(value)
                            elif variable == 'coarsen':
                                model.coarsen = int(value)
                            elif variable == 'show_error': # include the error bars?
                                model.show_error = value
                            elif variable == 'auto_error': # should we auto-scale the error bars to make red. chi^2 1?
                                model.auto_error = value
                            elif variable == 'auto_trim':
                                if value == 0:
                                    model.auto_trim = False
                                else:
                                    model.auto_trim       = True
                                    model.auto_trim_plus  = value
                                    model.auto_trim_minus = value

                            # otherwise we set the model p0 parameters
                            else:
                                if model.set_parameter(variable, value):
                                    model.guessed_list.append(variable)

                    except:
                        print '"'+s[n]+'" did not unpack.  Learn how to pack, Sillypants McCantpack.'
                        model.skip_next_optimization = 1

            # done with "else parse the inputs"

            except:
                print "Commands should look like 'min=3, max=4, a0=3'"



        # If we have good enough input to try a fit
        if not ask_again:

            # make a new data range to look at
            print "generating new array to fit"

            # store local copies of the data
            smooth_x = _numpy.array(data.xdata)
            smooth_y = _numpy.array(data.ydata)
            smooth_ye= _numpy.array(data.eydata)

            # presmooth the ydata first
            _fun.smooth_array(smooth_y, model.smoothing)
            [smooth_x,smooth_y,smooth_ye] = _fun.coarsen_data(smooth_x,smooth_y,smooth_ye,model.coarsen)

            # trim the data down into new arrays
            [x, y, ye] = _fun.trim_data(smooth_x, smooth_y, smooth_ye, [xmin, xmax])

            if len(x) <= len(model.p0)+1: # need at least 3 data points
                print "not enough data to fit!"
                model.skip_next_optimization = 1



            # guess at the initial parameters
            if not model.have_a_guess:
                model.guess(x, y, x_background_index1, x_background_index2)
                model.have_a_guess=True
            elif model.have_a_guess == "all but background":
                print "guessing the background..."
                model.guess_background(y, x_background_index1, x_background_index2)

            # now fit!
            pfit = {}
            if model.hold: model.skip_next_optimization = 1

            if model.skip_next_optimization:
                print "SKIPPING OPTIMIZATION..."
                pfit["parameters"] = model.p0
            else:
                print "OPTIMIZING..."


                if model.auto_trim:
                    # optimize roughly
                    fit_output = model.optimize(x,y,ye)

                    print "thnick thnick:", [fit_output[0][1]-model.auto_trim_minus*fit_output[0][2], fit_output[0][1]+model.auto_trim_plus*fit_output[0][2]]

                    # trim the data down into new arrays
                    [x, y, ye] = _fun.trim_data(smooth_x, smooth_y, smooth_ye,
                                    [fit_output[0][1]-model.auto_trim_minus*fit_output[0][2],
                                     fit_output[0][1]+model.auto_trim_plus*fit_output[0][2]])

                    # optimize with new range using the old fit to start
                    fit_output = model.optimize(x,y,ye,p0=fit_output[0])

                    print "thnick thnick:", [fit_output[0][1]-model.auto_trim_minus*fit_output[0][2], fit_output[0][1]+model.auto_trim_plus*fit_output[0][2]]

                    # trim the data down into new arrays
                    [x, y, ye] = _fun.trim_data(smooth_x, smooth_y, smooth_ye,
                                    [fit_output[0][1]-model.auto_trim_minus*fit_output[0][2],
                                     fit_output[0][1]+model.auto_trim_plus*fit_output[0][2]])

                    # optimize with new range using the old fit to start
                    fit_output = model.optimize(x,y,ye,p0=fit_output[0])



                if model.auto_error:
                    # optimize the first time (rough)
                    fit_output = model.optimize(x,y,ye)

                    # guess the correction to the y-error we're fitting (sets the reduced chi^2 to 1)
                    sigma_y = _numpy.sqrt(model.residuals_variance(fit_output[0],x,y,ye))

                    print "reduced chi^2 =", sigma_y**2
                    print "scaling error by", sigma_y, "and re-optimizing..."

                    if ye==None: ye = x*0.0+sigma_y
                    else:        ye = sigma_y*ye

                    # optimize with new improved errors, using the old fit to start
                    fit_output = model.optimize(x,y,ye,p0=fit_output[0])

                else:
                    fit_output = model.optimize(x,y,ye)



                # get fit parameters
                pfit["parameters"] = fit_output[0]
                pfit["covariance"] = fit_output[1]
                pfit["reduced_chi_squared"] = model.residuals_variance(pfit["parameters"],x,y,ye)

                print "reduced chi^2 is now",pfit["reduced_chi_squared"]
                if pfit["covariance"] is not None:
                    # get the error vector and correlation matrix from (scaled) covariance
                    [errors, correlation] = _fun.decompose_covariance(pfit["covariance"])
                else:
                    print "WARNING: No covariance matrix popped out of model.optimize()"
                    errors = pfit["parameters"]
                    correlation = None


                pfit["errors"]      = _numpy.array(errors)
                pfit["correlation"] = correlation

            # now come up with arrays for the fit curve
            if model.plot_all:
                x_plot  = _numpy.array(x)
                y_plot  = _numpy.array(y)
                if data.eydata==None: ye_plot = x_plot*0.0+ye[0]
                else:                 ye_plot = _numpy.array(ye)
            else:
                x_plot  = _numpy.array(x)
                y_plot  = _numpy.array(y)
                ye_plot = _numpy.array(ye)

            # sort the data results in case the data is jaggy.
            matrix_to_sort = _numpy.array([x_plot, y_plot, ye_plot])
            sorted_matrix = _fun.sort_matrix(matrix_to_sort, 0)
            x_plot  = sorted_matrix[0]
            y_plot  = sorted_matrix[1]
            ye_plot = sorted_matrix[2]


            # loop over the x-data and get the other curves to plot
            xfit        = x_plot
            yfit        = []
            yguess      = []
            yback       = []
            yguessback  = []
            for z in x_plot:
                yfit.append(model.evaluate(pfit["parameters"], z))
                yguess.append(model.evaluate(model.p0, z))
                yback.append(model.background(pfit["parameters"], z))
                yguessback.append(model.background(model.p0, z))

            if model.subtract:
                print "subtracting the background..."
                y_plot = y_plot - yback
                yfit   = yfit - yback
                yguess = yguess - yguessback
                yback  = yback*0.0
                yguessback = yguessback*0.0

            # now plot!

            # get rid of the old plot
            if model.clear_plot:
                axes.clear()
                axes2.clear()

            # plot the data, fit, background
            # if we're supposed to, plot the guess too
            if model.show_guess:
                axes.plot(xfit , yguess,     color='gray', label='guess')
                axes.plot(xfit,  yguessback, color='gray', label='guess background')
            axes.plot(xfit, yback, color='red', label='fit background')

            # plot the actual data

            # if there's no error data, or we're not supposed to show it, or we're on hold, just plot the raw data
            if ye_plot==None or not model.show_error or model.skip_next_optimization:
                  axes.plot    (x_plot, y_plot,          linestyle='', marker='D', mfc='blue', mec='w', label='data')

            # otherwise plot with error bars
            else: axes.errorbar(x_plot, y_plot, ye_plot, linestyle='', marker='D', mfc='blue', mec='w', label='data')

            # now plot the fit
            axes.plot(xfit, yfit,  color='red', label='fit')


            # now plot the residuals on the upper graph
            if not model.skip_next_optimization:
                axes2.plot(x_plot, 0*x_plot, linestyle='-', color='k')
                axes2.errorbar(x_plot, (y_plot-yfit)/ye_plot, ye_plot*0+1, linestyle='', marker='o', mfc='blue', mec='w')
                axes2.xaxis.set_ticklabels([])

            # come up with a title
            title1 = []
            for key in data.constants.keys():
                title1.append(str(key)+"="+str(data.constants[key]))
            title1 = _fun.join(title1,", ")

            # second line of the title is the model
            title2 = str(data.__class__) + ", " + str(model.__class__)

            title3 = "(no fit performed)"
            if not model.skip_next_optimization:
                title3 = []
                for i in range(0,len(model.pnames)):
                    title3.append(model.pnames[i]+"=%(p).4g+/-%(pe).2g" %{"p":pfit["parameters"][i], "pe":pfit["errors"][i]})
                title3 = _fun.join(title3,", ")

                # ask if it looks nice
                print
                for j in range(0, len(pfit["parameters"])):
                    print model.pnames[j]+' = '+str(pfit["parameters"][j])+" +/- "+str(pfit["errors"][j])

            axes2.set_title(title1+"\n"+title2+"\nFIT: "+title3)
            axes.set_xlabel(data.xlabel)
            axes.set_ylabel(data.ylabel)

            # set the position of the legend
            axes.legend(loc=[1.01,0], borderpad=0.02, prop=_FontProperties(size=7))

            # set the label spacing in the legend
            axes.get_legend().labelsep = 0.01

            # set up the title label
            axes2.title.set_horizontalalignment('right')
            axes2.title.set_size(8)
            axes2.title.set_position([1.0,1.010])

            _tweaks.auto_zoom(axes)

            if not model.skip_next_optimization:
                _tweaks.auto_zoom(axes2)

            # update the plot
            _pylab.draw()


            if model.skip_next_optimization:
                model.skip_next_optimization = 0;



        _tweaks.raise_figure_window()
        _tweaks.raise_pyshell()

        # now ask again
        print
        if auto_fast: command="y"
        else:
            print '%(a)i/%(b)i last: "' % {'a':model.m+1, 'b':model.last+1} + model.last_command + '"'
            command = raw_input('what now? ').strip()






def _fit_massive(data, model=_models.complete_lorentz_flat_background(), show_guess=1, skip_first_try=0, clear_plot=1, invert=0, autofile=1):
    d = _dialogs.Directory();
    if d == "": return
    contents = _os.listdir(d) # doesn't include root path
    contents.sort()

    for file in contents:
        print "Directory: "+file
        if _os.path.isdir(d+_prefs.path_delimiter+file):
            paths = _glob.glob(d+_prefs.path_delimiter+file+"\\*.DAT")
            print paths
            if fit(data, model, show_guess, skip_first_try, clear_plot, invert, autofile, paths) == "quit": return

    return
