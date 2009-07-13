import scipy as _scipy
import numpy as _numpy

import _functions as _fun


pi   = 3.1415926535
u0   = 1.25663706e-6
uB   = 9.27400949e-24
e    = 1.60217e-19
h    = 6.626068e-34
hbar = h/(2*pi)

#
# Classes
#
class model_base:

    # this is something the derived classes must do, to define
    # their fit variables
    pnames = ["height", "center", "width", "height2", "offset"]

    # dumb stuff I should probably make less silly (needed for interactive_fitting_loop())
    clear_plot   = True
    last_command = ""
    smoothing    = 0
    have_a_guess = 0
    skip_next_optimization = 0
    auto_trim    = 0
    auto_error   = 1
    plot_all     = 0
    subtract     = 0
    show_guess   = 0
    show_error   = 1
    m            = 0
    last         = 0

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
        This actuall performs the optimization on xdata and ydata.
        p0="internal"   is the initial parameter guess, such as [1,2,44.3].
                        "internal" specifies to use the model's internal guess result
                        but you better have guessed!
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

        i = self.pnames.index(name)
        if i < 0:  print "***parameter '"+name+"' does not exist***"
        else:      self.p0[i] = float(value)
        return

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
        return(p[2])

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

    pnames = ["A", "a", "b", "c", "x0","y0"]

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

