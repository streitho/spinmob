import numpy as _numpy



def chi_squared(p, f, xdata, ydata):
    return(sum( (ydata - f(p,xdata))**2 ))

def fit_linear(xdata, ydata, xrange=None):
    """

    Returns [slope, intercept] of line of best fit, excluding data
    outside the range defined by xrange

    """

    [x,y,e] = trim_data(xdata, ydata, None, xrange)

    ax  = avg(x)
    ay  = avg(y)
    axx = avg(x*x)
    ayy = avg(y*y)
    ayx = avg(y*x)

    slope     = (ayx - ay*ax) / (axx - ax*ax)
    intercept = ay - slope*ax

    return([slope, intercept])


