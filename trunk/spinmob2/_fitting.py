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


def decompose_covariance(c):
    """
    This decomposes a covariance matrix into an error vector and a correlation matrix
    """

    # make it a kickass copy of the original
    c = _numpy.array(c)

    # first get the error vector
    e = []
    for n in range(0, len(c[0])): e.append(_numpy.sqrt(c[n][n]))

    # now cycle through the matrix, dividing by e[1]*e[2]
    for n in range(0, len(c[0])):
        for m in range(0, len(c[0])):
            c[n][m] = c[n][m] / (e[n]*e[m])

    return [_numpy.array(e), _numpy.array(c)]

def assemble_covariance(error, correlation):
    """
    This takes an error vector and a correlation matrix and assembles the covariance
    """

    covariance = []
    for n in range(0, len(error)):
        covariance.append([])
        for m in range(0, len(error)):
            covariance[n].append(correlation[n][m]*error[n]*error[m])
    return _numpy.array(covariance)

