import numpy as _numpy

from scipy.integrate import quad
from scipy.integrate import inf


def derivative(xdata, ydata):
    """
    performs d(ydata)/d(xdata) with nearest-neighbor slopes
    must be well-ordered, returns [xdata, D_ydata]
    """
    D_ydata = []
    D_xdata = []
    for n in range(1, len(xdata)-1):
        D_xdata.append(xdata[n])
        D_ydata.append((ydata[n+1]-ydata[n-1])/(xdata[n+1]-xdata[n-1]))

    return [D_xdata, D_ydata]


def integrate(f, x1, x2):
    """
    f(x) = ...
    integrated from x1 to x2
    """

    return quad(f, x1, x2)[0]

def integrate2d(f, x1, x2, y1, y2):
    """
    f(x,y) = ...
    integrated from x1 to x2, y1 to y2
    """
    def fx(y):
        def g(x): return f(x,y)
        return integrate(g, x1, x2)

    return quad(fx, y1, y2)[0]

def integrate3d(f, x1, x2, y1, y2, z1, z2):
    """
    f(x,y,z) = ...
    integrated from x1 to x2, y1 to y2, z1 to z2
    """

    def fxy(z):
        def g(x,y): return f(x,y,z)
        return(integrate2d(g, x1, x2, y1, y2))

    return quad(fxy, z1, z2)[0]

