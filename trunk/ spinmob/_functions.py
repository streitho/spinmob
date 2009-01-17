#############################################################
# various functions that I like to use

import numpy as _numpy
import matplotlib as _matplotlib
import pylab as _pylab
from scipy.integrate import quad
from scipy.integrate import inf
import cPickle      as _cPickle



import _dialogs                         ;reload(_dialogs)
import _pylab_tweaks                    ;reload(_dialogs)



def array_shift(a, n, fill="average"):
    """
    This will return an array with all the elements shifted forward in index by n.

    a is the array
    n is the amount by which to shift (can be positive or negative)

    fill="average"      fill the new empty elements with the average of the array
    fill="wrap"         fill the new empty elements with the lopped-off elements
    fill=37.2           fill the new empty elements with the value 37.2
    """

    new_a = _numpy.array(a)

    if n==0: return new_a

    fill_array = _numpy.array([])
    fill_array.resize(_numpy.abs(n))

    # fill up the fill array before we do the shift
    if   fill is "average": fill_array = 0.0*fill_array + _numpy.average(a)
    elif fill is "wrap" and n >= 0:
        for i in range(0,n): fill_array[i] = a[i-n]
    elif fill is "wrap" and n < 0:
        for i in range(0,-n): fill_array[i] = a[i]
    else:   fill_array = 0.0*fill_array + fill

    # shift and fill
    if n > 0:
        for i in range(n, len(a)): new_a[i] = a[i-n]
        for i in range(0, n):      new_a[i] = fill_array[i]
    else:
        for i in range(0, len(a)+n): new_a[i] = a[i-n]
        for i in range(0, -n):       new_a[-i-1] = fill_array[-i-1]

    return new_a

def average(a):
    sum = 0.0
    for x in a: sum += x
    return sum/len(a)


def chi_squared(p, f, xdata, ydata):
    return(sum( (ydata - f(p,xdata))**2 ))

def coarsen_array(array, level=1):
    """
    Returns a shorter array of binned data (every level+1 data points).
    """

    if level is 0: return _numpy.array(array)

    new_array = []

    # loop over 0, 2, 4, ...
    for n in range(0, len(array), level+1):
        # reset the count
        x     = 0.0
        count = 0.0

        # loop over this bin
        for m in range(n, n+level+1):
            # make sure we're not past the end of the array
            if m < len(array):
                x += array[m]
                count += 1

        # append the average to the new array
        new_array.append(x / count)

    return _numpy.array(new_array)




def coarsen_data(xdata, ydata, yerror=None, level=1):
    """
    This does averaging of the data, returning coarsened (numpy) [xdata, ydata, yerror]
    Errors are averaged in quadrature.
    """

    new_xdata = []
    new_ydata = []
    new_error = []

    # if level = 1, loop over 0, 2, 4, ...
    for n in range(0, len(xdata), level+1):
        count = 0.0
        sumx  = 0.0
        sumy  = 0.0
        sume2 = 0.0 # sum squared

        # if n==2, loop 2, 3
        for m in range(n, n+level+1):
            if m < len(xdata):
                sumx  += xdata[m]
                sumy  += ydata[m]
                if not yerror==None:
                    sume2 += yerror[m]**2
                count += 1.0

        new_xdata.append(sumx/count)
        new_ydata.append(sumy/count)
        new_error.append(sume2**0.5/count)

    xdata = _numpy.array(new_xdata)
    ydata = _numpy.array(new_ydata)
    if not yerror==None: yerror = _numpy.array(new_error)
    return [xdata,ydata,yerror]




def coarsen_matrix(Z, xlevel=0, ylevel=0):
    """
    This returns a coarsened numpy matrix.
    """
    Z_ycoarsened = _numpy.array(Z)

    # first coarsen the columns (if necessary)
    if ylevel:
        Z_ycoarsened = []
        for c in Z: Z_ycoarsened.append(coarsen_array(c, ylevel))

    # now coarsen the rows
    if xlevel: return coarsen_array(Z_ycoarsened, xlevel)
    else:      return _numpy.array(Z_ycoarsened)

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


def difference(ydata1, ydata2):
    """

    Returns the number you should add to ydata1 to make it line up with ydata2

    """

    y1 = _numpy.array(ydata1)
    y2 = _numpy.array(ydata2)

    return(sum(y2-y1)/len(ydata1))




def distort_matrix_X(Z, X, f, new_xmin, new_xmax, subsample=3):
    """
    Applies a distortion (remapping) to the matrix Z (and x-values X) using function f.
    returns new_Z, new_X

    f is an INVERSE function old_x(new_x)

    Z is a matrix. X is an array where X[n] is the x-value associated with the array Z[n].

    new_xmin, new_xmax is the possible range of the distorted x-variable for generating Z

    points is how many elements the stretched Z should have. "auto" means use the same number of bins
    """

    Z = _numpy.array(Z)
    X = _numpy.array(X)
    points = len(Z)*subsample


    # define a function for searching
    def zero_me(new_x): return f(new_x)-target_old_x

    # do a simple search to find the new_x that gives old_x = min(X)
    target_old_x = min(X)
    new_xmin = find_zero_bisect(zero_me, new_xmin, new_xmax, _numpy.abs(new_xmax-new_xmin)*0.0001)
    target_old_x = max(X)
    new_xmax = find_zero_bisect(zero_me, new_xmin, new_xmax, _numpy.abs(new_xmax-new_xmin)*0.0001)

    # now loop over all the new x values
    new_X = []
    new_Z = []
    bin_width = float(new_xmax-new_xmin)/(points)
    for new_x in frange(new_xmin, new_xmax, bin_width):

        # make sure we're in the range of X
        if f(new_x) <= max(X) and f(new_x) >= min(X):

            # add this guy to the array
            new_X.append(new_x)

            # get the interpolated column
            new_Z.append( interpolate(X,Z,f(new_x)) )

    return _numpy.array(new_Z), _numpy.array(new_X)




def distort_matrix_Y(Z, Y, f, new_ymin, new_ymax, subsample=3):
    """
    Applies a distortion (remapping) to the matrix Z (and y-values Y) using function f.
    returns new_Z, new_Y

    f is a function old_y(new_y)

    Z is a matrix. Y is an array where Y[n] is the y-value associated with the array Z[:,n].

    new_ymin, new_ymax is the range of the distorted x-variable for generating Z

    points is how many elements the stretched Z should have. "auto" means use the same number of bins
    """

    # just use the same methodology as before by transposing, distorting X, then
    # transposing back
    new_Z, new_Y = distort_matrix_X(Z.transpose(), Y, f, new_ymin, new_ymax, subsample)
    return new_Z.transpose(), new_Y





def elements_are_numbers(array, start_index=0, end_index=-1):
    if len(array) == 0: return 0

    if end_index < 0: end_index=len(array)-1
    try:
        for n in range(start_index, end_index+1): float(array[n])
        return 1

    except:
        return 0

def elements_are_strings(array, start_index=0, end_index=-1):
    if len(array) == 0: return 0

    if end_index < 0: end_index=len(array)-1

    for n in range(start_index, end_index+1):
        if not type(array[n]) == str: return 0
    return 1

def frange(start, end, inc=1.0):
    """A range function, that does accept float increments..."""

    start = 1.0*start
    end   = 1.0*end
    inc   = 1.0*inc

    # if we got a dumb increment
    if not inc: return _numpy.array([start,end])

    # if the increment is going the wrong direction
    if 1.0*(end-start)/inc < 0.0:
        inc = -inc

    # get the number of steps
    steps = int(1.0*(end-start)/inc)+1

    L = []
    for n in range(0,steps):
        L.append(start+inc*n)

    return _numpy.array(L)


def index(value, array):
    for n in range(0,len(array)):
        if value == array[n]:
            return(n)
    return(-1)

def index_nearest(value, array):
    """
    expects a _numpy.array
    returns the global minimum of (value-array)^2
    """

    a = (array-value)**2
    return index(a.min(), a)

def index_next_crossing(value, array, starting_index=0, direction=1):
    """
    starts at starting_index, and walks through the array until
    it finds a crossing point with value

    set direction=-1 for down crossing
    """

    for n in range(starting_index, len(array)-1):
        if  (value-array[n]  )*direction >= 0         \
        and (value-array[n+1])*direction <  0: return n

    # no crossing found
    return -1




def insert_ordered(value, array):
    """
    This will insert the value into the array, keeping it sorted, and returning the
    index where it was inserted
    """

    index = 0

    # search for the last array item that value is larger than
    for n in range(0,len(array)):
        if value >= array[n]: index = n+1

    array.insert(index, value)
    return index

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






def interpolate(xarray,yarray,x):
    """

    returns the y value of the linear interpolated function
    y(x)

    """
    if not len(xarray) == len(yarray):
        print "lengths don't match.", len(xarray), len(yarray)
        return None
    if x < min(xarray) or x > max(xarray):
        print "x=" + str(x) + " is not in " + str(min(xarray)) + " to " + str(max(xarray))
        return None

    # find the index of the first value in xarray higher than x
    for n2 in range(1, len(xarray)):
        if x >= min(xarray[n2], xarray[n2-1]) and x <= max(xarray[n2], xarray[n2-1]):
            break
        if n2 == len(xarray):
            print "couldn't find x anywhere."
            return None
    n1 = n2-1

    # now we have the indices surrounding the x value
    # interpolate!

    return yarray[n1] + (x-xarray[n1])*(yarray[n2]-yarray[n1])/(xarray[n2]-xarray[n1])



def invert_increasing_function(f, f0, xmin, xmax, tolerance, max_iterations=100):
    """
    This will try try to qickly find a point on the f(x) curve between xmin and xmax that is
    equal to f0 within tolerance.
    """

    for n in range(max_iterations):
        # start at the middle
        x = 0.5*(xmin+xmax)

        df = f(x)-f0
        if _numpy.fabs(df) < tolerance: return x

        # if we're high, set xmin to x etc...
        if df > 0: xmin=x
        else:      xmax=x

    print "Couldn't find value!"
    return 0.5*(xmin+xmax)

def is_a_number(s):
    try: float(s); return 1
    except:        return 0


def dumbguy_minimize(f, xmin, xmax, xstep):
    """
    This just steps x and looks for a peak

    returns x, f(x)
    """

    prev = f(xmin)
    this = f(xmin+xstep)
    for x in frange(xmin+xstep,xmax,xstep):
        next = f(x+xstep)

        # see if we're on top
        if this < prev and this < next: return x, this

        prev = this
        this = next

    return x, this

def find_zero_bisect(f, xmin, xmax, xprecision):
    """
    This will bisect the range and zero in on zero.
    """
    if f(xmax)*f(xmin) > 0:
        print "find_zero_bisect(): no zero on the range",xmin,"to",xmax
        return None

    temp = min(xmin,xmax)
    xmax = max(xmin,xmax)
    xmin = temp

    xmid = (xmin+xmax)*0.5
    while xmax-xmin > xprecision:
        y = f(xmid)

        # pick the direction with one guy above and one guy below zero
        if y > 0:
            # move left or right?
            if f(xmin) < 0: xmax=xmid
            else:           xmin=xmid

        # f(xmid) is below zero
        elif y < 0:
            # move left or right?
            if f(xmin) > 0: xmax=xmid
            else:           xmin=xmid

        # yeah, right
        else: return xmid

        # bisect again
        xmid = (xmin+xmax)*0.5

    return xmid


def reverse(array):
    """
    returns a reversed numpy array
    """
    l = list(array)
    l.reverse()
    return _numpy.array(l)

def write_to_file(path, string):
    file = open(path, 'w')
    file.write(string)
    file.close()

def append_to_file(path, string):
    file = open(path, 'a')
    file.write(string)
    file.close()

def read_lines(path):
    file = open(path, 'r')
    a = file.readlines()
    file.close()
    return(join(a,'').replace("\r", "\n").split("\n"))

def data_to_file(path, xarray, yarray, delimiter=" ", mode="w"):
    file = open(path, mode)
    for n in range(0, len(xarray)):
        file.write(str(xarray[n]) + delimiter + str(yarray[n]) + '\n')
    file.close()









def data_from_file(path, delimiter=" "):
    lines = read_lines(path)
    x = []
    y = []
    for line in lines:
       s=line.split(delimiter)
       if len(s) > 1:
           x.append(float(s[0]))
           y.append(float(s[1]))
    return([_numpy.array(x), _numpy.array(y)])


def join(array_of_strings, delimiter=' '):
    if array_of_strings == []: return ""

    output = str(array_of_strings[0])
    for n in range(1, len(array_of_strings)):
        output += delimiter + str(array_of_strings[n])
    return(output)




def smooth(array, index, amount):
    """

    Returns the average of the data at index +/- amount

    """

    sum = array[index]
    count = 1.

    for n in range(1, amount+1):
        if index+n >= len(array):
            break
        sum   += array[index+n]
        count += 1.

    for n in range(1, amount+1):
        if index-n < 0:
            break
        sum   += array[index-n]
        count += 1.

    return(sum/count)


def smooth_array(array, amount=1):
    """

    Returns the nearest-neighbor (+/- amount) smoothed array
    This modifies the array!

    It does NOT slice off the funny end points

    """

    # we have to store the old values in a temp array to keep the
    # smoothing from affecting the smoothing
    temp_array = _numpy.array(array)

    for n in range(amount, len(temp_array)-amount):
        array[n] = smooth(temp_array, n, amount)



def sort_matrix(a,n=0):
    """
    This will rearrange a[n] from lowest to highest, and
    rearrange the rest in the same way.

    This modifies the array AND returns it.
    """

    # loop over the length of one column
    for i in range(len(a[n])-1):
        # if this one is higher than the next, switch
        if a[n][i] > a[n][i+1]:
            # loop over all columns
            for j in range(len(a)):
                temp = a[j][i]
                a[j][i] = a[j][i+1]
                a[j][i+1] = temp

            # now call the sorting function again
            sort_matrix(a,n)

    return a

def submatrix(matrix,i1,i2,j1,j2):
    """
    returns the submatrix defined by the index bounds i1-i2 and j1-j2

    Endpoints included!
    """

    new = []
    for i in range(i1,i2+1):
        new.append(matrix[i][j1:j2+1])
    return _numpy.array(new)






def avg(array):
    return(float(sum(array))/float(len(array)))








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






def find_two_peaks(data, remove_background=True):
    """

    Returns two indicies for the two maxima

    """

    y  = _numpy.array( data            )
    x  = _numpy.array( range(0,len(y)) )

    # if we're supposed to, remove the linear background
    if remove_background:
        [slope, offset] = fit_linear(x,y)
        y = y - slope*x
        y = y - min(y)

    # find the global maximum
    max1   = max(y)
    n1     = index(max1, y)

    # now starting at n1, work yourway left and right until you find
    # the left and right until the data drops below a 1/2 the max.
    # the first side to do this gives us the 1/2 width.
    np = n1+1
    nm = n1-1
    yp = max1
    ym = max1
    width = 0
    while 0 < np < len(y) and 0 < nm < len(y):
        yp = y[np]
        ym = y[nm]

        if yp <= 0.5*max1 or ym <= 0.5*max1:
            width = np - n1
            break

        np += 1
        nm -= 1



    # if we didn't find it, we pooped out
    if width == 0:
        return [n1,-1]

    # this means we have a valid 1/2 width.  Find the other max in the
    # remaining data
    n2 = nm
    while 1 < np < len(y)-1 and 1 < nm < len(y)-1:
        if y[np] > y[n2]:
            n2 = np
        if y[nm] > y[n2]:
            n2 = nm
        np += 1
        nm -= 1

    return([n1,n2])





def is_close(x, array, fraction=0.0001):
    """

    compares x to all of the values in array.  If it's fraction close to
    any, returns true

    """

    result = False
    for n in range(0,len(array)):
        if array[n] == 0:
            if x == 0:
                result = True
        elif abs((x-array[n])/array[n]) < fraction:
            result = True

    return(result)






def combine_dictionaries(a, b):
    """
    returns the combined dictionary.  a's values preferentially chosen
    """

    c = {}
    for key in b.keys(): c[key]=b[key]
    for key in a.keys(): c[key]=a[key]
    return c

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
def ubersplit(s, delimiters=['\t','\r',' ']):

    # run through the string, replacing all the delimiters with the first delimiter
    for d in delimiters: s = s.replace(d, delimiters[0])
    return s.split(delimiters[0])

def trim_data(xdata, ydata, yerror, xrange):
    """
    Removes all the data except that between min(xrange) and max(xrange)
    This does not destroy the input arrays.
    """

    if xrange == None: return [_numpy.array(xdata), _numpy.array(ydata), _numpy.array(yerror)]

    xmax = max(xrange)
    xmin = min(xrange)

    x = []
    y = []
    ye= []
    for n in range(0, len(xdata)):
        if xdata[n] >= xmin and xdata[n] <= xmax:
            x.append(xdata[n])
            y.append(ydata[n])
            if not yerror == None: ye.append(yerror[n])

    if yerror == None: ye = None
    else: ye = _numpy.array(ye)
    return [_numpy.array(x), _numpy.array(y), ye]

def find_peaks(array, baseline=0.1, return_subarrays=False):
    """
    This will try to identify the indices of the peaks in array, returning a list of indices in ascending order.

    Runs along the data set until it jumps above baseline. Then it considers all the subsequent data above the baseline
    as part of the peak, and records the maximum of this data as one peak value.
    """

    peaks = []

    if return_subarrays:
        subarray_values  = []
        subarray_indices = []

    # loop over the data
    n = 0
    while n < len(array):
        # see if we're above baseline, then start the "we're in a peak" loop
        if array[n] > baseline:

            # start keeping track of the subarray here
            if return_subarrays:
                subarray_values.append([])
                subarray_indices.append(n)

            # find the max
            ymax=baseline
            nmax = n
            while n < len(array) and array[n] > baseline:
                # add this value to the subarray
                if return_subarrays:
                    subarray_values[-1].append(array[n])

                if array[n] > ymax:
                    ymax = array[n]
                    nmax = n

                n = n+1

            # store the max
            peaks.append(nmax)

        else: n = n+1

    if return_subarrays: return peaks, subarray_values, subarray_indices
    else:                return peaks


def find_N_peaks(array, N=4, max_iterations=100, rec_max_iterations=3, recursion=1):
    """
    This will run the find_peaks algorythm, adjusting the baseline until exactly N peaks are found.
    """

    if recursion<0: return None

    # get an initial guess as to the baseline
    ymin = min(array)
    ymax = max(array)

    for n in range(max_iterations):

        # bisect the range to estimate the baseline
        y1 = (ymin+ymax)/2.0

        # now see how many peaks this finds. p could have 40 for all we know
        p, s, i = find_peaks(array, y1, True)

        # now loop over the subarrays and make sure there aren't two peaks in any of them
        for n in range(len(i)):
            # search the subarray for two peaks, iterating 3 times (75% selectivity)
            p2 = find_N_peaks(s[n], 2, rec_max_iterations, rec_max_iterations=rec_max_iterations, recursion=recursion-1)

            # if we found a double-peak
            if not p2==None:
                # push these non-duplicate values into the master array
                for x in p2:
                    # if this point is not already in p, push it on
                    if not x in p: p.append(x+i[n]) # don't forget the offset, since subarrays start at 0


        # if we nailed it, finish up
        if len(p) == N: return p

        # if we have too many peaks, we need to increase the baseline
        if len(p) > N: ymin = y1

        # too few? decrease the baseline
        else:          ymax = y1

    return None

def printer(arguments="-color", threaded=True):

    global _prefs

    # get the current figure
    f = _pylab.gcf()

    # output the figure to postscript
    postscript_path = _os.environ['PYTHONPATH'] + "\\temp\\graph.ps"
    f.savefig(postscript_path)

    c = _prefs['print_command'] + ' ' + arguments + ' "' + postscript_path + '"'
    print c

    # now run the ps printing command
    if threaded:
        _thread.start_new_thread(_os.system, (c,))
        # bring back the figure and command line
        _pylab_tweaks.get_figure_window(f)
        _pylab_tweaks.get_pyshell()
    else:
        _os.system(c)


def save_object(object, path="ask", text="Save this object where?"):
    if path=="ask": path = _dialogs.Save("*.pickle", text=text)
    if path == "": return

    if len(path.split(".")) <= 1 or not path.split(".")[-1] == "pickle":
        path = path + ".pickle"

    object._path = path

    f = open(path, "w")
    _cPickle.dump(object, f)
    f.close()

def load_object(path="ask", text="Load a pickled object."):
    if path=="ask": path = _dialogs.SingleFile("*.pickle", text=text)
    if path == "": return None

    f = open(path, "r")
    object = _cPickle.load(f)
    f.close()

    object._path = path
    return object

def replace_lines_in_files(search_string, replacement_line):

    # have the user select some files
    paths = _dialog.MultipleFiles('DIS AND DAT|*.*')
    if paths == []: return

    for path in paths:
        shutil.copy(path, path+".backup")
        lines = _fun.read_lines(path)
        for n in range(0,len(lines)):
            if lines[n].find(search_string) >= 0:
                print lines[n]
                lines[n] = replacement_line.strip() + "\n"
        _fun.write_to_file(path, _fun.join(lines, ''))

    return

def search_and_replace_in_files(search, replace, depth=100, paths="ask", confirm=True):

    # have the user select some files
    if paths=="ask":
        paths = _dialog.MultipleFiles('DIS AND DAT|*.*')
    if paths == []: return

    for path in paths:
        lines = _fun.read_lines(path)

        if depth: N=min(len(lines),depth)
        else:     N=len(lines)

        for n in range(0,N):
            if lines[n].find(search) >= 0:
                lines[n] = lines[n].replace(search,replace).strip()
                print path.split('\\')[-1]+ ': "'+lines[n]+'"'
                _wx.Yield()

        # only write if we're not confirming
        if not confirm: _fun.write_to_file(path, _fun.join(lines, '\n'))

    if confirm:
        if raw_input("ja? ")=="yes":
            search_and_replace_in_files(search,replace,depth,paths,False)

    return