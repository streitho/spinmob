import numpy as _numpy



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




def avg(array):
    return(float(sum(array))/float(len(array)))


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


def erange(start, end, steps):
    """
    Returns a numpy array over the specified range taking geometric steps.
    """
    if start == 0:
        print "Nothing you multiply zero by gives you anything but zero. Try picking something small."
        return None
    if end == 0:
        print "It takes an infinite number of steps to get to zero. Try a small number?"
        return None

    # figure out our multiplication scale
    x = 1.0*(end/start)**(1.0/(steps-1))

    # now generate the array
    a = []
    for n in range(0,steps): a.append(start*x**n)

    # tidy up the last element (there's often roundoff error)
    a[-1] = end

    return _numpy.array(a)



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


def interpolate(xarray, yarray, x, rigid_limits=True):
    """

    returns the y value of the linear interpolated function
    y(x). Assumes increasing xarray!

    rigid_limits=False means when x is outside xarray's range,
    use the endpoint as the y-value.

    """
    if not len(xarray) == len(yarray):
        print "lengths don't match.", len(xarray), len(yarray)
        return None
    if x < xarray[0] or x > xarray[-1]:
        if rigid_limits:
            print "x=" + str(x) + " is not in " + str(min(xarray)) + " to " + str(max(xarray))
            return None
        else:
            if x < xarray[0]: return yarray[0]
            else:             return yarray[-1]

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


def reverse(array):
    """
    returns a reversed numpy array
    """
    l = list(array)
    l.reverse()
    return _numpy.array(l)



def average_subarray(array, index, amount):
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

def shift(a, n, fill="average"):
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
        array[n] = average_subarray(temp_array, n, amount)



def sort_matrix(a,n=0):
    """
    This will rearrange the array a[n] from lowest to highest, and
    rearrange the rest of a[i]'s in the same way.

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




