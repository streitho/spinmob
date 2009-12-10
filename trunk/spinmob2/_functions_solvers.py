import numpy as _numpy





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
