#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Tigramite -- Time Series Graph based Measures of Information Transfer
#
# Methods are described in:
#    J. Runge, J. Heitzig, V. Petoukhov, and J. Kurths, Phys. Rev. Lett. 108, 258701 (2012)
#    J. Runge, J. Heitzig, N. Marwan, and J. Kurths, Phys. Rev. E 86, 061121 (2012)
#                                                    AND http://arxiv.org/abs/1210.2748
#    J. Runge, V. Petoukhov, and J. Kurths, Journal of Climate, 27.2 (2014)
#
# Please cite all references when using the method.
#
# Copyright (C) 2012-2014 Jakob Runge <jakobrunge@gmail.com>
# URL: <http://tocsy.pik-potsdam.de/tigramite.php>

"""
Module contains functions for the package tigramite.
"""

#
#  Import essential packages
#
#  Import NumPy for the array object and fast numerics
import numpy

import sys


def lowhighpass_filter(data, cutperiod, pass_periods='low'):
    """
    Butterworth low- or high pass filter. This function applies a linear filter twice, 
    once forward and once backwards. The combined filter has linear phase.

    Assumes data of shape (T, N) or (T,)

    :type data: array
    :param data: data

    :type cutperiod: integer
    :param cutperiod: cutt off period

    :type pass_periods: string
    :param pass_periods: 'low' or 'high'

    :rtype: array
    :returns: filtered data
    """
    try:
        from scipy.signal import butter, filtfilt
    except:
        print 'Could not import scipy.signal for butterworth filtering!'

    fs = 1.
    order = 3
    ws = 1. / cutperiod / (0.5 * fs)
    b, a = butter(order, ws, pass_periods)
    if numpy.rank(data) == 1:
        #        data = lfilter(b, a, data)
        data = filtfilt(b, a, data)
    else:
        for i in range(data.shape[1]):
            #            data[:,i] = lfilter(b, a, data[:,i])
            data[:, i] = filtfilt(b, a, data[:, i])

    return data


def smooth(data, smooth_width, kernel='gaussian',
           data_mask=None, residuals=False):
    """
    Returns either smoothed time series or the difference between the original
    and the smoothed time series (=residuals) of a kernel smoothing with
    gaussian (smoothing kernel width = twice the sigma!) or heaviside window,
    equivalent to a running mean.

    Assumes data of shape (T, N) or (T,)

    :type data: array
    :param data: data

    :type smooth_width: float
    :param smooth_width: smoothing kernel width

    :type kernel: str
    :param kernel: kernel type: 'gaussian' or 'heaviside'

    :type residuals: bool
    :param residuals: True if residuals are desired

    :rtype: array
    :returns: smoothed/residual data
    """

    print("%s %s smoothing with " % ({True: "Take residuals of a ", 
                                   False:""}[residuals], kernel) +
          "window width %.2f (2*sigma for a gaussian!)" % (smooth_width))

    totaltime = len(data)
    if kernel == 'gaussian':
        window = numpy.exp(-(numpy.arange(totaltime).reshape((1, totaltime)) -
                             numpy.arange(totaltime).reshape((totaltime, 1))) 
                            ** 2 / ((2. * smooth_width / 2.) ** 2))
    elif kernel == 'heaviside':
        import scipy.linalg
        wtmp = numpy.zeros(totaltime)
        wtmp[:numpy.ceil(smooth_width / 2.)] = 1
        window = scipy.linalg.toeplitz(wtmp)

    if data_mask is None:
        if numpy.rank(data) == 1:
            smoothed_data = (data * window).sum(axis=1) / window.sum(axis=1)
        else:
            smoothed_data = numpy.zeros(data.shape)
            for i in range(data.shape[1]):
                smoothed_data[:, i] = (
                    data[:, i] * window).sum(axis=1) / window.sum(axis=1)
    else:
        if numpy.rank(data) == 1:
            smoothed_data = ((data * window * data_mask).sum(axis=1) 
                             / (window * data_mask).sum(axis=1))
        else:
            smoothed_data = numpy.zeros(data.shape)
            for i in range(data.shape[1]):
                smoothed_data[:, i] = ((
                    data[:, i] * window * data_mask[:, i]).sum(axis=1) 
                    / (window * data_mask[:, i]).sum(axis=1))

    if residuals:
        return data - smoothed_data
    else:
        return smoothed_data


def weighted_avg_and_std(values, axis, weights):
    """
    Returns the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.

    :type values: array
    :param values: data

    :type axis: int
    :param axis: axis to average/std about

    :rtype: tuple
    :returns: tuple of average and std
    """

    values[numpy.isnan(values)] = 0.
    average = numpy.ma.average(values, axis=axis, weights=weights)
#    print average.shape, weights.shape, values.shape
#    print values-average[:,numpy.newaxis]
    variance = numpy.sum(
        weights * (values - numpy.expand_dims(average, axis)) ** 2, axis=axis) / weights.sum(axis=axis)

    return (average, numpy.sqrt(variance))


def time_bin_with_mask(data, time_bin_length, data_mask=None):
    """
    Returns time binned data where only about non-masked values is averaged.

    Assumes data of shape (T, N) or (T,)

    :type data: array
    :param data: data

    :type data_mask: array
    :param data_mask: mask indicator

    :type time_bin_length: int
    :param time_bin_length: length of time bin

    :rtype: tuple
    :returns: binned data, new length, grid_size
    """

#    print fulldata.shape, fulldata_mask.shape, time_bin_length
    n_time = len(data)

    if data_mask is None:
        data_mask = numpy.ones(data.shape)

    if numpy.rank(data) == 1.:
        data.shape = (n_time, 1)
        data_mask.shape = (n_time, 1)
#    print 'time binning...'
    bindata = numpy.zeros(
        (n_time / time_bin_length,) + data.shape[1:], dtype="float32")
    index = 0
    for i in xrange(0, n_time - time_bin_length + 1, time_bin_length):
        # print weighted_avg_and_std(fulldata[i:i+time_bin_length], axis=0,
        # weights=fulldata_mask[i:i+time_bin_length])[0]
        bindata[index] = weighted_avg_and_std(
            data[i:i + time_bin_length], axis=0, weights=data_mask[i:i + time_bin_length])[0]
        index += 1

    n_time, grid_size = bindata.shape

    return (bindata.squeeze(), n_time)


def ordinal_patt_array(array, array_mask, dim=2, step=1, verbosity=0):
    """
    Returns symbolified array of ordinal patterns with dimension dim and stepsize step.

    Each data vector (X_t, ..., X_t+(dim-1)*step) is converted to its rank vector.
    E.g., (0.2, -.6, 1.2) --> (1,0,2) which is then assigned to a unique integer (see Article)... there are faculty(dim) possible rank vectors.

    The symb_array is step*(dim-1) shorter than the original array!

    Reference: B. Pompe and J. Runge (2011). Momentary information transfer as a coupling measure of time series. Phys. Rev. E, 83(5), 1–12. doi:10.1103/PhysRevE.83.051122

    :type array: array
    :param array: data

    :type array: array_mask
    :param array: data mask

    :type dim: int
    :param dim: pattern dimension

    :type step: int
    :param step: delay of pattern embedding vector

    :rtype: tuple
    :returns: converted data, new length
    """

    import scipy
    from scipy.misc import factorial

    assert dim > 1

    patt_time = int(array.shape[0] - step * (dim - 1))
    assert patt_time > 0
    n_time, nodes = array.shape

    patt = numpy.zeros((patt_time, nodes), dtype='int32')
    patt_mask = numpy.zeros((patt_time, nodes), dtype='int32')
    fac = factorial(numpy.arange(10)).astype('int32')

    code = r"""
        int n,t,k,i,j,p,tau,start,mask;
        double v[dim];

        start = step*(dim-1);
        for(n = 0; n < nodes; n++){
            for(t = start; t < n_time; t++){
                mask = 1;
                for(k = 0; k < dim; k++){
                    tau = k*step;
                    v[k] = array(t - tau, n);
                    mask *= array_mask(t - tau, n);
                }
                if( v[0] < v[1]){
                    p = 1;
                }
                else{
                    p = 0;
                }
                for (i = 2; i < dim; i++){
                    for (j = 0; j < i; j++){
                        if( v[j] < v[i]){
                            p += fac(i);
                        }
                    }
                }
                patt(t-start, n) = p;
                patt_mask(t-start, n) = mask;
            }
        }
    """
    vars = ['array', 'array_mask', 'patt', 'patt_mask',
            'dim', 'step', 'fac', 'nodes', 'n_time']

    scipy.weave.inline(code, vars,
                       type_converters=scipy.weave.converters.blitz,
                       extra_compile_args=["-O3"])

    return (patt, patt_mask, patt_time)


def quantile_bin_array(data, bins=6):
    """
    Returns symbolified array with equal-quantile binning.

    :type data: array
    :param data: data

    :type bins: int
    :param bins: number of bins

    :rtype: data
    :returns: converted data
    """

    T, N = data.shape

    # get the bin quantile steps
    bin_edge = numpy.ceil(T / float(bins))

    symb_array = numpy.zeros((T, N), dtype='int32')

    # get the lower edges of the bins for every time series
    edges = numpy.sort(data, axis=0)[::bin_edge, :].T
    bins = edges.shape[1]

    # This gives the symbolic time series
    symb_array = (data.reshape(T, N, 1) >= edges.reshape(1, N, bins)).sum(
        axis=2) - 1

    return symb_array.astype('int32')


def get_sig_thres(sig_lev, df):
    """
    Returns the significance threshold of the pearson correlation coefficient according to 
    a Student's t-distribution with df degrees of freedom.

    One- or two-tailedness should be accounted for by the choice of sig_lev, i.e., 95% two-tailed
    significance corresponds to sig_lev = 0.975


    :type sig_lev: float
    :param sig_lev: significance level

    :type df: int
    :param df: degrees of freedom

    :rtype: float
    :returns: significance threshold
    """
    try:
        import scipy.stats
    except:
        print "Couldn't import scipy.stats... no significance threshold calculated!"

    return scipy.stats.t.ppf(sig_lev, df) / numpy.sqrt(df + scipy.stats.t.ppf(sig_lev, df) ** 2)


def var_network(graph=numpy.array([[[0., 0.2], [0., 0.5]],
                                   [[0., 0.], [0., 0.3]]]),
                inv_inno_cov=None, inno_cov=None, use='inno_cov',
                T=100):
    """
    Static method to generate a realization of a vector-autoregressive
    process with possibly correlated innovations.

    Useful for testing.

    Example:
    graph=numpy.array([[[0.2,0.,0.],[0.5,0.,0.]],
                       [[0.,0.1,0. ],[0.3,0.,0.]]])

    represents a process

    X_1(t) = 0.2 X_1(t-1) + 0.5 X_2(t-1) + eps_1(t)
    X_2(t) = 0.3 X_2(t-1) + 0.1 X_1(t-2) + eps_2(t)

    with inv_inno_cov being the negative (except for diagonal) inverse
    covariance matrix of (eps_1(t), eps_2(t)) OR inno_cov being
    the covariance.


    :type graph: array
    :param graph: lagged connectivity matrices.

    :type inv_inno_cov: array or None
    :param inv_inno_cov: inverse covariance matrix of innovations

    :type inno_cov: array or None
    :param inno_cov:  covariance matrix of innovations

    :type use: string
    :param use: specifier

    :type T: integer
    :param T: sample size

    :rtype: array
    :returns: realization of VAR process
    """

    N, N, P = graph.shape

    # Test stability
    stabmat = numpy.zeros((N * P, N * P))
    index = 0
    for i in range(0, N * P, N):
        stabmat[:N, i:i + N] = graph[:, :, index]
        if index < P - 1:
            stabmat[i + N:i + 2 * N, i:i + N] = numpy.identity(N)

        index += 1

    eig = numpy.linalg.eig(stabmat)[0]
    assert numpy.all(numpy.abs(eig) < 1.), "Nonstationary process!"

    X = numpy.random.randn(N, T)

    if use == 'inv_inno_cov' and inv_inno_cov is not None:
        mult = -numpy.ones((N, N))
        mult[numpy.diag_indices_from(mult)] = 1
        inv_inno_cov *= mult
        noise = numpy.random.multivariate_normal(
            mean=numpy.zeros(N),
            cov=numpy.linalg.inv(inv_inno_cov),
            size=T)
    elif use == 'inno_cov' and inno_cov is not None:
        noise = numpy.random.multivariate_normal(
            mean=numpy.zeros(N), cov=inno_cov, size=T)
    else:
        noise = numpy.random.randn(T, N)

    for t in xrange(P, T):
        Xpast = numpy.repeat(
            X[:, t - P:t][:, ::-1].reshape(1, N, P), N, axis=0)
        X[:, t] = (Xpast * graph).sum(axis=2).sum(axis=1) + noise[t]

    return X.transpose()

class Logger(object):
    """Class to append print output to a string which can be saved"""
    def __init__(self):
        self.terminal = sys.stdout
        self.log = "" #open("log.dat", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log += message  #  .write(message)  