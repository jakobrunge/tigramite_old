#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Tigramite -- Time Series Graph based Measures of Information Transfer
#
# Methods are described in:
#    J. Runge, J. Heitzig, V. Petoukhov, and J. Kurths,
#    Phys. Rev. Lett. 108, 258701 (2012)
#
#    J. Runge, J. Heitzig, N. Marwan, and J. Kurths,
#    Phys. Rev. E 86, 061121 (2012)
#    AND http://arxiv.org/abs/1210.2748
#
#    J. Runge, V. Petoukhov, and J. Kurths, Journal of Climate, 27.2 (2014)
#
# Please cite all references when using the method.
#
# Copyright (C) 2012-2015 Jakob Runge <jakobrunge@gmail.com>

"""
Module contains functions for the package tigramite.
"""

#
#  Import essential packages
#


#  Import NumPy for the array object and fast numerics
import numpy

#  Import scipy.weave for C++ inline code and other packages
from scipy import weave, linalg,  special, stats

#  Additional imports
import itertools
import warnings

__docformat__ = "epytext en"
"""The epydoc documentation format for this file."""


def _sanity_checks(which='pc_algo',
                   data=None, estimate_parents_neighbors='both',
                   tau_min=0, tau_max=5,
                   initial_conds=2, max_conds=6, max_trials=5,
                   measure='par_corr', measure_params=None,
                   significance='analytic', sig_lev=0.975, sig_samples=100,
                   fixed_thres=0.015,
                   mask=False, mask_type=None, data_mask=None,
                   initial_parents_neighbors=None,
                   selected_variables=None,
                   parents_neighbors=None,
                   cond_mode='none',
                   solid_contemp_links=True,
                   confidence=False, conf_lev=0.95, conf_samples=100,
                   verbosity=0):
    """Sanity checks of data and params."""

    T, N = data.shape

    # Checks
    if type(data) != numpy.ndarray:
        raise TypeError("data is of type %s, " % type(data) +
                        "must be numpy.ndarray")
    if N > T:
        warnings.warn("data.shape = %s," % str(data.shape) +
                      " is it of shape (observations, variables) ?")
    if numpy.isnan(data).sum() != 0:
        raise ValueError("NaNs in the data")

    if significance not in ['analytic', 'shuffle', 'fixed']:
        raise ValueError("significance must be one of "
                         "'analytic', 'shuffle', 'fixed'")
    if significance == 'analytic' and (sig_lev < .5 or sig_lev >= 1.):
        raise ValueError("sig_lev = %.2f, " % sig_lev +
                         "but must be between 0.5 and 1")
    if significance == 'shuffle' and sig_samples*(1.-sig_lev) < 1.:
        raise ValueError("sig_samples*(1.-sig_lev) is %.2f"
                         % (sig_samples*(1.-sig_lev)) + ", must be >> 1")
    if significance == 'fixed' and fixed_thres <= 0.:
        raise ValueError("fixed_thres = %.2f, must be > 0" % fixed_thres)

    if mask:
        if type(data_mask) != numpy.ndarray:
            raise TypeError("data_mask is of type %s, " % type(data_mask) +
                            "must be numpy.ndarray")
        if numpy.isnan(data_mask).sum() != 0:
            raise ValueError("NaNs in the data_mask")
        if mask_type is None or len(set(mask_type) - set(['x', 'y', 'z'])) > 0:
            raise ValueError("mask_type = %s, but must be list containing"
                             % mask_type + " 'x','y','z', or any combination")
        if data.shape != data_mask.shape:
            raise ValueError("shape mismatch: data.shape = %s"
                             % str(data.shape) +
                             " but data_mask.shape = %s, must be the same"
                             % str(data_mask.shape))

    if measure not in ['par_corr', 'reg', 'cmi_knn', 'cmi_symb',
                       'cmi_gauss']:
        raise ValueError("measure must be one of "
                         "'par_corr', 'reg', 'cmi_knn', "
                         "'cmi_symb', 'cmi_gauss'")
    if ((measure == 'cmi_knn')
       and (measure_params['knn'] > T/2.
            or measure_params['knn'] < 1)):
        raise ValueError("knn = %s , " % str(measure_params['knn']) +
                         "should be between 1 and T/2")
    if measure == 'cmi_symb':
        if 'int' not in str(data.dtype):
            raise TypeError("data needs to be of integer type "
                            "for symbolic estimation")
    if measure in ['cmi_knn', 'cmi_symb'] and T < 500:
        warnings.warn("T = %s ," % str(T) +
                      " unreliable estimation using %s estimator" % measure)

    # Checks only for pc_algo
    if which == 'pc_algo':
        if (measure == 'cmi_knn'):
            if (measure_params is None
               or type(measure_params) != dict
               or 'knn' not in measure_params.keys()):
                raise ValueError("measure_params must be dictionary "
                                 "containing key 'knn'. "
                                 "Recommended value 'knn':10.")
        if tau_min > tau_max or min(tau_min, tau_max) < 0:
            raise ValueError("tau_max = %d, tau_min = %d, " % (
                             tau_max, tau_min)
                             + "but 0 <= tau_min <= tau_max")
        if (estimate_parents_neighbors == 'parents'
                or estimate_parents_neighbors == 'both') and tau_max < 1:
            raise ValueError("tau_max = %d, tau_min = %d, " % (
                             tau_max, tau_min)
                             + "but tau_max >= tau_min > 0 to "
                             "estimate parents")
        if initial_conds > max_conds:
            raise ValueError("initial_conds must be <= max_conds")
        if max_trials <= 0:
            raise ValueError("max_trials must be > 0")
        if (initial_parents_neighbors is not None
                and _check_parents_neighbors(
                    initial_parents_neighbors, N) is False):
            raise ValueError("initial_parents_neighbors must provide "
                             "parents/neighbors for all variables j in format:"
                             " {..., j:[(var1, lag1), (var2, lag2), ...], "
                             " ...}, "
                             "where vars must be within 0..N-1 and lags <= 0")
        if estimate_parents_neighbors not in ['neighbors', 'parents', 'both']:
            raise ValueError("estimate_parents_neighbors must be one of "
                             "'neighbors', 'parents', 'both'")

    # Checks only for lag functions
    elif which == 'lagfuncs':
        if 'cmi' in measure:
            # if (measure_params is None
            #    or type(measure_params) != dict
            #    or 'rescale_cmi' not in measure_params.keys()):
            #     raise ValueError("measure_params must be dictionary "
            #                      "containing key 'rescale_cmi'. "
            #                      "'rescale_cmi':True rescales the CMI to the "
            #                      "partial correlation scale. ")
            if (measure == 'cmi_knn'):
                if (measure_params is None
                    or type(measure_params) != dict
                    or 'knn' not in measure_params.keys()):
                    raise ValueError("measure_params must contain key 'knn', "
                                     "recommended value 'knn':10.")
        if tau_max < 0:
            raise ValueError("tau_max = %d, " % (tau_max)
                             + "but 0 <= tau_max")
        if confidence not in ['analytic', 'bootstrap', False]:
            raise ValueError("confidence must be one of "
                             "'analytic', 'bootstrap', False")
        if confidence:
            if (conf_lev < .5 or conf_lev >= 1.):
                raise ValueError("conf_lev = %.2f, " % conf_lev +
                                 "but must be between 0.5 and 1")
            if confidence == 'bootstrap' and conf_samples*(1. - conf_lev) / 2. < 1.:
                raise ValueError("conf_samples*(1.-conf_lev)/2 is %.2f"
                                 % (conf_samples*(1.-conf_lev)/2.) +
                                 ", must be >> 1")
        if (parents_neighbors is not None
                and _check_parents_neighbors(
                    parents_neighbors, N) is False):
            raise ValueError("parents_neighbors must provide parents/neighbors"
                             " for all variables j in format: {..., "
                             "j:[(var1, lag1), (var2, lag2), ...], ...}, "
                             "where vars must be in [0..N-1] and lags <= 0")
        if cond_mode not in ['none', 'parents_x', 'parents_y', 'parents_xy']:
            raise ValueError("cond_mode must be one of "
                             "'none', 'parents_x', 'parents_y', 'parents_xy'")
        if selected_variables is not None:
            if (numpy.any(numpy.array(selected_variables) < 0) or
               numpy.any(numpy.array(selected_variables) >= N)):
                raise ValueError("selected_variables must be within 0..N-1")


def pc_algo_all(data, estimate_parents_neighbors='both',
                tau_min=0, tau_max=5,
                initial_conds=2, max_conds=6, max_trials=5,
                measure='par_corr', measure_params=None,
                significance='analytic', sig_lev=0.975, sig_samples=100,
                fixed_thres=0.015,
                mask=False, mask_type=None, data_mask=None,
                initial_parents_neighbors=None,
                verbosity=0):
    """
    Function to estimate parents of all processes of multivariate dataset.


    :rtype: dictionary
    :returns: results
    """

    T, N = data.shape

    _sanity_checks(which='pc_algo',
                   data=data,
                   estimate_parents_neighbors=estimate_parents_neighbors,
                   tau_min=tau_min, tau_max=tau_max,
                   initial_conds=initial_conds, max_conds=max_conds,
                   max_trials=max_trials,
                   measure=measure, measure_params=measure_params,
                   significance=significance, sig_lev=sig_lev,
                   sig_samples=sig_samples,
                   fixed_thres=fixed_thres,
                   mask=mask, mask_type=mask_type, data_mask=data_mask,
                   initial_parents_neighbors=initial_parents_neighbors,
                   verbosity=verbosity)

    if verbosity > 0:
        print("\n" + "-"*60 +
              "\nEstimating parents for all variables:"
              "\n" + "-"*60)

    parents_neighbors = dict([(i, []) for i in range(N)])

    if (estimate_parents_neighbors == 'parents'
            or estimate_parents_neighbors == 'both'):
        for j in range(N):
            res = _pc_algo(data, j,
                           parents_or_neighbors='parents',
                           initial_parents_neighbors=initial_parents_neighbors,
                           all_parents=None,
                           tau_min=tau_min,
                           tau_max=tau_max, initial_conds=initial_conds,
                           max_conds=max_conds,
                           max_trials=max_trials,
                           measure=measure,
                           significance=significance, sig_lev=sig_lev,
                           sig_samples=sig_samples,
                           fixed_thres=fixed_thres,
                           measure_params=measure_params,
                           mask=mask, mask_type=mask_type, data_mask=data_mask,
                           verbosity=verbosity)
            parents_neighbors[j] = res

    if (estimate_parents_neighbors == 'neighbors'
            or estimate_parents_neighbors == 'both'):
        if verbosity > 0:
            print("\n" + "-"*60 +
                  "\nEstimating neighbors for all variables:"
                  "\n" + "-"*60)
        for j in range(N):
            res = _pc_algo(data, j,
                           parents_or_neighbors='neighbors',
                           initial_parents_neighbors=initial_parents_neighbors,
                           all_parents=parents_neighbors,
                           tau_min=tau_min,
                           tau_max=tau_max, initial_conds=initial_conds,
                           max_conds=max_conds, max_trials=max_trials,
                           measure=measure,
                           significance=significance, sig_lev=sig_lev,
                           sig_samples=sig_samples, fixed_thres=fixed_thres,
                           measure_params=measure_params,
                           mask=mask, mask_type=mask_type, data_mask=data_mask,
                           verbosity=verbosity)
            parents_neighbors[j] += res

    if verbosity > 0:
        print("\n" + "-"*60)
        if estimate_parents_neighbors == 'both':
            print("\nResulting sorted parents and neighbors:")
        else:
            print("\nResulting sorted %s:" % estimate_parents_neighbors)
        for j in range(N):
            print("\n    Variable %d: %s" % (j, parents_neighbors[j]))
        print("\n" + "-"*60)

    return parents_neighbors


def _check_parents_neighbors(graph, N):
    """Checks that graph contains keys for variables and properness of links"""

    if not isinstance(graph, dict):
        return False
    if (set(graph.keys()) != set(range(N))):
        return False
    for i in graph.keys():
        if not isinstance(graph[i], list):
            return False
        for tup in graph[i]:
            if not isinstance(tup, tuple):
                return False
        if len(graph[i]) > 0:
            w = numpy.array(graph[i])
            # Parents/neighbors indices must be within [0, N-1]
            if not numpy.any((0 <= w[:, 0]) & (w[:, 0] < N)):
                return False
            # Lags must be smaller equal zero
            if not numpy.any(w[:, 1] <= 0):
                return False

    return True


class Conditions():

    def __init__(self, parents):
        self.parents = parents
        self.checked_conds = []

    def next_condition(self, dim, check_only=False):
        # print 'len(self.parents),  self.dim ', len(self.parents),  dim
        # if len(self.parents) > self.dim:
        for comb in itertools.combinations(self.parents, dim):
            # print 'comb ', list(comb), self.checked_conds
            if set(list(comb)) not in self.checked_conds:
                if check_only is False:
                    self.checked_conds.append(set(list(comb)))
                    return list(comb)
                else:
                    return True

        return False

    def update(self, parents):
        self.parents = parents

    def check_convergence(self, dim):
        if self.next_condition(dim, check_only=True) is False:
            return True
        else:
            return False


def _pc_algo(data, j,
             parents_or_neighbors='parents',
             initial_parents_neighbors=None, all_parents=None,
             tau_min=0, tau_max=5,
             initial_conds=1, max_conds=6, max_trials=5,
             measure='par_corr',
             significance='fixed', sig_lev=0.975, sig_samples=100,
             fixed_thres=0.015,
             measure_params=None,
             mask=False, mask_type=None, data_mask=None,
             verbosity=0):

    T, N = data.shape

    # For regressions the PC algorithm is run with par_corr,
    # the independence test is the same
    if measure == 'reg':
        measure = 'par_corr'

    if (measure == 'cmi_knn'):
        if (measure_params is None
           or type(measure_params) != dict
           or 'knn' not in measure_params.keys()):
            raise ValueError("measure_params must be dictionary "
                             "containing key 'knn'. "
                             "Recommended value 'knn':10.")

    if parents_or_neighbors == 'neighbors':
        tau_min = tau_max = 0
    elif parents_or_neighbors == 'parents':
        tau_min = max(1, tau_min)
        if tau_max < tau_min:
            raise ValueError("tau_max = %d, tau_min = %d," % (tau_max, tau_min)
                             + " but tau_max >= tau_min > 0 to estimate "
                             "parents")
    else:
        raise ValueError("parents_or_neighbors must be either 'parents' or"
                         " 'neighbors'")

    if initial_parents_neighbors is None:
        nodes_and_estimates = dict([((var, -lag), numpy.infty)
                                    for var in range(N)
                                    for lag in range(tau_min, tau_max + 1)])
    else:
        nodes_and_estimates = dict(
                    [((var, lag), numpy.infty)
                        for (var, lag) in initial_parents_neighbors[j]
                        if -lag in range(tau_min, tau_max + 1)])

    if parents_or_neighbors == 'neighbors':
        if (j, 0) in nodes_and_estimates.keys():
            del nodes_and_estimates[(j, 0)]
        if all_parents is None:
            raise ValueError("all_parents must be specified, "
                             "at least empty lists")
        else:
            if set(all_parents.keys()) != set(range(N)):
                raise ValueError("all_parents must provide lists for all "
                                 "variables")

    # For the first iteration the nodes order is arbitrary
    sorted_nodes = nodes_and_estimates.keys()
    conditions = Conditions(sorted_nodes)

    if verbosity > 1:
        print("\n    " + "-"*60 +
              "\n    Estimating %s for variable %d :" % (parents_or_neighbors, j)
              + "\n    " + "-"*60)
        if initial_parents_neighbors is not None:
            print("    ckecking only nodes %s" % str(nodes_and_estimates.keys()))

    conds_dim = 0
    while conds_dim < len(sorted_nodes) and conds_dim <= max_conds:

        if verbosity > 1:
            if parents_or_neighbors == 'neighbors':
                print("\n    Number of conditions (except for parents)"
                      " = %d:" % conds_dim)
            else:
                print("\n    Number of conditions = %d:" % conds_dim)

        # print 'strongest ', sorted_nodes

        # Reset values
        # nodes_and_estimates = dict.fromkeys(nodes_and_estimates, numpy.infty)

        comb_index = 0
        while comb_index < max_trials:

            # Pick strongest parent(s) first, in lexicographic order
            conds_y = conditions.next_condition(conds_dim)

            # print conds_y
            if conds_y is False:
                if verbosity > 1:
                    print("\n        No combinations for %d " % conds_dim +
                          "conditions left...")
                break

            # Add parents of all contemporaneous conditions
            if parents_or_neighbors == 'neighbors':
                cond_neighbors = conds_y[:]
                for node in cond_neighbors:
                    conds_y += _get_parents(all_parents[node[0]][:max_conds])
                # Also add parents of Y
                conds_y += all_parents[j][:max_conds]

            if verbosity > 1:
                print("\n        combination index = %d:" % comb_index)

            # Iterate through nodes (except those in conditions)
            for (i, tau) in [node for node in sorted_nodes
                             if node not in conds_y]:

                lag = [-tau]

                if verbosity > 1:
                    print("\n            link (%d, %d) --> %d:" % (i, tau, j) +
                          "\n            with conds_y = %s" % (str(conds_y)))

                # Calculate lag function
                lag_func = _calculate_lag_function(
                    measure=measure, data=data,
                    data_mask=data_mask,
                    mask=mask, mask_type=mask_type,
                    var_x=i, var_y=j, selected_lags=lag,
                    conds_x=[], conds_y=conds_y,
                    tau_max=tau_max,
                    significance=significance, sig_lev=sig_lev,
                    sig_samples=sig_samples, fixed_thres=fixed_thres,
                    measure_params=measure_params,
                    verbosity=verbosity)

                # print lag_func
                cmi = lag_func['cmi'][lag]
                sig_thres = lag_func['cmi_sig'][lag]

                if numpy.abs(cmi) < sig_thres:
                    # Remove node
                    del nodes_and_estimates[(i, tau)]
                    sorted_nodes.remove((i, tau))
                else:
                    # Take minimum estimate among all previous CMIs
                    nodes_and_estimates[(i, tau)] = min(
                        numpy.abs(cmi),
                        nodes_and_estimates[(i, tau)])

            # Sort parents, strongest are checked as conditions first
            sorted_nodes = sorted(nodes_and_estimates,
                                  key=nodes_and_estimates.get, reverse=True)
            conditions.update(sorted_nodes)

            if verbosity > 1:
                if len(sorted_nodes) < 10:
                    print("\n        --> %d nodes for variable %d left:"
                          % (len(sorted_nodes), j))
                    print("             %s     " % (
                        str(sorted_nodes)))
                else:
                    print("\n        --> %d nodes for variable %d left"
                          % (len(sorted_nodes), j))

            comb_index += 1

        # Sort parents, strongest are checked as conditions first
        sorted_nodes = sorted(nodes_and_estimates,
                              key=nodes_and_estimates.get, reverse=True)
        conditions.update(sorted_nodes)

        if verbosity > 1:
            if len(sorted_nodes) < 10:
                print("\n    --> sorted nodes for variable %d:"
                      % (j))
                print("         %s     " % (
                    str(sorted_nodes)))
            else:
                print("\n     --> %d sorted nodes for variable %d left"
                      % (len(sorted_nodes), j))

        if conds_dim == 0:
            conds_dim = initial_conds
        else:
            conds_dim += 1

        if conds_dim >= len(sorted_nodes) and len(sorted_nodes) > 1:
            if verbosity > 1:
                print("\n    Check whether also condition sets smaller than "
                      "remaining parents have been used...")
            convergence = conditions.check_convergence(dim=len(sorted_nodes)-1)
            if convergence:
                break
            else:
                if verbosity > 1:
                    print("\n    Yes --> returning to conds_dim = len(nodes)-1")
                conds_dim = len(sorted_nodes)-1

    if verbosity > 1:
        print("\n    Algorithm converged.")

    if len(sorted_nodes) > 1:

        if verbosity > 1:
            print("\n    Now sorting nodes by ITY...")

        for (i, tau) in sorted_nodes:
            # print (i, tau), nodes_and_estimates[(i, tau)],
            conds_y = sorted_nodes[:]
            # Add parents of all contemporaneous conditions
            if parents_or_neighbors == 'neighbors':
                cond_neighbors = conds_y[:]
                for node in cond_neighbors:
                    conds_y += _get_parents(all_parents[node[0]][:max_conds])
                # Also add parents of Y
                conds_y += all_parents[j][:max_conds]

            nodes_and_estimates[(i, tau)] = abs(_calculate_lag_function(
                measure=measure, data=data,
                data_mask=data_mask,
                mask=mask, mask_type=mask_type,
                var_x=i, var_y=j, selected_lags=[-tau],
                conds_x=[], conds_y=conds_y,
                tau_max=tau_max,
                significance=False,
                measure_params=measure_params,
                verbosity=0)['cmi'][[-tau]])
            # print nodes_and_estimates[(i, tau)]

        sorted_nodes = sorted(nodes_and_estimates,
                              key=nodes_and_estimates.get, reverse=True)

        if verbosity > 1:
            if len(sorted_nodes) < 10:
                print("\n    --> sorted nodes for variable %d:"
                      % (j))
                print("         %s     " % (
                    str(sorted_nodes)))
            else:
                print("\n     --> %d sorted nodes for variable %d left"
                      % (len(sorted_nodes), j))

    return sorted_nodes


def _get_conditions(parents_neighbors, i, j,
                    cond_mode='none',
                    contemp=False,
                    max_conds=4):
    """
    get conditions for cond_mode = 'none' (MI), 'parents_xy' (MIT),
    'parents_y' (ITY), 'parents_x' (ITX) ...

    NOTE: only up to max_conds conditions are used.
    This is the number of conditions used for both X and Y.
    Therefore, the total number of conditions for directed links
    can be 2*max_conds.
    For solid_contemp_links=True (in main script) additionally for the
    number of neighbors as well as the number of parents per neighbor
    leading to 2*max_conds + 2*max_conds*max_conds.

    :type cond_mode: str
    :param cond_mode: selected conditions type

    :type i: int
    :param i: process index (driver)
    """

    # Parents of X (node i)
    if cond_mode == 'parents_y' or cond_mode == 'none':
        conds_x = []
    elif (cond_mode == 'parents_xy' or cond_mode == 'parents_x'):
        if contemp:
            conds_x = _get_parents(parents_neighbors[i])[:max_conds]
            neighbors = _get_neighbors(parents_neighbors[i],
                                       exclude=(j, 0))[:max_conds]
            conds_x += neighbors
            for node in neighbors:
                conds_x += _get_parents(parents_neighbors[node[0]])[:max_conds]
        else:
            conds_x = _get_parents(parents_neighbors[i])[:max_conds]

    # Parents of Y (node j)
    if cond_mode == 'parents_x' or cond_mode == 'none':
        conds_y = []
    elif (cond_mode == 'parents_xy' or cond_mode == 'parents_y'):
        if contemp:
            conds_y = _get_parents(parents_neighbors[j])[:max_conds]
            neighbors = _get_neighbors(parents_neighbors[j],
                                       exclude=(i, 0))[:max_conds]
            conds_y += neighbors
            for node in neighbors:
                conds_y += _get_parents(parents_neighbors[node[0]])[:max_conds]
        else:
            conds_y = _get_parents(parents_neighbors[j])[:max_conds]

    return conds_x, conds_y


def get_lagfunctions(data, selected_variables=None, parents_neighbors=None,
                     cond_mode='none',
                     solid_contemp_links=True,
                     tau_max=1, max_conds=4,
                     measure='par_corr',
                     measure_params=None,
                     significance='analytic', sig_lev=0.95, sig_samples=100,
                     fixed_thres=0.015,
                     confidence=False, conf_lev=0.95, conf_samples=100,
                     mask=False, mask_type=None, data_mask=None,
                     verbosity=0
                     ):
    """Function that estimates the MI, MIT, ITY, ... for all pairs (i, j).

    :rtype: dictionary
    :returns: results
    """

    T, N = data.shape

    _sanity_checks(which='lagfuncs',
                   data=data,
                   tau_max=tau_max,
                   measure=measure, measure_params=measure_params,
                   significance=significance, sig_lev=sig_lev,
                   sig_samples=sig_samples,
                   fixed_thres=fixed_thres,
                   mask=mask, mask_type=mask_type, data_mask=data_mask,
                   selected_variables=selected_variables,
                   parents_neighbors=parents_neighbors,
                   cond_mode=cond_mode,
                   solid_contemp_links=solid_contemp_links,
                   confidence=confidence, conf_lev=conf_lev,
                   conf_samples=conf_samples,
                   verbosity=verbosity)

    if selected_variables is None:
        selected_variables = range(N)

    def calc(j, contemp_only=False):

        if contemp_only:
            tau_max_here = 0
        else:
            tau_max_here = tau_max

        lagfunc_slice = numpy.zeros((N, tau_max + 1))
        sig_thres_slice = numpy.zeros((N, tau_max + 1))
        conf_slice = numpy.zeros((N, tau_max + 1, 2))

        for i in range(N):

            conds_x, conds_y = _get_conditions(
                parents_neighbors=parents_neighbors,
                i=i,
                j=j,
                contemp=contemp_only,
                max_conds=max_conds,
                cond_mode=cond_mode)

            if verbosity > 1:
                print("\n        lag function %d --> %d:" % (i, j) +
                      "\n        with conds_x = %s" % str(conds_x) +
                      "\n        with conds_y = %s" % str(conds_y))

            # Calculate lag function
            lag_func = _calculate_lag_function(
                measure=measure,
                data=data,
                data_mask=data_mask,
                mask=mask, mask_type=mask_type,
                var_x=i, var_y=j, selected_lags=None,
                conds_x=conds_x, conds_y=conds_y,
                tau_max=tau_max_here,
                significance=significance, sig_lev=sig_lev,
                sig_samples=sig_samples, fixed_thres=fixed_thres,
                confidence=confidence, conf_lev=conf_lev,
                conf_samples=conf_samples,
                measure_params=measure_params,
                verbosity=verbosity)

            lagfunc_slice[i, :] = lag_func['cmi'][:]
            sig_thres_slice[i, :] = lag_func['cmi_sig'][:]

            if confidence:
                conf_slice[i, :] = lag_func['cmi_conf'][:]

        return lagfunc_slice, sig_thres_slice, conf_slice

    lagfuncs = numpy.zeros((N,  N, tau_max + 1))
    sig_thres = numpy.zeros((N,  N, tau_max + 1))
    confs = numpy.zeros((N, N, tau_max + 1, 2))

    for j in selected_variables:
        (lagfuncs[:, j],
         sig_thres[:, j],
         confs[:, j]) = calc(j, contemp_only=False)

    if solid_contemp_links:
        for j in selected_variables:
            res = calc(j, contemp_only=True)
            (lagfuncs[:, j, 0],
             sig_thres[:, j, 0],
             confs[:, j, 0]) = (res[0][:, 0], res[1][:, 0], res[2][:, 0])

    # if ('cmi' in measure and measure_params['rescale_cmi']):
    #     lagfuncs = _par_corr_trafo(lagfuncs)
    #     sig_thres = _par_corr_trafo(sig_thres)
    #     confs = _par_corr_trafo(confs)

    return lagfuncs, sig_thres, confs


##
# Helper functions
##

def get_sig_thres(sig_lev, df, measure='par_corr', array=None):
    """Returns the significance threshold of the pearson correlation
    coefficient according to a Student's t-distribution with df degrees
    of freedom.

    One- or two-tailedness should be accounted for by the choice of sig_lev,
    i.e., 95% two-tailed significance corresponds to sig_lev = 0.975

    :type sig_lev: float
    :param sig_lev: significance level

    :type df: int
    :param df: degrees of freedom

    :rtype: float
    :returns: significance threshold
    """

    if measure == 'par_corr':
        thres = stats.t.ppf(
            sig_lev, df) / numpy.sqrt(df + stats.t.ppf(sig_lev, df) ** 2)
    elif measure == 'cmi_gauss':
        thres = stats.t.ppf(
            sig_lev, df) / numpy.sqrt(df + stats.t.ppf(sig_lev, df) ** 2)
        thres = _par_corr_to_cmi(thres)
    elif measure == 'reg':
        # Here the degrees of freedom are minus one because of the intercept
        _, coeff_error, resid_error = _estimate_standardized_regression(array)
        thres = stats.t.ppf(sig_lev, df-1) * coeff_error
    else:
        raise ValueError("Analytical significance level only available"
                         " for measures 'par_corr', 'reg', and 'cmi_gauss'")
        # thres = stats.t.ppf(
        #     sig_lev, df) / numpy.sqrt(df + stats.t.ppf(sig_lev, df) ** 2)
        # thres = -0.5 * numpy.log(1. - thres ** 2)

    return thres


def _calculate_lag_function(measure, data, data_mask=None,
                            mask=False,
                            mask_type=None, min_samples=20,
                            var_x=0, var_y=1, conds_x=None, conds_y=None,
                            measure_params=None,
                            tau_max=0,
                            selected_lags=False,
                            significance=False, sig_lev=0.95,
                            sig_samples=100, fixed_thres=0.015,
                            confidence=False, conf_lev=0.95,
                            conf_samples=100,
                            verbosity=0):
    """
    Wrapper around measure estimation to compute lag function
    and take care of masks, ...

    :rtype: dictionary of arrays
    :returns: dictionary of arrays with CMIs / partial correlations

    """

    if conds_x is None:
        conds_x = []
    if conds_y is None:
        conds_y = []

    if not selected_lags:
        selected_lags = range(tau_max + 1)

    T_data, N = data.shape

    cmi = numpy.zeros(tau_max + 1)
    cmi_sig = numpy.zeros(tau_max + 1)
    cmi_conf = numpy.zeros((tau_max + 1, 2))

    auto = var_x == var_y

    for tau in range(tau_max + 1):

        if (tau not in selected_lags) or (auto and tau == 0):
            continue

        # Construct lists of tuples for estimating
        # I(X_t-tau; Y_t | Z^Y_t, Z^X_t-tau)
        # with conditions for X shifted by tau
        X = [(var_x, -tau)]
        Y = [(var_y, 0)]
        Z = conds_y + [(node[0], -tau + node[1]) for node in conds_x]

        array, xyz = _construct_array(
                        X=X, Y=Y, Z=Z,
                        tau_max=tau_max,
                        data=data,
                        mask=mask,
                        data_mask=data_mask,
                        mask_type=mask_type,
                        verbosity=verbosity)

        dim, T_eff = array.shape
        if T_eff < min_samples:
            cmi[tau] = numpy.nan
            cmi_sig[tau] = numpy.nan
            cmi_conf[tau] = numpy.nan
            if verbosity > 1:
                print("Only %d overlapping samples for link (%s, %s) "
                      "with tau = %d, setting estimate to NaN"
                      " (change current threshold min_samples = %d in code)"
                      % (T_eff, var_x, var_y, tau, min_samples))
            continue

        cmi[tau] = _get_estimate(array=array, measure=measure, xyz=xyz,
                                 measure_params=measure_params,
                                 verbosity=verbosity)

        # Calculate empirical significance level, x and its conditions are
        # shuffled separately of y with its conditions
        # if conditions are in both parents, they are randomly
        # shuffled or not shuffled for each surrogate...
        if significance:
            cmi_sig[tau] = _get_significance_estimate(
                array=array, xyz=xyz, significance=significance,
                measure=measure,
                sig_samples=sig_samples, sig_lev=sig_lev,
                fixed_thres=fixed_thres,
                measure_params=measure_params,
                verbosity=verbosity)

        if confidence:
            cmi_conf[tau] = _get_confidence_estimate(
                array=array, xyz=xyz,
                confidence=confidence,
                value=cmi[tau],
                measure=measure,
                conf_samples=conf_samples,
                conf_lev=conf_lev,
                measure_params=measure_params,
                verbosity=verbosity)

        if verbosity > 1:
            printstr = "            %s (tau=%d) = %.3f" % (measure,
                                                           tau,
                                                           cmi[tau])
            if significance:
                printstr += " | sig thres = %.3f" % cmi_sig[tau]
            if confidence:
                printstr += " | conf bounds = (%.3f, %.3f)" % (
                    cmi_conf[tau][0], cmi_conf[tau][1])

            printstr += " | sample length %d, dimension %d" % (
                T_eff, dim)

            print(printstr)

    return {'cmi': cmi, 'cmi_sig': cmi_sig, 'cmi_conf': cmi_conf}


def _construct_array(X, Y, Z, tau_max, data, mask=False,
                     data_mask=None, mask_type=None,
                     verbosity=0):
    """Returns array containing ( (X, Y, Z), samples) and XYZ identifier"""

    def uniq(input):
        output = []
        for x in input:
            if x not in output:
                output.append(x)
        return output

    if mask_type is None:
        mask_type = ['x', 'y', 'z']

    T, N = data.shape

    # Remove duplicates in X, Y, Z
    X = uniq(X)
    Y = uniq(Y)
    Z = uniq(Z)

    if len(X) == 0:
        raise ValueError("X must be non-zero")
    if len(Y) == 0:
        raise ValueError("Y must be non-zero")

    # If a node in Z occurs already in X or Y, remove it from Z
    Z = [node for node in Z if (node not in X) and (node not in Y)]

    # Check that all lags are non-positive and indices are in [0,N-1]
    XYZ = X + Y + Z
    dim = len(XYZ)
    if numpy.array(XYZ).shape != (dim, 2):
        raise ValueError("X, Y, Z must be lists of tuples in format"
                         " [(var, -lag),...], eg., [(2, -2), (1, 0), ...]")
    if numpy.any(numpy.array(XYZ)[:, 1] > 0):
        raise ValueError("nodes are %s, " % str(XYZ) +
                         "but all lags must be non-positive")
    if (numpy.any(numpy.array(XYZ)[:, 0] >= N)
       or numpy.any(numpy.array(XYZ)[:, 0] < 0)):
        raise ValueError("variable indices %s," % str(numpy.array(XYZ)[:, 0]) +
                         " but must be in [0, %d]" % (N-1))
    if numpy.all(numpy.array(Y)[:, 1] < 0):
        raise ValueError("Y-nodes are %s, " % str(Y) +
                         "but one of the Y-nodes must have zero lag")

    max_lag = max(abs(numpy.array(XYZ)[:, 1].min()), tau_max)

    # Setup XYZ identifier
    xyz = numpy.array([0 for i in range(len(X))] +
                      [1 for i in range(len(Y))] +
                      [2 for i in range(len(Z))])

    # Setup and fill array with lagged time series
    array = numpy.zeros((dim, T - max_lag))
    for i, node in enumerate(XYZ):
        var, lag = node
        array[i, :] = data[max_lag + lag: T + lag, var]

    if mask:
        # Remove samples with data_mask == 0
        # conditional on which mask_type is used
        array_mask = numpy.zeros((dim, T - max_lag))
        for i, node in enumerate(XYZ):
            var, lag = node
            array_mask[i, :] = data_mask[max_lag + lag: T + lag, var]

        use_indices = numpy.ones(T - max_lag, dtype='int')
        if 'x' in mask_type:
            use_indices *= numpy.prod(array_mask[xyz == 0, :], axis=0)
        if 'y' in mask_type:
            use_indices *= numpy.prod(array_mask[xyz == 1, :], axis=0)
        if 'z' in mask_type:
            use_indices *= numpy.prod(array_mask[xyz == 2, :], axis=0)
        array = array[:, use_indices == 1]

    if verbosity > 2:
        print("            Constructed array of shape " +
              "%s from\n" % str(array.shape) +
              "            X = %s\n" % str(X) +
              "            Y = %s\n" % str(Y) +
              "            Z = %s" % str(Z))
        if mask:
            print("            with masked samples in "
                  "%s removed" % str(mask_type))

    return array, xyz


def _get_estimate(array, measure, xyz, measure_params, verbosity=0):

    if measure == 'par_corr':
        return _estimate_partial_correlation(
            array=numpy.copy(array))
    elif measure == 'reg':
        return _estimate_standardized_regression(
            array=numpy.copy(array), verbosity=verbosity)[0]
    elif measure == 'cmi_gauss':
        return _par_corr_to_cmi(_estimate_partial_correlation(
                                array=numpy.copy(array)))
    elif measure == 'cmi_knn':
        return _estimate_cmi_knn(
            array=numpy.copy(array),
            k=measure_params['knn'], xyz=xyz)
    elif measure == 'cmi_symb':
        return _estimate_cmi_symb(
            array=numpy.copy(array), xyz=xyz)


def _get_significance_estimate(array, xyz, significance, measure,
                               sig_samples, sig_lev, fixed_thres,
                               measure_params,
                               verbosity=0):
    """Computes significance threshold."""

    dim, T = array.shape

    if significance == 'shuffle':

        x_indices = numpy.where(xyz == 0)[0]

        if verbosity > 2:
            print("            Shuffle significance test: jointly shuffling "
                  "indices %s of array" % str(x_indices))

        cmitmp = numpy.zeros(sig_samples)
        for sam in range(sig_samples):

            perm = numpy.random.permutation(T)
            array_shuffled = numpy.copy(array)
            for i in x_indices:
                array_shuffled[i] = array[i, perm]

            cmitmp[sam] = _get_estimate(array=array_shuffled,
                                        measure=measure, xyz=xyz,
                                        measure_params=measure_params)

        # Sort and get quantile
        cmitmp.sort()
        if verbosity > 2:
            print("            ...done!")
        return cmitmp[sig_lev * sig_samples]

    elif significance == 'analytic':
        return get_sig_thres(
            sig_lev, df=T - dim, measure=measure, array=array)

    elif significance == 'fixed':
        return fixed_thres


def _get_confidence_estimate(array, xyz, confidence, measure, value,
                             conf_samples, conf_lev,
                             measure_params,
                             verbosity=0):

    dim, T = array.shape

    # confidence interval is two-sided
    c_int = (1. - (1. - conf_lev) / 2.)

    if confidence == 'analytic':

        if measure == 'par_corr':
            value_tdist = value*numpy.sqrt(T-dim) / numpy.sqrt(1.-value**2)
            lower = (stats.t.ppf(q=1.-c_int, df=T-dim, loc=value_tdist)
                     / numpy.sqrt(T-dim + stats.t.ppf(q=1.-c_int, df=T-dim,
                                                      loc=value_tdist)**2))
            upper = (stats.t.ppf(q=c_int, df=T-dim, loc=value_tdist)
                     / numpy.sqrt(T-dim + stats.t.ppf(q=c_int, df=T-dim,
                                                      loc=value_tdist)**2))
            return (lower, upper)

        elif measure == 'cmi_gauss':
            value_tdist = (_par_corr_trafo(value)*numpy.sqrt(T-dim)
                           / numpy.sqrt(1.-_par_corr_trafo(value)**2))
            lower = (stats.t.ppf(q=1.-c_int, df=T-dim, loc=value_tdist)
                     / numpy.sqrt(T-dim + stats.t.ppf(q=1.-c_int, df=T-dim,
                                                      loc=value_tdist)**2))
            upper = (stats.t.ppf(q=c_int, df=T-dim, loc=value_tdist)
                     / numpy.sqrt(T-dim + stats.t.ppf(q=c_int, df=T-dim,
                                                      loc=value_tdist)**2))
            return (_par_corr_to_cmi(lower), _par_corr_to_cmi(upper))

        elif measure == 'reg':
            # Here the degrees of freedom are -1 because of the intercept
            _, coeff_error, resid_error = _estimate_standardized_regression(
                                              array)
            lower = (stats.t.ppf(q=1.-c_int, df=T-dim-1,
                                 loc=value/coeff_error) * coeff_error)
            upper = (stats.t.ppf(q=c_int, df=T-dim-1,
                                 loc=value/coeff_error) * coeff_error)
            return (lower, upper)

        else:
            raise ValueError("Analytical confidence level only available "
                             "for measures 'par_corr', 'reg', and 'cmi_gauss'")

    elif confidence == 'bootstrap':
        if verbosity > 2:
            print("            Bootstrap confidence test")

        if measure == 'cmi_knn':
            conf_knn_class = ConfidenceCMIknn(
                                array=numpy.copy(array),
                                k=measure_params['knn'], xyz=xyz)

        cmitmp = numpy.zeros(conf_samples)
        for sam in range(conf_samples):

            if measure == 'cmi_knn':
                # For knn a bootstrap would create ties in the data,
                # therefore here only the k's are bootstrapped,
                # see Runge PhD thesis
                cmitmp[sam] = conf_knn_class.get_single_estimate()

            else:
                array_bootstrap = array[:, numpy.random.randint(0, T, T)]
                if measure == 'par_corr':
                    cmitmp[sam] = _estimate_partial_correlation(
                        array=array_bootstrap)
                elif measure == 'cmi_gauss':
                    cmitmp[sam] = _par_corr_to_cmi(
                        _estimate_partial_correlation(
                            array=array_bootstrap))
                elif measure == 'reg':
                    cmitmp[sam] = _estimate_standardized_regression(
                        array=array_bootstrap)[0]
                elif measure == 'cmi_symb':
                    cmitmp[sam] = _estimate_cmi_symb(
                        array=array_bootstrap, xyz=xyz)

        # Sort and get quantile
        cmitmp.sort()
        return (cmitmp[(1. - c_int) * conf_samples],
                cmitmp[c_int * conf_samples])

    else:
        raise ValueError("Confidence must be 'analytic' or 'bootstrap'")


def _get_parents(nodes, exclude=None):

    graph = []
    for var, lag in nodes:
        if lag != 0 and (var, lag) != exclude:
            graph.append((var, lag))

    return graph


def _get_neighbors(nodes, exclude=None):

    graph = []
    for var, lag in nodes:
        if lag == 0 and (var, lag) != exclude:
            graph.append((var, lag))

    return graph


def _par_corr_trafo(cmi):
    """Transformation of conditional mutual information to partial correlation
       scale using the (multivariate) Gaussian assumption.
    """

    # Set negative values to small positive number
    # (zero would be interpreted as non-significant in some functions)
    if numpy.rank(cmi) == 0:
        if cmi < 0.:
            cmi = 1E-8
    else:
        cmi[cmi < 0.] = 1E-8

    return numpy.sqrt(1. - numpy.exp(-2. * cmi))


def _par_corr_to_cmi(par_corr):
    """Transformation of partial correlation to conditional mutual information
       scale using the (multivariate) Gaussian assumption.
    """

    return -0.5*numpy.log(1. - par_corr**2)


##
# Functions to estimate different measures of conditional association
##

def _estimate_partial_correlation(array):
    """Returns the partial correlation of y and x, conditioning on
    confounds of shape (dim, T).

    The first two indices of array are assumed to be X and Y.

    :type array: array
    :param array: data array of shape (dim, T)

    :rtype: float
    :returns: partial correlation
    """

    D, T = array.shape
    if numpy.isnan(array).sum() != 0:
        raise ValueError("nans in the array!")

    # Standardize
    array -= array.mean(axis=1).reshape(D, 1)
    array /= array.std(axis=1).reshape(D, 1)
    if numpy.isnan(array).sum() != 0:
        raise ValueError("nans after standardizing, "
                         "possibly constant array!")

    (x, y) = _get_residuals(array)

    return numpy.dot(x, y) / numpy.sqrt(numpy.dot(x, x) * numpy.dot(y, y))

    # Equivalent to
#        inv_cov_num = linalg.pinv(numpy.corrcoef(array))
#        return -inv_cov_num[0,1]/numpy.sqrt(inv_cov_num[0,0]*inv_cov_num[1,1])


def _estimate_standardized_regression(array, standardize=True, verbosity=0):
    """Returns the partial regression of y and x, conditionning on
    confounds of shape (dim, T).

    The first two indices of array are assumed to be X and Y.

    :type array: array
    :param array: data array of shape (dim, T)

    :rtype: float
    :returns: partial regression
    """

    try:
        import statsmodels.api as sm
    except:
        print("Could not import statsmodels, regression not possible!")

    dim, _ = array.shape

    # Standardize
    if standardize:
        array -= array.mean(axis=1).reshape(dim, 1)
        array /= array.std(axis=1).reshape(dim, 1)
        if numpy.isnan(array).sum() != 0:
            raise ValueError("nans after standardizing, "
                             "possibly constant array!")

    Y = array[1, :]
    X = array[[0] + range(2, dim), :].T

    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()

    # The first parameter is the constant, which we are not interested in here
    parameter = results.params[1]
    standard_error = results.bse[1]
    resid_error = results.mse_resid
    if verbosity > 3:
        print(results.summary())

    return parameter, standard_error, resid_error


def _get_nearest_neighbors(array, xyz, k, standardize=True):

    dim, T = array.shape

    if standardize:
        # Standardize
        array = array.astype('float32')
        array -= array.mean(axis=1).reshape(dim, 1)
        array /= array.std(axis=1).reshape(dim, 1)
        # If the time series is constant, return nan rather than raising
        # Exception
        if numpy.isnan(array).sum() != 0:
            raise ValueError("nans after standardizing, "
                             "possibly constant array!")

    # Add noise to destroy ties...
    array += (1E-6 * array.std(axis=1).reshape(dim, 1)
              * numpy.random.rand(array.shape[0], array.shape[1]))

    # Flatten for fast weave.inline access
    array = array.flatten()

    dim_x = int(numpy.where(xyz == 0)[0][-1] + 1)
    dim_y = int(numpy.where(xyz == 1)[0][-1] + 1 - dim_x)
    # dim_z = maxdim - dim_x - dim_y

    # Initialize
    k_xz = numpy.zeros(T, dtype='int32')
    k_yz = numpy.zeros(T, dtype='int32')
    k_z = numpy.zeros(T, dtype='int32')

    code = """
    int i, j, index=0, t, m, n, d, kxz, kyz, kz, indexfound[T];//
    double  dz=0., dxyz=0., dx=0., dy=0., eps, epsmax;
    double dist[T*dim], dxyzarray[k+1];

    // Loop over time
    for(i = 0; i < T; i++){

        // Growing cube algorithm: Test if n = #(points in epsilon-
        // environment of reference point i) > k
        // Start with epsilon for which 95% of points are inside the cube
        // for a multivariate Gaussian
        // eps increased by 2 later, also the initial eps
        eps = 1.*pow( float(k)/float(T), 1./dim);

        // n counts the number of neighbors
        n = 0;
        while(n <= k){
            // Increase cube size
            eps *= 2.;
            // Start with zero again
            n = 0;
            // Loop through all points
            for(t = 0; t < T; t++){
                d = 0;
                while(fabs(array[d*T + i] - array[d*T + t] ) < eps
                        && d < dim){
                        d += 1;
                }
                // If all distances are within eps, the point t lies
                // within eps and n is incremented
                if(d == dim){
                    indexfound[n] = t;
                    n += 1;
                }
            }
        }

        // Calculate distance to points only within epsilon environment
        // according to maximum metric
        for(j = 0; j < n; j++){
            index = indexfound[j];

            dxyz = 0.;
            for(d = 0; d < dim; d++){
                dist[d*T + j] = fabs(array[d*T + i] - array[d*T + index]);
                dxyz = fmax( dist[d*T + j], dxyz);
            }

            // Use insertion sort
            dxyzarray[j] = dxyz;
            if ( j > 0 ){
                // only list of smallest k+1 distances need to be kept!
                m = fmin(k, j-1);
                while ( m >= 0 && dxyzarray[m] > dxyz ){
                    dxyzarray[m+1] = dxyzarray[m];
                    m -= 1;
                }
                dxyzarray[m+1] = dxyz;
            }

        }

        // Epsilon of k-th nearest neighbor in joint space
        epsmax = dxyzarray[k];

        // Count neighbors within epsmax in subspaces, since the reference
        // point is included, all neighbors are at least 1
        kz = 0;
        kxz = 0;
        kyz = 0;
        for(j = 0; j < T; j++){

            // X-subspace
            dx = fabs(array[0*T + i] - array[0*T + j]);
            for(d = 1; d < dim_x; d++){
                dist[d*T + j] = fabs(array[d*T + i] - array[d*T + j]);
                dx = fmax( dist[d*T + j], dx);
            }

            // Y-subspace
            dy = fabs(array[dim_x*T + i] - array[dim_x*T + j]);
            for(d = dim_x; d < dim_x+dim_y; d++){
                dist[d*T + j] = fabs(array[d*T + i] - array[d*T + j]);
                dy = fmax( dist[d*T + j], dy);
            }

            // Z-subspace, if empty, dz stays 0
            dz = 0.;
            for(d = dim_x+dim_y; d < dim ; d++){
                dist[d*T + j] = fabs(array[d*T + i] - array[d*T + j]);
                dz = fmax( dist[d*T + j], dz);
            }

            // For no conditions, kz is counted up to T
            if (dz < epsmax){
                kz += 1;
                if (dx < epsmax){
                    kxz += 1;
                }
                if (dy < epsmax){
                    kyz += 1;
                }
            }
        }
        // Write to numpy arrays
        k_xz[i] = kxz;
        k_yz[i] = kyz;
        k_z[i] = kz;

    }
    """

    vars = ['array', 'T', 'dim_x', 'dim_y',
            'k', 'dim', 'k_xz', 'k_yz', 'k_z']

    weave.inline(
        code, vars, headers=["<math.h>"], extra_compile_args=['-O3'])

    return k_xz, k_yz, k_z


def _estimate_cmi_knn(array, k, xyz, norm=0, standardize=True,
                      verbosity=0):
    """Estimation of CMI I(X;Y|Z) as in S. Frenzel and B. Pompe,
    Phys. Rev. Lett. 99, 204101 (2007).

    :type array: array
    :param array: input data in shape (maxdim, T)

    :type k: integer
    :param k: nearest neighbor parameter

    :type xyz: array
    :param xyz: xyz=0 stands for X, xyz=1 for Y and xyz=2 for Z

    :type norm: integer
    :param norm: desired normalization of CMI

    :type standardize: boolean
    :param standardize: whether to standardize data before estimation

    :type verbosity: integer
    :param verbosity: verbosity

    :rtype: float
    :returns: estimated CMI
    """

    k_xz, k_yz, k_z = _get_nearest_neighbors(array=array, xyz=xyz,
                                             k=k, standardize=standardize)

    ixy_z = (special.digamma(k)
             + (- special.digamma(k_xz)
                - special.digamma(k_yz)
                + special.digamma(k_z)).mean())

    if norm == 0:
        cmi = ixy_z
    elif norm == 1:
        cmi = numpy.sqrt(1. - numpy.exp(-2. * numpy.maximum(0., ixy_z)))
    # elif norm == 2: cmi = ixy_z /  (- special.digamma(k_yz) +
    # special.digamma(k_z)).mean()

    return cmi


class ConfidenceCMIknn():

    def __init__(self, array, k, xyz, norm=0, standardize=True, verbosity=0):
        """Estimation of CMI I(X;Y|Z) as in S. Frenzel and B. Pompe,
        Phys. Rev. Lett. 99, 204101 (2007).

        :type array: array
        :param array: input data in shape (maxdim, T)

        :type k: integer
        :param k: nearest neighbor parameter

        :type xyz: array
        :param xyz: xyz=0 stands for X, xyz=1 for Y and xyz=2 for Z

        :type norm: integer
        :param norm: desired normalization of CMI

        :type standardize: boolean
        :param standardize: whether to standardize data before estimation

        :type verbosity: integer
        :param verbosity: verbosity

        :rtype: float
        :returns: estimated CMI
        """

        dim, self.T = array.shape
        self.k = int(k)
        self.norm = norm

        self.k_xz, self.k_yz, self.k_z = _get_nearest_neighbors(
                                            array=array,
                                            xyz=xyz,
                                            k=k,
                                            standardize=standardize)

    def get_single_estimate(self):
        """Returns a bootstrap estimate of CMI using precomputed
         k_xz, k_yz, k_z"""

        randints = numpy.random.randint(0, self.T, self.T)
        ixy_z = (special.digamma(self.k)
                 + (- special.digamma(self.k_xz[randints])
                    - special.digamma(self.k_yz[randints])
                    + special.digamma(self.k_z[randints])).mean())

        if self.norm == 0:
            cmi = ixy_z
        elif self.norm == 1:
            cmi = numpy.sqrt(1. - numpy.exp(-2. * numpy.maximum(0., ixy_z)))
        # elif norm == 2: cmi = ixy_z /  (- special.digamma(k_yz) +
        # special.digamma(k_z)).mean()

        return cmi


def _get_residuals(array):
    """Returns residuals of linear multivariate regression of first two rows
    on remaining rows."""

    x = array[0, :]
    y = array[1, :]
    if len(array) > 2:
        confounds = array[2:, :]
        ortho_confounds = linalg.qr(
            numpy.fastCopyAndTranspose(confounds), mode='economic')[0].T
        x -= numpy.dot(numpy.dot(ortho_confounds, x), ortho_confounds)
        y -= numpy.dot(numpy.dot(ortho_confounds, y), ortho_confounds)

    return (x, y)


##
# Methods for binning CMI estimation
##

def _plogp_vector(T, grass=False):
    """
    Precalculation of p*log(p) needed for entropies.

    :type T: float
    :param T: sample length

    :type grass: bool
    :param grass: use ln or Grassberger's estimator

    :rtype: array
    :returns: p*log(p) array from p=1 to p=T
    """

    gfunc = numpy.zeros(T + 1, dtype='float32')

    if grass:
        gfunc = numpy.zeros(T + 1)
        # This calculates the summands of Grassberger's best analytic
        # estimator for Shannon's Entropy
        gfunc[0] = 0.
        gfunc[1] = - .5772156649015328606065 - numpy.log(2.)
        for t in range(2, T + 1):
            if t % 2 == 0:
                gfunc[t] = t * (gfunc[t - 1] / (t - 1.) + 2. / (t - 1.))
            else:
                gfunc[t] = t * gfunc[t - 1] / (t - 1.)

    else:
        gfunc = numpy.zeros(T + 1)
        gfunc[1:] = numpy.arange(
            1, T + 1, 1) * numpy.log(numpy.arange(1, T + 1, 1))

    def plogp_func(t):

        return gfunc[t]

    return numpy.vectorize(plogp_func)


def _bincount_hist(symb_array):
    """
    Computes histogram from symbolic array.

    :type symb_array: array of integers
    :param symb_array: symbolic data

    :rtype: array
    :returns: (unnormalized) histogram
    """

    base = int(symb_array.max() + 1)

    D, T = symb_array.shape

    # Needed because numpy.bincount cannot process longs
    if type(base ** D) != int:
        raise ValueError("Too many bins and/or dimensions, "
                         "numpy.bincount cannot process longs")
    if base ** D * 16. / 8. / 1024. ** 3 > 3.:
        raise ValueError("Dimension exceeds 3 GB of necessary "
                         "memory (change this code line if you got more...)")
    if D * base ** D > 2 ** 65:
        raise ValueError("base = %d, D = %d: Histogram failed: "
                         "dimension D*base**D exceeds int64 data type"
                         % (base, D))

    flathist = numpy.zeros((base ** D), dtype='int16')
    multisymb = numpy.zeros(T, dtype='int64')

    for i in range(D):
        multisymb += symb_array[i, :] * base ** i

    result = numpy.bincount(multisymb)
    flathist[:len(result)] += result

    return flathist.reshape(tuple([base, base]
                                  + [base for i in range(D - 2)])).T


def _estimate_cmi_symb(array, xyz, norm=0):
    """
    Estimates CMI from symbolic array using histograms.

    :type array: array
    :param array: input data in shape (maxdim, T)

    :type xyz: array
    :param xyz: xyz=0 stands for X, xyz=1 for Y and xyz=2 for Z

    :type maxdim: integer
    :param maxdim: maximum actual dimension

    :type T: integer
    :param T: sample length

    :type norm: integer
    :param norm: desired normalization of CMI

    :rtype: float
    :returns: estimated CMI
    """

    maxdim, T = array.shape

    plogp = _plogp_vector(T)

    # High-dimensional Histogram
    hist = _bincount_hist(array)

    # Entropies by use of vectorized function plogp (either log or
    # Grassberger's estimator)
    hxyz = (-(plogp(hist)).sum() + plogp(T)) / float(T)
    hxz = (-(plogp(hist.sum(axis=1))).sum() + plogp(T)) / \
        float(T)
    hyz = (-(plogp(hist.sum(axis=0))).sum() + plogp(T)) / \
        float(T)
    hz = (-(plogp(hist.sum(axis=0).sum(axis=0))).sum() +
          plogp(T)) / float(T)

    ixy_z = hxz + hyz - hz - hxyz
#            hx = hxz - hz
#            hy = hyz - hz
#        print hxz, '\t', hyz, '\t', hz, '\t', hxyz, '\t', ixy_z
    if norm == 0:
        cmi = ixy_z
    elif norm == 1:
        cmi = numpy.sqrt(1. - numpy.exp(-2. * numpy.maximum(0., ixy_z)))

    return cmi