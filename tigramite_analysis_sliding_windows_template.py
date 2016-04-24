#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# TiGraMITe -- Time Series Graph Based Measures of Information Transfer
#
# Methods are described in:
#    J. Runge et al., Nature Communications, 6, 8502 (2015)
#    J. Runge, J. Heitzig, V. Petoukhov, and J. Kurths,
#       Phys. Rev. Lett. 108, 258701 (2012)
#    J. Runge, J. Heitzig, N. Marwan, and J. Kurths,
#       Phys. Rev. E 86, 061121 (2012)
#    J. Runge, V. Petoukhov, and J. Kurths, Journal of Climate, 27.2 (2014)
#
# Please cite all references when using the method.
#
# Copyright (C) 2012-2016 Jakob Runge <jakobrunge@posteo.de>
# https://github.com/jakobrunge/tigramite.git
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Script to estimate time series graph and measures like MIT, ITY...
"""

#
#  Import essential tigramite modules
#
from tigramite_src import tigramite_preprocessing as pp
# import tigramite_preprocessing_geo as ppgeo

from tigramite_src import tigramite_estimation_beta as tigramite_estimation
from tigramite_src import tigramite_plotting

# import Parallel module (based on mpi4py)
# import mpi

#  Import NumPy for the array object and fast numerics
import numpy

# import file handling packages
import os
import sys
import pickle

# import plotting functions
import matplotlib
from matplotlib import pyplot


###
# Which operations to perform
###

# Estimate parents/neighbors and lag funcdionts;
# If False, these will be taken from results dict with name:
# os.path.expanduser(save_folder) + project_name + '_results.pkl'
estimate = True

# Plotting functions: the sections below can be flexibly adapted
plot_time_series = True
plot_lag_functions = True
plot_graph = True


###
# Some parameters used in different steps of the script
###

save_folder = 'test/'
project_name = 'test_sliding'
save_fig_format = 'pdf'
verbosity = 3

# Optionally start logging all print output, uncomment below as well
# sys.stdout = pp.Logger()


params = {'xtick.labelsize': 5,
          'ytick.labelsize': 5,
          'xtick.major.size': 1,      # major tick size in points
          'xtick.minor.size': .5,      # minor tick size in points
          'ytick.major.size': 1,      # major tick size in points
          'ytick.minor.size': .5,      # minor tick size in points
          'xtick.major.pad': 1,      # distance to major tick label
          'xtick.minor.pad': 1,
          'ytick.major.pad': 2,      # distance to major tick label
          'ytick.minor.pad': 2,
          'axes.labelsize': 8,
          'axes.linewidth': .3,
          'text.fontsize': 8,
          'lines.markersize': 4,            # markersize, in points
          'legend.fontsize': 5,
          'legend.numpoints': 2,
          'legend.handlelength': 1.
          }
matplotlib.rcParams.update(params)


###
# Data preparation: provide numpy arrays "fulldata" (float) and
# "sample_selector" (bool), both of shape
# (Time, Variables)
# and datatime (float array) of shape (Time,)
###

# Test process: Vector-Autoregressive Process, see docs in "pp"-module
a = .7
c1 = .6
c2 = -.6
c3 = .8
T = 1000
links_coeffs = {0: [((0, -1), a)],
                1: [((1, -1), a), ((0, -1), c1)],
                2: [((2, -1), a), ((1, -2), c2)],
                3: [((3, -1), a), ((0, -3), c3)],
                }

fulldata, true_parents_neighbors = pp.var_process(links_coeffs,
                                                  use='inv_inno_cov', T=T)
T, N = fulldata.shape

###
# Possibly supply mask as a boolean array. Samples with a "0" are masked out.
# The variable sample_selector needs to be of the same shape as fulldata.
###

sample_selector = numpy.ones(fulldata.shape).astype('bool')

##
# Possibly construct symbolic time series for use with measure = 'symb'
##

# (fulldata, sample_selector, T) = pp.ordinal_patt_array(
#                                   fulldata, sample_selector,
#                                   dim=2, step=1, verbosity=0)

# fulldata = pp.quantile_bin_array(fulldata, bins = 3)
# print fulldata

##
# Define time sequence (only used for plotting)
##
datatime = numpy.arange(0, fulldata.shape[0], 1.)

# Initialize results dictionary with important variables that are used
# in different analysis steps and should be saved to the results dictionary.
# All less important (eg plotting) parameters can be local...

d = {
    # Data
    'fulldata': fulldata,
    'N': fulldata.shape[1],
    'T': fulldata.shape[0],
    'datatime': datatime,
    'window_length': 200,
    'window_steps': 200,

    # Analyze only masked samples
    # selector_type needs to be a list containing 'x' or 'y'or 'z' or any
    # combination. This will ignore masked values if they are in the
    # lagged variable X, the 'driven' variable Y and/or the condition Z in
    # the association measure I(X;Y | Z), which enables to, e.g., only
    # consider the impacton summer months. More use cases will bediscussed
    # in future papers...
    'selector': False,
    'sample_selector': sample_selector,
    'selector_type': ['y'],

    # Measure of association and params
    # - 'par_corr': linear partial correlation,
    # - 'reg': linear standardized partial regression
    # - 'cmi_knn': conditional mutual information (CMI)
    #   estimated using nearest neighbors
    # - 'cmi_symb': CMI using symbolic time series
    #   (from binning or ordinal patterns, the data
    #   can be converted using the functions in
    #   the module "pp")
    'measure': 'reg',

    # Quantities to estimate using estimated parents
    # 'none' for MI/cross correlation
    # 'parents_xy' for MIT
    # 'parents_y' for ITY
    # 'parents_x' for ITX
    # These measures are described in Runge et al. PRE (2012).
    'cond_types': ['none', 'parents_xy'],

    # for measure='cmi_knn': nearest neighbor parameter
    # used in causal algorithm (higher k reduce the
    # variance of the estimator, better for
    # independence tests).
    # Recommended: 5% - 50% of time series length
    'measure_params_algo': None,

    # nearest neighbor parameter used in the
    # subsequent estimation of MI, MIT, ITY, ...
    # (smaller k has smaller bias)
    # Recommended: 5..10, independent of T
    'measure_params_lagfuncs': None,

    # Causal Algorithm for estimation of parents/neighbors
    # Maximum time lag up to which links are tested
    'tau_max': 12,

    # Initial number of conditions to use, corresponds
    # to n_0 in Runge et al. PRL (2012).
    # Larger initial_conds speeds up the algorithm,
    # but leads to slightly more false positives.
    'initial_conds': 2,

    # Maximum number of conditions to use
    # (parameter n_max in my phd thesis)
    # Recommended: 4..6  for CMI estimation,
    # for 'par_corr' or 'reg' more can be used.
    'max_conds': 20,

    # Maximum number of combinations of conditions
    # to check in algorithm (corresponds to number i
    # of iterations in n-loop in Runge PRL (2012))
    # Recommended: 3..6
    'max_trials': 10,

    # True for solid links as defined in Runge PRL + PRE (2012)
    # Recommended is  "True".
    'solid_contemp_links': True,

    # Significance testing in algorithm and lag functions estimation
    # - 'fixed': fixed threshold (specified below)
    # - 'analytic': sig_lev for analytical sample
    #  distribution of partial correlation or
    #  regression (Student's t)
    # - 'full_shuffle': shuffle test as described in
    #   Runge et al. PRL (2012)
    # Recommended for CMI: 'full_shuffle' or 'fixed'
    # Recommended for par_corr and reg: 'analytic'
    'significance': 'analytic',

    # significance level (1-alpha). Note that for
    # 'par_corr' or 'reg' the test is two-sided,
    # such that 0.95 actually corresponds to a 90%
    # significance level
    # Here the divisor "/ 2." account for a two-sided level
    'sig_lev': (1. - .01 / 2.),

    # Higher significance levels require a larger
    # number of shuffle test samples, i.e. 0.9 needs
    # about 50 samples, 0.95 about 100, .98 about 500.
    'sig_samples': 100,

    # fixed threshold for CMI. I recommend to use a
    # shuffle test for CMI to get an idea of typical
    # values (see output in command line). Note that
    # shuffle significance thresholds depend on the
    # estimation dimension.
    'fixed_thres': 0.05,

    # Confidence bounds to be displayed
    # in lag functions (not used in the algorithm)
    # - False: no bounds
    # - 'analytic': conf_lev for analytical sample
    #  distribution (Student's t)
    # - 'bootstrap': bootstrap confidence bounds
    # Recommended for CMI: 'bootstrap'
    # Recommended for par_corr and reg: 'analytic'
    'confidence': 'analytic',

    # 0.9 corresponds to 90% confidence interval.
    'conf_lev': .9,
    'conf_samples': 100,

    # Variable names and node positions for graph plots (in figure coords)
    # These can be adapted to basemap plots in plot section below
    'var_names': ['0', '1', '2', '3'],
    'node_pos': {'y': numpy.array([0.5, 1., 0., 0.5]),
                 'x': numpy.array([0., 0.5, 0.5, 1.])},
}


time_steps = numpy.arange(0, d['T'], d['window_steps'])

###
# Space for operations on the data using functions in tigramite modules
###

if estimate:
    ###
    # Estimate parents and neighbors
    ###

    if d['solid_contemp_links']:
        estimate_parents_neighbors = 'both'
    else:
        estimate_parents_neighbors = 'parents'

    d['results'] = {}

    for time_step in time_steps:

        d['results'][time_step] = {}

        data = d['fulldata'][time_step: time_step + d['window_length']]
        if d['selector']:
            sample_selector = d['sample_selector'][time_step:
                                                   time_step +
                                                   d['window_length']]
        else:
            sample_selector = None

        d['results'][time_step]['parents_neighbors'] = \
            tigramite_estimation.pc_algo_all(
            data=data,
            selector=d['selector'],
            selector_type=d['selector_type'],
            sample_selector=sample_selector,

            measure=d['measure'],
            measure_params=d['measure_params_algo'],

            estimate_parents_neighbors='both',
            tau_max=d['tau_max'],
            initial_conds=d['initial_conds'],
            max_conds=d['max_conds'],
            max_trials=d['max_trials'],
            significance=d['significance'],
            sig_lev=d['sig_lev'],
            fixed_thres=d['fixed_thres'],

            verbosity=verbosity)

        ###
        # Estimate lag functions for MIT, ITY, ...
        ###

        # 'none' for MI/cross correlation
        # 'parents_xy' for MIT
        # 'parents_y' for ITY
        # 'parents_x' for ITX
        # These measures are described in Runge et al. PRE (2012).
        # cond_types = ['none', 'parents_y', 'parents_xy']
        # d['cond_types'] = cond_types

        for which in d['cond_types']:

            if verbosity > 0:
                print("Estimating lag functions for condition type %s" % which)

            (d['results'][time_step][which],
             d['results'][time_step]['sig_thres_' + which],
             d['results'][time_step]['conf_' + which]
             ) = tigramite_estimation.get_lagfunctions(
                data=data,
                selector=d['selector'],
                selector_type=d['selector_type'],
                sample_selector=sample_selector,

                parents_neighbors=d['results'][time_step]['parents_neighbors'],
                cond_mode=which,
                solid_contemp_links=d['solid_contemp_links'],

                tau_max=d['tau_max'],
                max_conds=d['max_conds'],
                measure=d['measure'],
                measure_params=d['measure_params_lagfuncs'],

                significance=d['significance'],
                sig_lev=d['sig_lev'],
                sig_samples=d['sig_samples'],
                fixed_thres=d['fixed_thres'],

                confidence=d['confidence'],
                conf_lev=d['conf_lev'],
                conf_samples=d['conf_samples'],

                verbosity=verbosity
            )

    if verbosity > 0:
        print("Saving results as %s" % (os.path.expanduser(save_folder) +
                                        project_name +
                                        '_results.pkl'))

    pickle.dump(d, open(os.path.expanduser(save_folder) + project_name +
                        '_results.pkl', 'w'))
else:
    # Load results dict (Note that this will override the results dict
    # with parameters as specified above)
    d = pickle.load(open(os.path.expanduser(save_folder) +
                         project_name + '_results.pkl', 'r'))


if plot_time_series:

    if verbosity > 0:
        print("Plotting time series")

    fig, axes = pyplot.subplots(d['N'], sharex=True,
                                frameon=False, figsize=(4, 4))

    for i in range(d['N']):
        tigramite_plotting.add_timeseries(
            fig=fig, axes=axes, i=i,
            time=datatime,
            dataseries=d['fulldata'][:, i],
            label=d['var_names'][i],
            selector=d['selector'],
            sample_selector=d['sample_selector'][:, i],
            grey_masked_samples=True,
            data_linewidth=.5,
            skip_ticks_data_x=1,
            skip_ticks_data_y=2,
            unit=None,
            last=(i == d['N'] - 1),
            time_label='years',
            label_fontsize=8,
        )

    fig.subplots_adjust(bottom=0.15, top=.9, left=0.15, right=.95, hspace=.3)
    fig.savefig(os.path.expanduser(save_folder) + project_name +
                '_data.%s' % save_fig_format)


if plot_lag_functions:

    # Local params
    cond_types = d['cond_types']
    cond_attributes = {}
    cond_attributes['color'] = {'none': 'grey', 'parents_y': 'black'}
    cond_attributes['label'] = {'none': 'Corr', 'parents_y': 'ITY'}

    if d['measure'] == 'par_corr' or d['measure'] == 'reg':
        two_sided_thres = True
    else:
        two_sided_thres = False

    for which in cond_types:

        if verbosity > 0:
            print("Plotting lag functions %s" % which)

        minimum, maximum = 0., 0.
        for time_step in time_steps:
            minimum, maximum = (
                min(minimum, d['results'][time_step][which].min()),
                max(maximum, d['results'][time_step][which].max()))

        lag_func_matrix = tigramite_plotting.setup_matrix(
            N=d['N'],
            tau_max=d['tau_max'],
            var_names=d['var_names'],
            figsize=(3, 3),
            minimum=minimum,
            maximum=maximum,
            label_space_left=0.2,
            label_space_top=.1,
            legend_width=.2,
            legend_fontsize=8,
            x_base=1., y_base=0.4,
            plot_gridlines=False,
            lag_units='months',
            label_fontsize=8)

        n_colors = len(time_steps)
        cm = pyplot.get_cmap('jet')
        cgen = [cm(1. * col / n_colors) for col in range(n_colors)]

        for it, time_step in enumerate(time_steps):

            lag_func_matrix.add_lagfuncs(
                lagfuncs=d['results'][time_step][which],
                sig_thres=d['results'][time_step]['sig_thres_' + which],
                conf=d['results'][time_step]['conf_' + which],
                color=cgen[it],
                label=str(time_step),
                two_sided_thres=two_sided_thres,
                marker='.',
                markersize=1,
                alpha=1.,
                plot_confidence=d['confidence'],
                conf_lev=d['conf_lev'],
            )

        lag_func_matrix.savefig(os.path.expanduser(save_folder) +
                                project_name +
                                "_%s_lag_functions.%s"
                                % (which, save_fig_format))


if plot_graph:

    if verbosity > 0:
        print("Plotting graph")

    min_ensemble_frac = .3
    arrow_linewidth = 5.

    # Local params
    cond_types = d['cond_types']

    for which in cond_types:

        fig = pyplot.figure(figsize=(3.375, 2.4), frameon=False)
        ax = fig.add_subplot(111, frame_on=False)

        # Here basemap instances or other images can be appended to ax
        # the graph is then plotted (the node positions should be adapted)

        robust_lagfuncs = numpy.zeros((len(time_steps),
                                       d['N'], d['N'], d['tau_max'] + 1))
        robust_lagfuncs_sig_thres = numpy.zeros(
            (len(time_steps),
             d['N'], d['N'], d['tau_max'] + 1))

        for it, time_step in enumerate(time_steps):
            robust_lagfuncs[it] = d['results'][time_step][which]
            robust_lagfuncs_sig_thres[it] = \
                d['results'][time_step]['sig_thres_' + which]

        # minimum = robust_lagfuncs.min()
        # maximum = robust_lagfuncs.max()

        lagfuncs, _ = pp.weighted_avg_and_std(
            robust_lagfuncs, axis=0,
            weights=(numpy.abs(robust_lagfuncs) > robust_lagfuncs_sig_thres))
        robustness_links = (numpy.abs(robust_lagfuncs) >
                            robust_lagfuncs_sig_thres).mean(axis=0)
        sig_thres = numpy.zeros((d['N'], d['N'], d['tau_max'] + 1))
        sig_thres[robustness_links < min_ensemble_frac] = numpy.infty

        # Draw line that marks maximum line width
        x, y = numpy.array([[0.0, 0.05], [0.95, 0.95]])
        line = matplotlib.lines.Line2D(x, y, lw=arrow_linewidth,
                                       color='grey', alpha=1.,
                                       transform=ax.transAxes)
        ax.add_line(line)
        pyplot.text(0.08, .92, '100%' + '\n' + 'robust' + '\n' +
                    r'%.1f' % (min_ensemble_frac * 100.) +
                    '%', fontsize=6, transform=ax.transAxes,
                    verticalalignment='center',
                    horizontalalignment='left')
        x, y = numpy.array([[0.0, 0.05], [0.89, 0.89]])
        line = matplotlib.lines.Line2D(x, y,
                                       lw=min_ensemble_frac * arrow_linewidth,
                                       color='grey', alpha=1.,
                                       transform=ax.transAxes)
        ax.add_line(line)

        tigramite_plotting.plot_graph(
            fig=fig, ax=ax,
            lagfuncs=lagfuncs,
            sig_thres=sig_thres,
            var_names=d['var_names'],
            link_colorbar_label=d['measure'] + ' ' + which + ' (cross)',
            node_colorbar_label=d['measure'] + ' ' + which + ' (auto)',
            label_fontsize=8,
            # save_name=os.path.expanduser(save_folder) + project_name +
            # '_%s_graph.%s' % (which, save_fig_format),

            alpha=1.,
            node_size=20,
            node_label_size=10,
            node_pos=d['node_pos'],
            vmin_nodes=0,
            vmax_nodes=1,
            node_ticks=.4,
            cmap_nodes='OrRd',

            link_label_fontsize=8,
            arrow_linewidth=arrow_linewidth,
            arrowhead_size=20,
            curved_radius=.2,
            vmin_edges=-1,
            vmax_edges=1.,
            edge_ticks=.4,
            cmap_edges='RdBu_r',
            # link_width can be an array of the same shape
            # as lagfuncs, each non-zero entry
            # corresponding to the width of that
            # link. Will be normalized and scaled
            # with arrow_linewidth
            # Useful for ensemble analysis where the
            # width might correspond to the fraction
            # of graphs where a link is significant
            link_width=robustness_links,
        )
    fig.subplots_adjust(left=0.1, right=.9, bottom=.25, top=.95)
    savestring = os.path.expanduser(os.path.expanduser(save_folder) +
                                    project_name + '_%s_graph.%s' %
                                    (which, save_fig_format))
    pyplot.savefig(savestring)
