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
Module contains functions for the package tigramite.
"""

#
#  Import essential packages
#
#  Import NumPy for the array object and fast numerics
import numpy
import matplotlib.transforms as transforms
from matplotlib import pyplot, ticker
from matplotlib.ticker import FormatStrFormatter

from copy import deepcopy
import os
import tigramite_estimation_beta as tigramite_estimation


# TODO: Add proper docstrings for most functions...

def par_corr_trafo(cmi):
    """Transformation of CMI to partial correlation scale.

    Using the (multivariate) Gaussian assumption.

    Args:
        cmi (float): CMI value.

    Returns:
        Rescaled value.
    """
    # Set negative values to small positive number
    # (zero would be interpreted as non-significant in some functions)
    if numpy.ndim(cmi) == 0:
        if cmi < 0.:
            cmi = 1E-8
    else:
        cmi[cmi < 0.] = 1E-8

    return numpy.sqrt(1. - numpy.exp(-2. * cmi))


def par_corr_to_cmi(par_corr):
    """Transformation of partial correlation to CMI scale.

    Using the (multivariate) Gaussian assumption.

    Args:
        par_corr (float): par_corr value.

    Returns:
        Rescaled value.
    """

    return -0.5 * numpy.log(1. - par_corr**2)


def myround(x, base=5, round_mode='updown'):
    """Rounds x to a float with precision base.

    :type x: float
    :param x: number

    :type base: float
    :param base: precision to be rounded to

    :rtype: float
    :returns: the rounded number.

    Args:
        x (float): Number.
        base (int, optional): Precision.
        round_mode (str, optional): 'updown', 'up', or 'down'.

    Returns:
        Rounded number.
    """
    if round_mode == 'updown':
        return base * round(float(x) / base)
    elif round_mode == 'down':
        return base * numpy.floor(float(x) / base)
    elif round_mode == 'up':
        return base * numpy.ceil(float(x) / base)

    return base * round(float(x) / base)


def make_nice_axes(ax, where=None, skip=2, color=None):
    """Makes nice axes.

    Args:
        ax (TYPE): Axis.
        where (None, optional): Specifies which axes to draw.
        skip (int, optional): Specifies the skipping of ticks.
        color (None, optional): Specifies the color of each axis.

    Raises:
        ValueError: Description
    """

    if where is None:
        where = ['left', 'bottom']
    if color is None:
        color = {'left': 'black', 'right': 'black',
                 'bottom': 'black', 'top': 'black'}

    if type(skip) == int:
        skip_x = skip_y = skip
    else:
        skip_x = skip[0]
        skip_y = skip[1]

    for loc, spine in ax.spines.iteritems():
        if loc in where:
            spine.set_position(('outward', 5))  # outward by 10 points
            spine.set_color(color[loc])
            if loc == 'left' or loc == 'right':
                pyplot.setp(ax.get_yticklines(), color=color[loc])
                pyplot.setp(ax.get_yticklabels(), color=color[loc])
            if loc == 'top' or loc == 'bottom':
                pyplot.setp(ax.get_xticklines(), color=color[loc])
        elif loc in [item for item in ['left', 'bottom', 'right', 'top']
                     if item not in where]:
            spine.set_color('none')  # don't draw spine

        else:
            raise ValueError('unknown spine location: %s' % loc)

    # ax.xaxis.get_major_formatter().set_useOffset(False)

    # turn off ticks where there is no spine
    if 'top' in where and 'bottom' not in where:
        ax.xaxis.set_ticks_position('top')
        ax.set_xticks(ax.get_xticks()[::skip_x])
    elif 'bottom' in where:
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(ax.get_xticks()[::skip_x])
    else:
        ax.xaxis.set_ticks_position('none')
        ax.xaxis.set_ticklabels([])
    if 'right' in where and 'left' not in where:
        ax.yaxis.set_ticks_position('right')
        ax.set_yticks(ax.get_yticks()[::skip_y])
    elif 'left' in where:
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks(ax.get_yticks()[::skip_y])
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])

    ax.patch.set_alpha(0.)


def get_absmax(lagfuncs):
    """Get value at absolute maximum in lag function array.

    For an (N, N, tau)-array this comutes the lag of the absolute maximum
    along the tau-axis and stores the (positive or negative) value in
    the (N,N)-array absmax.

    Args:
        lagfuncs (array): Array of lag functions for each pair of nodes
            (autodependencies on diagonal)

    Returns:
        Value at absolute maximum.
    """

    absmax_indices = numpy.abs(lagfuncs).argmax(axis=2)
    i, j = numpy.indices(lagfuncs.shape[:2])

    return lagfuncs[i, j, absmax_indices]


def add_timeseries(fig, axes, i, time, dataseries, label,
                   selector=False,
                   sample_selector=None,
                   grey_masked_samples=False,
                   data_linewidth=1.,
                   skip_ticks_data_x=1,
                   skip_ticks_data_y=1,
                   unit=None,
                   last=False,
                   time_label='',
                   label_fontsize=10,
                   color='black',
                   grey_alpha=1.,
                   ):
    """Adds a time series plot to an axis.

    Args:
        fig (figure instance): Figure instance.
        axes (axis instance): Either gridded axis object or single axis
            instance.
        i (int): Index of axis in gridded axis object.
        time (array): Timelabel array.
        dataseries (array): One-dimensional data array.
        label (str): Variable label.
        selector (bool, optional): Whether to use masked data.
        sample_selector (array, optional): Data mask where False labels masked
            samples.
        grey_masked_samples (bool or str, optional): Whether to mark masked
            samples by grey fills ('fill') or grey data ('data').
        data_linewidth (float, optional): Linewidth.
        skip_ticks_data_x (int, optional): ...
        skip_ticks_data_y (int, optional): ...
        unit (str, optional): Units of variable.
        last (bool, optional): Specifiy whether this is the last panel where
            also the bottom axis is plotted.
        time_label (str, optional): ...
        label_fontsize (int, optional): ...
        color (str, optional): ...
        grey_alpha (float, optional): Opacity of line.
    """

    # axes[i].xaxis.get_major_formatter().set_useOffset(False)
    try:
        ax = axes[i]
    except:
        ax = axes

    if selector:
        maskdata = numpy.ma.masked_where(sample_selector == 0, dataseries)

        if grey_masked_samples == 'fill':
            ax.fill_between(time, maskdata.min(), maskdata.max(),
                            where=sample_selector == 0, color='grey',
                            interpolate=True,
                            linewidth=0., alpha=grey_alpha)
        elif grey_masked_samples == 'data':
            ax.plot(time, dataseries,
                    color='grey', marker='.', markersize=data_linewidth,
                    linewidth=data_linewidth, clip_on=False,
                    alpha=grey_alpha)

        ax.plot(time, maskdata,
                color=color, linewidth=data_linewidth, marker='.',
                markersize=data_linewidth, clip_on=False)
    else:
        ax.plot(time, dataseries,
                color=color, linewidth=data_linewidth, clip_on=False)

    if last:
        make_nice_axes(ax, where=['left', 'bottom'], skip=(
            skip_ticks_data_x, skip_ticks_data_y))
        ax.set_xlabel(r'%s' % time_label, fontsize=label_fontsize)
    else:
        make_nice_axes(
            ax, where=['left'], skip=(skip_ticks_data_x, skip_ticks_data_y))
    # ax.get_xaxis().get_major_formatter().set_useOffset(False)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.label_outer()

    ax.set_xlim(time[0], time[-1])

    trans = transforms.blended_transform_factory(
        fig.transFigure, ax.transAxes)
    if unit:
        ax.text(.02, .5, r'%s [%s]' % (label, unit), fontsize=label_fontsize,
                horizontalalignment='left', verticalalignment='center',
                rotation=90, transform=trans)
    else:
        ax.text(.02, .5, r'%s' % (label), fontsize=label_fontsize,
                horizontalalignment='left', verticalalignment='center',
                rotation=90, transform=trans)


def plot_timeseries(data, datatime, var_names, save_name,
                    var_units=None,
                    time_label='years',
                    selector=False,
                    sample_selector=None,
                    grey_masked_samples=True,
                    data_linewidth=1.,
                    skip_ticks_data_x=1,
                    skip_ticks_data_y=1,
                    label_fontsize=8,
                    figsize=(3.375, 3.),
                    ):
    """Create and save figure of stacked panels with time series.

    Args:
        data (array, optional): Data array of shape (time, variables).
        datatime (array): Timelabel array.
        var_names (list): List of variable names.
        save_name (str): figure file name.
        var_units (list, optional): List of variable units.
        time_label (str, optional): ...
        selector (bool, optional): Whether to use masked data.
        sample_selector (bool array, optional): Data mask where False labels
            masked samples.
        grey_masked_samples (bool or str, optional): Whether to mark masked
            samples by grey fills ('fill') or grey data ('data').
        data_linewidth (float, optional): Linewidth.
        skip_ticks_data_x (int, optional): ...
        skip_ticks_data_y (int, optional): ...
        label_fontsize (int, optional): ...
        figsize (tuple, optional): Figure size.

    """

    T, N = data.shape

    if var_units is None:
        var_units = ['' for i in range(N)]

    fig, axes = pyplot.subplots(N, sharex=True, frameon=False, figsize=figsize)

    for i in range(N):
        add_timeseries(fig=fig, axes=axes, i=i,
                       time=datatime,
                       dataseries=data[:, i],
                       label=var_names[i],
                       selector=selector,
                       sample_selector=sample_selector[:, i],
                       grey_masked_samples=grey_masked_samples,
                       data_linewidth=data_linewidth,
                       skip_ticks_data_x=skip_ticks_data_x,
                       skip_ticks_data_y=skip_ticks_data_y,
                       unit=var_units[i],
                       last=(i == N - 1),
                       time_label=time_label,
                       label_fontsize=label_fontsize,
                       )

    fig.subplots_adjust(bottom=0.15, top=.9, left=0.15, right=.95, hspace=.3)
    fig.savefig(save_name)


class setup_matrix():
    """Create matrix of lag function panels."""

    def __init__(self, N, tau_max, var_names,
                 figsize=(3, 3),
                 minimum=-1,
                 maximum=1,
                 label_space_left=0.0,
                 label_space_top=.98,
                 legend_width=.15,
                 legend_fontsize=10,
                 x_base=1., y_base=0.4,
                 plot_gridlines=False,
                 lag_units='months',
                 label_fontsize=10):
        """Initialize matrix of panels."""

        self.tau_max = tau_max

        self.labels = []
        self.lag_units = lag_units
        self.legend_width = legend_width
        self.legend_fontsize = legend_fontsize

        self.label_space_left = label_space_left
        self.label_space_top = label_space_top
        self.label_fontsize = label_fontsize

        self.fig = pyplot.figure(figsize=figsize)

        self.axes_dict = {}

        plot_index = 1
        for i in range(N):
            for j in range(N):
                self.axes_dict[(i, j)] = self.fig.add_subplot(N, N, plot_index)
                # Plot process labels
                if j == 0:
                    trans = transforms.blended_transform_factory(
                        self.fig.transFigure, self.axes_dict[(i, j)].transAxes)
                    self.axes_dict[(i, j)].text(0.01, .5, '%s' %
                                                str(var_names[i]),
                                                fontsize=label_fontsize,
                                                horizontalalignment='left',
                                                verticalalignment='center',
                                                transform=trans)
                if i == 0:
                    trans = transforms.blended_transform_factory(
                        self.axes_dict[(i, j)].transAxes, self.fig.transFigure)
                    self.axes_dict[(i, j)].text(.5, .99, r'${\to}$ ' + '%s' %
                                                str(var_names[j]),
                                                fontsize=label_fontsize,
                                                horizontalalignment='center',
                                                verticalalignment='top',
                                                transform=trans)

                # Make nice axis
                make_nice_axes(
                    self.axes_dict[(i, j)], where=['left', 'bottom'],
                    skip=(1, 1))
                if x_base is not None:
                    self.axes_dict[(i, j)].xaxis.set_major_locator(
                        ticker.FixedLocator(numpy.arange(0, self.tau_max + 1,
                                                         x_base)))
                    if x_base / 2. % 1 == 0:
                        self.axes_dict[(i, j)].xaxis.set_minor_locator(
                            ticker.FixedLocator(numpy.arange(0, self.tau_max +
                                                             1,
                                                             x_base / 2.)))
                if y_base is not None:
                    self.axes_dict[(i, j)].yaxis.set_major_locator(
                        ticker.FixedLocator(
                            numpy.arange(myround(minimum, y_base, 'down'),
                                         myround(maximum, y_base, 'up') +
                                         y_base, y_base)))
                    self.axes_dict[(i, j)].yaxis.set_minor_locator(
                        ticker.FixedLocator(
                            numpy.arange(myround(minimum, y_base, 'down'),
                                         myround(maximum, y_base, 'up') +
                                         y_base, y_base / 2.)))

                    self.axes_dict[(i, j)].set_ylim(
                        myround(minimum, y_base, 'down'),
                        myround(maximum, y_base, 'up'))
                self.axes_dict[(i, j)].label_outer()
                self.axes_dict[(i, j)].set_xlim(0, self.tau_max)
                if plot_gridlines:
                    self.axes_dict[(i, j)].grid(True, which='major',
                                                color='black',
                                                linestyle='dotted',
                                                dashes=(1, 1),
                                                linewidth=.05,
                                                zorder=-5)

                plot_index += 1

    def add_lagfuncs(self, lagfuncs, sig_thres,
                     conf=None,
                     color='grey',
                     label=None,
                     two_sided_thres=True,
                     marker='.',
                     markersize=3,
                     alpha=1.,
                     plot_confidence=False,
                     conf_lev=.95,
                     rescale_cmi=False,
                     verbosity=0,
                     ):
        """Add lag function plot from lagfuncs array."""

        if rescale_cmi:
            print("Rescaling CMI (and sig_thres) to partial correlation "
                  "scale [0, 1]")

            lagfuncs = par_corr_trafo(lagfuncs)
            sig_thres = par_corr_trafo(sig_thres)
            if plot_confidence and conf is not None:
                conf = par_corr_trafo(conf)

        if label is not None:
            self.labels.append((label, color, marker, markersize, alpha))

        if verbosity > 0:
            links = numpy.where(numpy.abs(lagfuncs) > sig_thres)
            for (i, j, tau) in zip(links[0], links[1], links[2]):
                print("\tlink %d -> %d at lag %d with sig-thres %.4f"
                      " has weight %.4f" % (i, j, tau, sig_thres[i, j, tau],
                                            lagfuncs)),
                if plot_confidence:
                    print(" with %2.f%% conf in %s" % ((1. - (1. - conf_lev) *
                                                        2) * 100, conf[i, j,
                                                                       tau]))
                else:
                    print ''

        for ij in self.axes_dict.keys():
            i = ij[0]
            j = ij[1]
            maskedres = numpy.copy(lagfuncs[i, j, int(i == j):])
            self.axes_dict[(i, j)].plot(range(int(i == j), self.tau_max + 1),
                                        maskedres,
                                        linestyle='', color=color,
                                        marker=marker, markersize=markersize,
                                        alpha=alpha, clip_on=False)
            if plot_confidence:
                maskedconfres = numpy.copy(conf[i, j, int(i == j):])
                self.axes_dict[(i, j)].plot(range(int(i == j),
                                                  self.tau_max + 1),
                                            maskedconfres[:, 0],
                                            linestyle='', color=color,
                                            marker='_',
                                            markersize=markersize - 2,
                                            alpha=alpha, clip_on=False)
                self.axes_dict[(i, j)].plot(range(int(i == j),
                                                  self.tau_max + 1),
                                            maskedconfres[:, 1],
                                            linestyle='', color=color,
                                            marker='_',
                                            markersize=markersize - 2,
                                            alpha=alpha, clip_on=False)

            self.axes_dict[(i, j)].plot(range(int(i == j), self.tau_max + 1),
                                        numpy.zeros(self.tau_max + 1 -
                                                    int(i == j)),
                                        color='black', linestyle='dotted',
                                        linewidth=.1)

            maskedsigres = sig_thres[i, j, int(i == j):]

            self.axes_dict[(i, j)].plot(range(int(i == j), self.tau_max + 1),
                                        maskedsigres,
                                        color=color, linestyle='solid',
                                        linewidth=.1, alpha=alpha)
            if two_sided_thres:
                self.axes_dict[(i, j)].plot(range(int(i == j),
                                                  self.tau_max + 1),
                                            -sig_thres[i, j, int(i == j):],
                                            color=color, linestyle='solid',
                                            linewidth=.1, alpha=alpha)

    def savefig(self, name):
        """Save matrix figure."""

        # Trick to plot legend
        if len(self.labels) > 0:
            axlegend = self.fig.add_subplot(111, frameon=False)
            axlegend.spines['left'].set_color('none')
            axlegend.spines['right'].set_color('none')
            axlegend.spines['bottom'].set_color('none')
            axlegend.spines['top'].set_color('none')
            axlegend.set_xticks([])
            axlegend.set_yticks([])

            # self.labels.append((label, color, marker, markersize, alpha))
            for item in self.labels:

                label = item[0]
                color = item[1]
                marker = item[2]
                markersize = item[3]
                alpha = item[4]

                axlegend.plot([], [], linestyle='', color=color,
                              marker=marker, markersize=markersize,
                              label=label, alpha=alpha)
            axlegend.legend(loc='upper left', ncol=1,
                            bbox_to_anchor=(1.05, 0., .1, 1.),
                            borderaxespad=0, fontsize=self.legend_fontsize
                            ).draw_frame(False)

            self.fig.subplots_adjust(left=self.label_space_left, right=1. -
                                     self.legend_width,
                                     top=1. - self.label_space_top,
                                     hspace=0.35, wspace=0.35)
            pyplot.figtext(
                0.5, 0.01, r'lag $\tau$ [%s]' % self.lag_units,
                horizontalalignment='center', fontsize=self.label_fontsize)
        else:
            self.fig.subplots_adjust(
                left=self.label_space_left, right=.95,
                top=1. - self.label_space_top,
                hspace=0.35, wspace=0.35)
            pyplot.figtext(
                0.55, 0.01, r'lag $\tau$ [%s]' % self.lag_units,
                horizontalalignment='center', fontsize=self.label_fontsize)

        self.fig.savefig(name)


def draw_network_with_curved_edges(
    fig, ax,
    G, pos,
    node_rings,
    node_labels, node_label_size, node_alpha=1., standard_size=100,
    standard_cmap='OrRd', standard_color='grey', log_sizes=False,
    cmap_links='YlOrRd', cmap_links_edges='YlOrRd', links_vmin=0.,
    links_vmax=1., links_edges_vmin=0., links_edges_vmax=1.,
    links_ticks=.2, links_edges_ticks=.2, link_label_fontsize=8,
    arrowstyle='simple', arrowhead_size=3., curved_radius=.2, label_fontsize=4,
    label_fraction=.5, link_colorbar_label='link',
    link_edge_colorbar_label='link_edge',
    undirected_curved=False, undirected_style='solid',
):  # , params=None):
    """Function to draw a network from networkx graph instance.

    Various attributes are used to specify the graph's properties.

    This function is just a beta-template for now that can be further
    customized.
    """

    from matplotlib.patches import FancyArrowPatch, Circle

    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])

    N = len(G)

    def draw_edge(ax, u, v, d, seen, arrowstyle='simple', directed=True):

        n1 = G.node[u]['patch']
        n2 = G.node[v]['patch']

        if directed:
            rad = curved_radius
#            facecolor = d['directed_color']
#            edgecolor = d['directed_edgecolor']
            if cmap_links is not None:
                facecolor = data_to_rgb_links.to_rgba(d['directed_color'])
            else:
                if d['directed_color'] is not None:
                    facecolor = d['directed_color']
                else:
                    facecolor = standard_color
            if d['directed_edgecolor'] is not None:
                edgecolor = data_to_rgb_links_edges.to_rgba(
                    d['directed_edgecolor'])
            else:
                if d['directed_edgecolor'] is not None:
                    edgecolor = d['directed_edgecolor']
                else:
                    edgecolor = standard_color
            width = d['directed_width']
            alpha = d['directed_alpha']
            if (u, v) in seen:
                rad = seen.get((u, v))
                rad = (rad + numpy.sign(rad) * 0.1) * -1.
            arrowstyle = arrowstyle
            link_edge = d['directed_edge']
            linestyle = 'solid'
        else:
            rad = undirected_curved * curved_radius
            if cmap_links is not None:
                facecolor = data_to_rgb_links.to_rgba(d['undirected_color'])
            else:
                if d['undirected_color'] is not None:
                    facecolor = d['undirected_color']
                else:
                    facecolor = standard_color
            if d['undirected_edgecolor'] is not None:
                edgecolor = data_to_rgb_links_edges.to_rgba(
                    d['undirected_edgecolor'])
            else:
                if d['undirected_edgecolor'] is not None:
                    edgecolor = d['undirected_edgecolor']
                else:
                    edgecolor = standard_color
            width = d['undirected_width']
            alpha = d['undirected_alpha']
            arrowstyle = '-'
            link_edge = d['undirected_edge']
            linestyle = undirected_style

#        print u,v, link_edge
        if link_edge:
            # Outer arrow
            e = FancyArrowPatch(n1.center, n2.center,  # patchA=n1,patchB=n2,
                                arrowstyle=arrowstyle,
                                connectionstyle='arc3,rad=%s' % rad,
                                mutation_scale=3 * width,
                                lw=0.,
                                alpha=alpha,
                                color=edgecolor,
                                #                                zorder = -1,
                                clip_on=False,
                                patchA=n1, patchB=n2)

            ax.add_patch(e)
        # Inner arrow
        e = FancyArrowPatch(n1.center, n2.center,  # patchA=n1,patchB=n2,
                            arrowstyle=arrowstyle,
                            connectionstyle='arc3,rad=%s' % rad,
                            mutation_scale=width,
                            lw=0.,
                            alpha=alpha,
                            color=facecolor,
                            clip_on=False,
                            patchA=n1, patchB=n2)
        ax.add_patch(e)

        if d['label'] is not None and directed:
            # Attach labels of lags
            trans = None  # patch.get_transform()
            path = e.get_path()
            verts = path.to_polygons(trans)[0]
            if len(verts) > 2:
                label_vert = verts[1, :]
                l = d['label']
                string = str(l)
                ax.text(label_vert[0], label_vert[1], string,
                        fontsize=link_label_fontsize,
                        verticalalignment='center',
                        horizontalalignment='center')      

        return rad

    # Fix lower left and upper right corner (networkx unfortunately rescales
    # the positions...)
    # c = Circle((0, 0), radius=.01, alpha=1., fill=False,
    #            linewidth=0., transform=fig.transFigure)
    # ax.add_patch(c)
    # c = Circle((1, 1), radius=.01, alpha=1., fill=False,
    #            linewidth=0., transform=fig.transFigure)
    # ax.add_patch(c)

    ##
    # Draw nodes
    ##
    node_sizes = numpy.zeros((len(node_rings.keys()), N))
    for ring in node_rings.keys():  # iterate through to get all node sizes
        if node_rings[ring]['sizes'] is not None:
            node_sizes[ring] = node_rings[ring]['sizes']
        else:
            node_sizes[ring] = standard_size

    max_sizes = node_sizes.max(axis=1)
    total_max_size = node_sizes.sum(axis=0).max()
    node_sizes /= total_max_size
    node_sizes *= standard_size
#    print  'node_sizes ', node_sizes

    # start drawing the outer ring first...
    for ring in node_rings.keys()[::-1]:
        #        print ring
        # dictionary of rings: {0:{'sizes':(N,)-array, 'color_array':(N,)-array
        # or None, 'cmap':string, 'vmin':float or None, 'vmax':float or None}}
        if node_rings[ring]['color_array'] is not None:
            color_data = node_rings[ring]['color_array']
            if node_rings[ring]['vmin'] is not None:
                vmin = node_rings[ring]['vmin']
            else:
                vmin = node_rings[ring]['color_array'].min()
            if node_rings[ring]['vmax'] is not None:
                vmax = node_rings[ring]['vmax']
            else:
                vmax = node_rings[ring]['color_array'].max()
            if node_rings[ring]['cmap'] is not None:
                cmap = node_rings[ring]['cmap']
            else:
                cmap = standard_cmap
            data_to_rgb = pyplot.cm.ScalarMappable(
                norm=None, cmap=pyplot.get_cmap(cmap))
            data_to_rgb.set_array(color_data)
            data_to_rgb.set_clim(vmin=vmin, vmax=vmax)
            colors = [data_to_rgb.to_rgba(color_data[n]) for n in G]

            if node_rings[ring]['colorbar']:
                # Create colorbars for nodes
                # cax_n = pyplot.axes([.8 + ring*0.11,
                # ax.figbox.bounds[1]+0.05, 0.025, 0.35], frameon=False) #
                # setup colorbar axes.
                # setup colorbar axes.
                cax_n = pyplot.axes([0.05, ax.figbox.bounds[1] + 0.02 +
                                     ring * 0.11,
                                     0.4, 0.025 +
                                     (len(node_rings.keys()) == 1) * 0.035],
                                    frameon=False)
                cb_n = pyplot.colorbar(
                    data_to_rgb, cax=cax_n, orientation='horizontal')
                try:
                    cb_n.set_ticks(numpy.arange(myround(vmin,
                                node_rings[ring]['ticks'], 'down'), myround(
                        vmax, node_rings[ring]['ticks'], 'up') +
                        node_rings[ring]['ticks'], node_rings[ring]['ticks']))
                except:
                    print 'no ticks given'
                cb_n.outline.remove()
                # cb_n.set_ticks()
                cax_n.set_xlabel(
                    node_rings[ring]['label'], labelpad=1,
                    fontsize=label_fontsize)
        else:
            colors = None
            vmin = None
            vmax = None

        for n in G:
            # if n==1: print node_sizes[:ring+1].sum(axis=0)[n]

            if type(node_alpha) == dict:
                alpha = node_alpha[n]
            else:
                alpha = 1.

            if colors is None:
                ax.scatter(pos[n][0], pos[n][1],
                           s=node_sizes[:ring + 1].sum(axis=0)[n] ** 2,
                           facecolors=standard_color,
                           edgecolors=standard_color, alpha=alpha,
                           clip_on=False, linewidth=.1, zorder=-ring)
            else:
                ax.scatter(pos[n][0], pos[n][1],
                           s=node_sizes[:ring + 1].sum(axis=0)[n] ** 2,
                           facecolors=colors[n], edgecolors='white',
                           alpha=alpha,
                           clip_on=False, linewidth=.1, zorder=-ring)

            if ring == 0:
                ax.text(pos[n][0], pos[n][1], node_labels[n],
                        fontsize=node_label_size,
                        horizontalalignment='center',
                        verticalalignment='center', alpha=alpha)

        if node_rings[ring]['sizes'] is not None:
            # Draw reference node as legend
            ax.scatter(0., 0., s=node_sizes[:ring + 1].sum(axis=0).max() ** 2,
                       alpha=1., facecolors='none', edgecolors='grey',
                       clip_on=False, linewidth=.1, zorder=-ring)

            if log_sizes:
                ax.text(0., 0., '         ' * ring + '%.2f' %
                        (numpy.exp(max_sizes[ring]) - 1.),
                        fontsize=node_label_size,
                        horizontalalignment='left', verticalalignment='center')
            else:
                ax.text(0., 0., '         ' * ring + '%.2f' % max_sizes[ring],
                        fontsize=node_label_size,
                        horizontalalignment='left', verticalalignment='center')

    ##
    # Draw edges of different types
    ##
    # First draw small circles as anchorpoints of the curved edges
    for n in G:
        # , transform = ax.transAxes)
        size = standard_size*.3
        c = Circle(pos[n], radius=size, alpha=0., fill=False, linewidth=0.)
        ax.add_patch(c)
        G.node[n]['patch'] = c

    # Collect all edge weights to get color scale
    all_links_weights = []
    all_links_edge_weights = []
    for (u, v, d) in G.edges(data=True):
        if u != v:
            if d['directed'] and d['directed_color'] is not None:
                all_links_weights.append(d['directed_color'])
            if d['undirected'] and d['undirected_color'] is not None:
                all_links_weights.append(d['undirected_color'])
            if d['directed_edge'] and d['directed_edgecolor'] is not None:
                all_links_edge_weights.append(d['directed_edgecolor'])
            if d['undirected_edge'] and d['undirected_edgecolor'] is not None:
                all_links_edge_weights.append(d['undirected_edgecolor'])

    if cmap_links is not None and len(all_links_weights) > 0:
        if links_vmin is None:
            links_vmin = numpy.array(all_links_weights).min()
        if links_vmax is None:
            links_vmax = numpy.array(all_links_weights).max()
        data_to_rgb_links = pyplot.cm.ScalarMappable(
            norm=None, cmap=pyplot.get_cmap(cmap_links))
        data_to_rgb_links.set_array(numpy.array(all_links_weights))
        data_to_rgb_links.set_clim(vmin=links_vmin, vmax=links_vmax)
        # Create colorbars for links
# cax_e = pyplot.axes([.8, ax.figbox.bounds[1]+0.5, 0.025, 0.35],
# frameon=False) # setup colorbar axes.
        # setup colorbar axes.
        cax_e = pyplot.axes([0.55, ax.figbox.bounds[1] + 0.02, 0.4, 0.025 +
                             (len(all_links_edge_weights) == 0) * 0.035],
                            frameon=False)

        cb_e = pyplot.colorbar(
            data_to_rgb_links, cax=cax_e, orientation='horizontal')
        try:
            cb_e.set_ticks(numpy.arange(myround(links_vmin, links_ticks,
                                                'down'),
                                     myround(links_vmax, links_ticks, 'up') +
                                     links_ticks, links_ticks))
        except:
            print 'no ticks given'

        cb_e.outline.remove()
        # cb_n.set_ticks()
        cax_e.set_xlabel(
            link_colorbar_label, labelpad=1, fontsize=label_fontsize)

    if cmap_links_edges is not None and len(all_links_edge_weights) > 0:
        if links_edges_vmin is None:
            links_edges_vmin = numpy.array(all_links_edge_weights).min()
        if links_edges_vmax is None:
            links_edges_vmax = numpy.array(all_links_edge_weights).max()
        data_to_rgb_links_edges = pyplot.cm.ScalarMappable(
            norm=None, cmap=pyplot.get_cmap(cmap_links_edges))
        data_to_rgb_links_edges.set_array(numpy.array(all_links_edge_weights))
        data_to_rgb_links_edges.set_clim(
            vmin=links_edges_vmin, vmax=links_edges_vmax)

        # Create colorbars for link edges
# cax_e = pyplot.axes([.8+.1, ax.figbox.bounds[1]+0.5, 0.025, 0.35],
# frameon=False) # setup colorbar axes.
        # setup colorbar axes.
        cax_e = pyplot.axes(
            [0.55, ax.figbox.bounds[1] + 0.05 + 0.1, 0.4, 0.025],
            frameon=False)

        cb_e = pyplot.colorbar(
            data_to_rgb_links_edges, cax=cax_e, orientation='horizontal')
        try:
            cb_e.set_ticks(numpy.arange(myround(links_edges_vmin,
                                                links_edges_ticks, 'down'),
                                        myround(links_edges_vmax,
                                                links_edges_ticks, 'up') +
                                        links_edges_ticks,
                                        links_edges_ticks))
        except:
            print 'no ticks given'
        cb_e.outline.remove()
        # cb_n.set_ticks()
        cax_e.set_xlabel(
            link_edge_colorbar_label, labelpad=1, fontsize=label_fontsize)

    # Draw edges
    seen = {}
    for (u, v, d) in G.edges(data=True):
        if u != v:
            if d['directed']:
                seen[(u, v)] = draw_edge(ax, u, v, d, seen, arrowstyle, directed=True)
            if d['undirected'] and (v, u) not in seen:
                seen[(u, v)] = draw_edge(ax, u, v, d, seen, directed=False)


def plot_graph(lagfuncs, sig_thres, var_names, fig, ax,
               link_colorbar_label='',
               node_colorbar_label='',
               rescale_cmi=False,
               link_width=None,
               node_pos=None,
               undirected_style='solid',
               arrow_linewidth=30.,
               vmin_edges=-1,
               vmax_edges=1.,
               edge_ticks=.4,
               cmap_edges='RdBu_r',
               vmin_nodes=0,
               vmax_nodes=1.,
               node_ticks=.4,
               cmap_nodes='OrRd',
               node_size=20,
               arrowhead_size=20,
               curved_radius=.2,
               label_fontsize=10,
               label_fraction=0.5,
               alpha=1.,
               node_label_size=10,
               link_label_fontsize=6,
               ):
    """Creates and saves a network plot from lag function and sig_thres array.

    This is still in beta, hopefully the arguments are self-explanatory,
    otherwise please look at the code.
    """
    import networkx
    if rescale_cmi:
        print("Rescaling CMI (and sig_thres) to partial correlation "
              "scale [0, 1]")
        lagfuncs = par_corr_trafo(lagfuncs)
        sig_thres = par_corr_trafo(sig_thres)

    if link_width is not None and not numpy.all(link_width >= 0.):
        raise ValueError("link_width must be non-negative")

    N, N, dummy = lagfuncs.shape
    tau_max = dummy - 1

    # Define graph links by absolute maximum (positive or negative like for
    # partial correlation)
    # lagfuncs[numpy.abs(lagfuncs) < sig_thres] = 0.

    net = get_absmax(lagfuncs * (numpy.abs(lagfuncs) >= sig_thres))
    G = networkx.DiGraph(net)

    node_color = numpy.zeros(N)
    # list of all strengths for color map
    all_strengths = []
    # Add attributes, contemporaneous and directed links are handled separately
    for (u, v, dic) in G.edges(data=True):
        # average lagfunc for link u --> v ANDOR u -- v
        if tau_max > 0:
            # argmax of absolute maximum
            argmax = numpy.abs(lagfuncs[u, v][1:]).argmax() + 1
        else:
            argmax = 0
        if u != v:
            # For contemp links masking or finite samples can lead to different
            # values for u--v and v--u
            # Here we use the  maximum for the width and weight (=color)
            # of the link
            # Draw link if u--v OR v--u at lag 0 is nonzero
            dic['undirected'] = ((numpy.abs(lagfuncs[u, v][0]) >=
                                  sig_thres[u, v][0]) or
                                 (numpy.abs(lagfuncs[v, u][0]) >=
                                  sig_thres[v, u][0]))
            dic['undirected_alpha'] = alpha
            # value at argmax of average
            if numpy.abs(lagfuncs[u, v][0] - lagfuncs[v, u][0]) > .0001:
                print("Contemporaneous I(%d; %d)=%.3f != I(%d; %d)=%.3f" % (
                      u, v, lagfuncs[u, v][0], v, u, lagfuncs[v, u][0]) +
                      " due to conditions, finite sample effects or "
                      "masking, here edge color = "
                      "larger (absolute) value.")
            dic['undirected_color'] = get_absmax(
                numpy.array([[[lagfuncs[u, v][0],
                               lagfuncs[v, u][0]]]])).squeeze()
            if link_width is None:
                dic['undirected_width'] = arrow_linewidth
            else:
                dic['undirected_width'] = link_width[
                    u, v, 0] / link_width.max() * arrow_linewidth

            all_strengths.append(dic['undirected_color'])

            if tau_max > 0:
                # True if ensemble mean at lags > 0 is nonzero
                dic['directed'] = numpy.any(
                    numpy.abs(lagfuncs[u, v][1:]) >= sig_thres[u, v][1:])
            else:
                dic['directed'] = False
            dic['directed_alpha'] = alpha
            if link_width is None:
                # fraction of nonzero values
                dic['directed_width'] = arrow_linewidth
            else:
                dic['directed_width'] = link_width[
                    u, v, argmax] / link_width.max() * arrow_linewidth

            # value at argmax of average
            dic['directed_color'] = lagfuncs[u, v][argmax]
            all_strengths.append(dic['directed_color'])

            # Sorted list of significant lags (only if robust wrt
            # d['min_ensemble_frac'])
            if tau_max > 0:
                lags = numpy.abs(lagfuncs[u, v][1:]).argsort()[::-1] + 1
                sig_lags = (numpy.where(numpy.abs(lagfuncs[u, v][
                            1:]) >= sig_thres[u, v][1:])[0] + 1).tolist()
            else:
                lags, sig_lags = [], []
            dic['label'] = str([l for l in lags if l in sig_lags])[1:-1]
        else:
            # Node color is max of average autodependency
            node_color[u] = lagfuncs[u, v][argmax]

        dic['directed_edge'] = False
        dic['directed_edgecolor'] = None
        dic['undirected_edge'] = False
        dic['undirected_edgecolor'] = None

    # If no links are present, set value to zero
    if len(all_strengths) == 0:
        all_strengths = [0.]

    if node_pos is None:
        pos = networkx.circular_layout(deepcopy(G))
#            pos = networkx.spring_layout(deepcopy(G))
    else:
        pos = {}
        for i in range(N):
            pos[i] = (node_pos['x'][i], node_pos['y'][i])

    node_rings = {0: {'sizes': None, 'color_array': node_color,
                      'cmap': cmap_nodes, 'vmin': vmin_nodes,
                      'vmax': vmax_nodes, 'ticks': node_ticks,
                      'label': node_colorbar_label, 'colorbar': True,
                      }
                  }

    draw_network_with_curved_edges(
        fig=fig, ax=ax,
        G=deepcopy(G), pos=pos,
        # dictionary of rings: {0:{'sizes':(N,)-array, 'color_array':(N,)-array
        # or None, 'cmap':string,
        node_rings=node_rings,
        # 'vmin':float or None, 'vmax':float or None, 'label':string or None}}
        node_labels=var_names, node_label_size=node_label_size,
        node_alpha=alpha, standard_size=node_size,
        standard_cmap='OrRd', standard_color='orange',
        log_sizes=False,
        cmap_links=cmap_edges, links_vmin=vmin_edges,
        links_vmax=vmax_edges, links_ticks=edge_ticks,

        cmap_links_edges='YlOrRd', links_edges_vmin=-1., links_edges_vmax=1.,
        links_edges_ticks=.2, link_edge_colorbar_label='link_edge',

        arrowstyle='simple', arrowhead_size=arrowhead_size,
        curved_radius=curved_radius, label_fontsize=label_fontsize,
        link_label_fontsize=link_label_fontsize, label_fraction=label_fraction,
        link_colorbar_label=link_colorbar_label,
        undirected_style=undirected_style)

    # fig.subplots_adjust(left=0.1, right=.9, bottom=.25, top=.95)
    # savestring = os.path.expanduser(save_name)
    # pyplot.savefig(savestring)


def plot_time_series_graph(lagfuncs, sig_thres, var_names, fig, ax,
                           link_colorbar_label='',
                           rescale_cmi=False,
                           link_width=None,
                           arrow_linewidth=20.,
                           vmin_edges=-1,
                           vmax_edges=1.,
                           edge_ticks=.4,
                           cmap_edges='RdBu_r',
                           order=None,
                           node_size=10,
                           arrowhead_size=20,
                           curved_radius=.2,
                           label_fontsize=10,
                           alpha=1.,
                           node_label_size=10,
                           # link_label_fontsize=8,
                           label_indent_left=0.,
                           label_indent_top=.95,
                           undirected_style='solid',
                           ):
    """Creates and saves plot of time series graph.

    Based on lag function and sig_thres array. This function needs docstrings!
    """

    import networkx

    if rescale_cmi:
        print("Rescaling CMI (and sig_thres) to partial correlation "
              "scale [0, 1]")
        lagfuncs = par_corr_trafo(lagfuncs)
        sig_thres = par_corr_trafo(sig_thres)

    if link_width is not None and not numpy.all(link_width >= 0.):
        raise ValueError("link_width must be non-negative")

    N, N, dummy = lagfuncs.shape
    tau_max = dummy - 1
    max_lag = tau_max + 1

    if order is None:
        order = range(N)

    if set(order) != set(range(N)):
        raise ValueError("order must be a permutation of range(N)")

    def translate(row, lag):
        return row * max_lag + lag

    # Define graph links by absolute maximum (positive or negative like for
    # partial correlation)
    tsg = numpy.zeros((N * max_lag, N * max_lag))
    tsg_attr = numpy.zeros((N * max_lag, N * max_lag))

    for i, j, tau in numpy.column_stack(numpy.where(numpy.abs(lagfuncs) >=
                                                    sig_thres)):
        #                    print '\n',i, j, tau
        #                    print numpy.where(nonmasked[:,j])[0]

        for t in range(max_lag):
            if (0 <= translate(i, t - tau) and
                translate(i, t - tau) % max_lag <= translate(j, t) % max_lag):
                # print translate(i, t-tau), translate(j, t), lagfuncs[i,j,tau]
                tsg[translate(i, t - tau), translate(j, t)
                    ] = lagfuncs[i, j, tau]
                tsg_attr[translate(i, t - tau), translate(j, t)
                         ] = lagfuncs[i, j, tau]

    G = networkx.DiGraph(tsg)

    # node_color = numpy.zeros(N)
    # list of all strengths for color map
    all_strengths = []
    # Add attributes, contemporaneous and directed links are handled separately
    for (u, v, dic) in G.edges(data=True):

        if u != v:

            if u % max_lag == v % max_lag:
                dic['undirected'] = True
                dic['directed'] = False
            else:
                dic['undirected'] = False
                dic['directed'] = True

            dic['undirected_alpha'] = alpha
            dic['undirected_color'] = get_absmax(
                numpy.array([[[tsg_attr[u, v],
                               tsg_attr[v, u]]]])
            ).squeeze()
            dic['undirected_width'] = arrow_linewidth
            all_strengths.append(dic['undirected_color'])

            dic['directed_alpha'] = alpha

            dic['directed_width'] = arrow_linewidth

            # value at argmax of average
            dic['directed_color'] = tsg_attr[u, v]
            all_strengths.append(dic['directed_color'])

            dic['label'] = None

        dic['directed_edge'] = False
        dic['directed_edgecolor'] = None
        dic['undirected_edge'] = False
        dic['undirected_edgecolor'] = None

    # If no links are present, set value to zero
    if len(all_strengths) == 0:
        all_strengths = [0.]

    posarray = numpy.zeros((N * max_lag, 2))
    for i in xrange(N * max_lag):

        posarray[i] = numpy.array([(i % max_lag), (1. - i / max_lag)])

    pos_tmp = {}
    for i in range(N * max_lag):
        # for n in range(N):
        #     for tau in range(max_lag):
        #         i = n*N + tau
        pos_tmp[i] = numpy.array([((i % max_lag) - posarray.min(axis=0)[0]) /
                                  (posarray.max(axis=0)[0] -
                                   posarray.min(axis=0)[0]),
                                  ((1. - i / max_lag) -
                                   posarray.min(axis=0)[1]) /
                                  (posarray.max(axis=0)[1] -
                                   posarray.min(axis=0)[1])])
#                    print pos[i]
    pos = {}
    for n in range(N):
        for tau in range(max_lag):
            pos[n * max_lag + tau] = pos_tmp[order[n] * max_lag + tau]

    node_rings = {0: {'sizes': None, 'color_array': None,
                      'label': '', 'colorbar': False,
                      }
                  }

    # ] for v in range(max_lag)]
    node_labels = ['' for i in range(N * max_lag)]

    draw_network_with_curved_edges(
        fig=fig, ax=ax,
        G=deepcopy(G), pos=pos,
        # dictionary of rings: {0:{'sizes':(N,)-array, 'color_array':(N,)-array
        # or None, 'cmap':string,
        node_rings=node_rings,
        # 'vmin':float or None, 'vmax':float or None, 'label':string or None}}
        node_labels=node_labels, node_label_size=node_label_size,
        node_alpha=alpha, standard_size=node_size,
        standard_cmap='OrRd', standard_color='grey',
        log_sizes=False,
        cmap_links=cmap_edges, links_vmin=vmin_edges,
        links_vmax=vmax_edges, links_ticks=edge_ticks,

        cmap_links_edges='YlOrRd', links_edges_vmin=-1., links_edges_vmax=1.,
        links_edges_ticks=.2, link_edge_colorbar_label='link_edge',

        arrowstyle='simple', arrowhead_size=arrowhead_size,
        curved_radius=curved_radius, label_fontsize=label_fontsize,
        label_fraction=.5,
        link_colorbar_label=link_colorbar_label, undirected_curved=True,
        undirected_style=undirected_style)

    for i in range(N):
        trans = transforms.blended_transform_factory(
            fig.transFigure, ax.transData)
        ax.text(label_indent_left, pos[order[i] * max_lag][1],
                '%s' % str(var_names[order[i]]), fontsize=label_fontsize,
                horizontalalignment='left', verticalalignment='center',
                transform=trans)

    for tau in numpy.arange(max_lag - 1, -1, -1):
        trans = transforms.blended_transform_factory(
            ax.transData, fig.transFigure)
        if tau == max_lag - 1:
            ax.text(pos[tau][0], label_indent_top, r'$t$',
                    fontsize=label_fontsize,
                    horizontalalignment='center',
                    verticalalignment='top', transform=trans)
        else:
            ax.text(pos[tau][0], label_indent_top,
                    r'$t-%s$' % str(max_lag - tau - 1),
                    fontsize=label_fontsize,
                    horizontalalignment='center', verticalalignment='top',
                    transform=trans)

    # fig.subplots_adjust(left=0.1, right=.98, bottom=.25, top=.9)
    # savestring = os.path.expanduser(save_name)
    # pyplot.savefig(savestring)
