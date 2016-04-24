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

import numpy
import tigramite_estimation_beta as te
import tigramite_preprocessing as pp

import nose
import nose.tools as nt


def assert_graphs_equal(actual, expected):
    for j in expected.keys():
        nt.assert_items_equal(actual[j], expected[j])


def _get_parent_graph(nodes, exclude=None):
    graph = {}
    for j in nodes.keys():
        graph[j] = []
        for var, lag in nodes[j]:
            if lag != 0 and (var, lag) != exclude:
                graph[j].append((var, lag))

    return graph


def _get_neighbor_graph(nodes, exclude=None):

    graph = {}
    for j in nodes.keys():
        graph[j] = []
        for var, lag in nodes[j]:
            if lag == 0 and (var, lag) != exclude:
                graph[j].append((var, lag))

    return graph


def cmi2parcorr_trafo(cmi):
    return numpy.sqrt(1. - numpy.exp(-2. * cmi))

verbosity = 0


#
#  Start
#

def test_pc_algo_all():
    print("\nTesting function 'pc_algo_all' to check whether PC algorithm "
          "returns the correct set of parents.")

    # Test data:
    T = 1000
    numpy.random.seed(42)
    # Graph as in Runge PRE 2012 Fig. 1
    cxz = .5
    ax = .5
    cxy = .5
    cwy = .5
    links_coeffs = {0: [((1, -1), cxz)],
                    1: [((1, -1), ax)],
                    2: [((1, -2), cxy), ((3, -1), cwy)],
                    3: [],
                    }

    fulldata, true_parents_neighbors = pp.var_process(links_coeffs, T=T)
    T, N = fulldata.shape

    max_trials = 5
    measure_params = {'knn': 100}
    significance = 'analytic'

    print("Checking different initial_conds...")
    for initial_conds in range(1, 4):
        # print("%d" % initial_conds)

        parents_neighbors = te.pc_algo_all(
            fulldata, estimate_parents_neighbors='both',
            tau_min=0, tau_max=4,
            initial_conds=initial_conds, max_conds=6, max_trials=max_trials,
            measure='par_corr',
            significance=significance, sig_lev=0.999995, sig_samples=100,
            fixed_thres=0.015,
            measure_params=measure_params,
            selector=False, selector_type=['y'],
            sample_selector=numpy.array([0]),
            initial_parents_neighbors=None,
            verbosity=verbosity)
        assert_graphs_equal(parents_neighbors, true_parents_neighbors)

    print("Checking initial_parents_neighbors...")
    parents_neighbors = te.pc_algo_all(
        fulldata, estimate_parents_neighbors='both',
        tau_min=0, tau_max=4,
        initial_conds=1, max_conds=6, max_trials=max_trials,
        measure='par_corr',
        significance=significance, sig_lev=0.995, sig_samples=100,
        fixed_thres=0.015,
        measure_params=measure_params,
        selector=False, selector_type=['y'], sample_selector=numpy.array([0]),
        initial_parents_neighbors=true_parents_neighbors,
        verbosity=verbosity)
    assert_graphs_equal(parents_neighbors, true_parents_neighbors)

    print("Checking estimate_parents_neighbors...")
    for estimate_parents_neighbors in ['parents', 'both']:
        # print("%s" % estimate_parents_neighbors)
        parents_neighbors = te.pc_algo_all(
            fulldata,
            estimate_parents_neighbors=estimate_parents_neighbors,
            tau_min=0, tau_max=4,
            initial_conds=1, max_conds=6, max_trials=max_trials,
            measure='par_corr',
            significance='analytic', sig_lev=0.995, sig_samples=100,
            fixed_thres=0.015,
            measure_params=measure_params,
            selector=False, selector_type=['y'],
            sample_selector=numpy.array([0]),
            initial_parents_neighbors=None,
            verbosity=verbosity)
        if estimate_parents_neighbors == 'both':
            assert_graphs_equal(parents_neighbors, true_parents_neighbors)
        elif estimate_parents_neighbors == 'parents':
            assert_graphs_equal(_get_parent_graph(parents_neighbors),
                                _get_parent_graph(true_parents_neighbors))


def test_construct_array():
    print("\nTesting function '_construct_array' to check whether array is "
          "correctly constructed from data.")

    data = numpy.array([[0, 10, 20, 30],
                        [1, 11, 21, 31],
                        [2, 12, 22, 32],
                        [3, 13, 23, 33],
                        [4, 14, 24, 34]])
    sample_selector = numpy.array([[1, 0, 0, 1],
                                   [1, 1, 1, 1],
                                   [0, 1, 1, 1],
                                   [1, 1, 0, 0],
                                   [1, 1, 1, 1]], dtype='bool')

    X = [(1, -1)]
    Y = [(0, 0)]
    Z = [(0, -1), (1, -2), (2, 0)]

    tau_max = 2

    # No masking
    res = te._construct_array(
        X=X, Y=Y, Z=Z,
        tau_max=tau_max,
        selector=False,
        data=data,
        sample_selector=sample_selector,
        selector_type=None, verbosity=verbosity)
    numpy.testing.assert_almost_equal(res[0],
                                      numpy.array([[11., 12., 13.],
                                                   [2., 3., 4.],
                                                   [1., 2., 3.],
                                                   [10., 11., 12.],
                                                   [22., 23., 24.]]))
    numpy.testing.assert_almost_equal(res[1], numpy.array([0, 1, 2, 2, 2]))

    # masking y
    res = te._construct_array(
        X=X, Y=Y, Z=Z,
        tau_max=tau_max,
        selector=True,
        data=data,
        sample_selector=sample_selector,
        selector_type=['y'], verbosity=verbosity)

    numpy.testing.assert_almost_equal(res[0],
                                      numpy.array([[12., 13.],
                                                   [3., 4.],
                                                   [2., 3.],
                                                   [11., 12.],
                                                   [23., 24.]]))

    numpy.testing.assert_almost_equal(res[1], numpy.array([0, 1, 2, 2, 2]))

    # masking all
    res = te._construct_array(
        X=X, Y=Y, Z=Z,
        tau_max=tau_max,
        selector=True,
        data=data,
        sample_selector=sample_selector,
        selector_type=['x', 'y', 'z'], verbosity=verbosity)

    numpy.testing.assert_almost_equal(res[0],
                                      numpy.array([[13.],
                                                   [4.],
                                                   [3.],
                                                   [12.],
                                                   [24.]]))

    numpy.testing.assert_almost_equal(res[1], numpy.array([0, 1, 2, 2, 2]))


def test_get_lagfunctions():

    print("\nTesting function 'get_lagfunctions' to check whether conditioning"
          " measures MI, ITY, ITX, and MIT work as expected.")

    # Graph as in Runge PRE 2012 Fig. 1
    cxz = .5
    ax = .5
    cxy = .5
    cwy = .5
    links_coeffs = {0: [((1, -1), cxz)],
                    1: [((1, -1), ax)],
                    2: [((1, -2), cxy), ((3, -1), cwy)],
                    3: [],
                    }
    numpy.random.seed(42)
    data, links = pp.var_process(links_coeffs, T=10000, verbosity=verbosity)
    T, N = data.shape

    i = 1
    j = 2
    tau = 2

    # Test that cross correlation is correct
    cond_mode = 'none'
    expected_cmi = 0.5 * numpy.log(1. + (cxy**2 * 1.**2) / (1. - ax**2) /
                                   (cwy**2 * 1.**2 + 1.**2))
    expected_parcorr = cmi2parcorr_trafo(expected_cmi)
    res = te.get_lagfunctions(data, parents_neighbors=links,
                              cond_mode=cond_mode,
                              measure='par_corr',
                              tau_max=2, verbosity=verbosity)
    numpy.testing.assert_allclose(res[0][i, j, tau], expected_parcorr,
                                  rtol=0.1)

    # Test that ITY is correct
    cond_mode = 'parents_y'
    expected_cmi = 0.5 * numpy.log(1. + (cxy**2 * 1.**2) / (1. - ax**2))
    expected_parcorr = cmi2parcorr_trafo(expected_cmi)
    res = te.get_lagfunctions(data, parents_neighbors=links,
                              cond_mode=cond_mode,
                              measure='par_corr',
                              tau_max=2, verbosity=verbosity)
    numpy.testing.assert_allclose(res[0][i, j, tau], expected_parcorr,
                                  rtol=0.1)

    # Test that ITX is correct
    cond_mode = 'parents_x'
    expected_cmi = 0.5 * numpy.log(1. + (cxy**2 * 1.**2) /
                                   (cwy**2 * 1.**2 + 1.**2))
    expected_parcorr = cmi2parcorr_trafo(expected_cmi)
    res = te.get_lagfunctions(data, parents_neighbors=links,
                              cond_mode=cond_mode,
                              measure='par_corr',
                              tau_max=2, verbosity=verbosity)
    numpy.testing.assert_allclose(res[0][i, j, tau], expected_parcorr,
                                  rtol=0.1)

    # Test that MIT is correct
    cond_mode = 'parents_xy'
    expected_cmi = 0.5 * numpy.log(1. + (cxy**2 * 1.**2) / (1.**2))
    expected_parcorr = cmi2parcorr_trafo(expected_cmi)
    res = te.get_lagfunctions(data, parents_neighbors=links,
                              cond_mode=cond_mode,
                              measure='par_corr',
                              tau_max=2, verbosity=verbosity)
    numpy.testing.assert_allclose(res[0][i, j, tau], expected_parcorr,
                                  rtol=0.1)
    # All non-links should be almost zero
    for nj in range(N):
        for ni in range(N):
            for ntau in range(3):
                if (ni, -ntau) not in links[nj]:
                    numpy.testing.assert_allclose(res[0][ni, nj, ntau], 0.,
                                                  atol=0.05)

    # Now testing a contemporaneous MIT
    # Graph below
    az = .5
    ax = .5
    ay = .5
    sxy = .3
    syz = .5
    sxz = .6
    links_coeffs = {0: [((0, -1), az), ((1, 0), sxz), ((2, 0), syz)],
                    1: [((1, -1), ax), ((0, 0), sxz), ((2, 0), sxy)],
                    2: [((2, -1), ay), ((0, 0), syz), ((1, 0), sxy)],
                    }

    data, links = pp.var_process(links_coeffs, T=5000,
                                 use='inno_cov', verbosity=verbosity)
    T, N = data.shape

    i = 0
    j = 1
    tau = 0
    cond_mode = 'parents_xy'
    expected_parcorr = - (syz * sxy - sxz * 1.**2) / numpy.sqrt(
        (syz**2 - 1.**2 * 1.**2) * (sxy**2 - 1.**2 * 1.**2))
    res = te.get_lagfunctions(data, parents_neighbors=links,
                              cond_mode=cond_mode,
                              measure='par_corr',
                              tau_max=2, verbosity=verbosity)
    numpy.testing.assert_allclose(res[0][i, j, tau], expected_parcorr,
                                  rtol=0.2)


def test_measures():

    print("\nTesting function '_calculate_lag_function' to check dependence "
          "measures are correctly estimated.")

    measure_params = {'knn': 10, }
    ax = 0.6
    ay = 0.3
    cxy = .7
    links_coeffs = {0: [((0, -1), ax)],
                    1: [((1, -1), ay), ((0, -1), cxy)],
                    }
    numpy.random.seed(42)
    data, links = pp.var_process(links_coeffs, T=5000,
                                 use='inno_cov', verbosity=verbosity)
    T, N = data.shape

    expected_cmi = 0.5 * numpy.log(1. + (cxy**2 * 1.**2) / (1.**2))
    expected_parcorr = cmi2parcorr_trafo(expected_cmi)
    gamma_x = 1.**2 / (1. - ax**2)
    gamma_xy = (ax * cxy * gamma_x) / (1. - ax * ay)
    gamma_y = (cxy**2 * gamma_x + 1.**2 + 2. *
               ay * cxy * gamma_xy) / (1. - ay**2)
    expected_reg = cxy * numpy.sqrt(gamma_x / gamma_y)

    for measure in ['par_corr', 'reg', 'cmi_knn', 'cmi_gauss']:
        res = te._calculate_lag_function(
            measure=measure,
            data=data,
            var_x=0, var_y=1,
            conds_x=links[0], conds_y=links[1],
            measure_params=measure_params,
            tau_max=1,
            selected_lags=[1],
            verbosity=verbosity)['cmi']
        if measure == 'par_corr':
            print("%s = %.3f (expected = %.3f)"
                  % (measure, res[1], expected_parcorr))
            numpy.testing.assert_allclose(res[1], expected_parcorr, rtol=0.1)
        elif measure == 'reg':
            print("%s = %.3f (expected = %.3f)"
                  % (measure, res[1], expected_reg))
            numpy.testing.assert_allclose(res[1], expected_reg, rtol=0.1)
        elif (measure == 'cmi_knn' or measure == 'cmi_gauss'):
            print("%s = %.3f (expected = %.3f)"
                  % (measure, res[1], expected_cmi))
            numpy.testing.assert_allclose(res[1], expected_cmi, rtol=0.1)

    # binning estimator
    symb_data = pp.quantile_bin_array(data, bins=5)
    res = te._calculate_lag_function(
        measure='cmi_symb',
        data=symb_data,
        var_x=0, var_y=1,
        conds_x=links[0], conds_y=links[1],
        measure_params=measure_params,
        tau_max=1,
        selected_lags=[1],
        verbosity=verbosity)['cmi']
    print("%s = %.3f (expected = %.3f)"
          % ('symb', res[1], expected_cmi))
    numpy.testing.assert_allclose(res[1], expected_cmi, rtol=0.3)

    # ordinal pattern estimator
    symb_data = pp.ordinal_patt_array(data,
                                      array_mask=numpy.ones(data.shape,
                                                            dtype='int32'),
                                      dim=3, step=2)[0]
    res = te._calculate_lag_function(
        measure='cmi_symb',
        data=symb_data,
        var_x=0, var_y=1,
        conds_x=links[0], conds_y=links[1],
        measure_params=measure_params,
        tau_max=1,
        selected_lags=[1],
        verbosity=verbosity)['cmi']
    print("%s = %.3f (expected = %.3f)"
          % ('ordinal-symb', res[1], expected_cmi))
    numpy.testing.assert_allclose(res[1], expected_cmi, rtol=0.4)


class test__get_significance_estimate():

    """Testing the false positive rate for independent processes."""
    # array = numpy.array([[0, 10, 20, 30],
    #                      [1, 11, 21, 31],
    #                      [2, 12, 22, 32],
    #                      [3, 13, 23, 33],
    #                      [4, 14, 24, 34]])

    def __init__(self):
        print("\nChecking significance tests analytic and shuffle")

        self.samples = 1000
        self.sig_lev = .95
        self.sig_samples = 1000
        self.T = 500
        self.rtol = .2
        self.measure_params = {'knn': 10, }

    def test_alphas(self):
        print("Checking whether sig_lev %.2f results in " % self.sig_lev +
              "%.2f false positives" % (1. - self.sig_lev))

        for significance in ['analytic']:  # , 'full_shuffle']:
            for measure in ['par_corr', 'reg']:
                res = self.get_fpr(measure, significance=significance)
                print("%s test for %s has FPR %.3f (expected = %.3f)" % (
                    significance, measure, res, 1. - self.sig_lev))
                numpy.testing.assert_allclose(res, 1. - self.sig_lev,
                                              rtol=self.rtol)

    def test_shuffle_vs_alpha(self):
        print("Checking that shuffle sig_thres equals analytical thres "
              "for sig_lev = %.2f, sig_samples = %d" % (self.sig_lev,
                                                        self.sig_samples))
        numpy.random.seed(42)
        # array = numpy.random.randn(2, self.T)
        links_coeffs = {0: [],
                        1: [((0, 0), .06)], }
        data, links = pp.var_process(links_coeffs, T=self.T,
                                     use='inno_cov', verbosity=verbosity)
        array = data.T
        xyz = numpy.array([0, 1])
        for measure in ['par_corr', 'reg']:
            expected = te._get_estimate(array=array, measure=measure, xyz=xyz,
                                        significance='analytic',
                                        sig_samples=self.sig_samples,
                                        sig_lev=self.sig_lev,
                                        measure_params=self.measure_params,
                                        verbosity=0)['sig_thres']
            res = te._get_estimate(array=array, measure=measure, xyz=xyz,
                                   significance='full_shuffle',
                                   sig_samples=self.sig_samples,
                                   sig_lev=self.sig_lev,
                                   measure_params=self.measure_params,
                                   verbosity=0)['sig_thres']
            print("shuffle %.2f sig thres for %s = %.4f (analytic = %.4f)"
                  % (self.sig_lev, measure, res, expected))
            numpy.testing.assert_allclose(res, expected, rtol=self.rtol)

    def get_fpr(self, measure, significance):
        fpr = numpy.zeros(self.samples)
        for sam in range(self.samples):
            # data, links = var_process(self.links_coeffs, T=self.T,
            #                           use='inno_cov', verbosity=verbosity)
            data = numpy.random.randn(self.T, 2)
            # .argsort(axis=0).argsort(axis=0)
            res = te._calculate_lag_function(
                measure=measure,
                data=data,
                var_x=0, var_y=1,
                conds_x=None, conds_y=None,
                measure_params=self.measure_params,
                tau_max=1,
                selected_lags=[1],
                significance=significance,
                sig_lev=self.sig_lev,
                sig_samples=self.sig_samples,
                verbosity=0)
            fpr[sam] = res['cmi_pval'][1]

        return (fpr <= (1. - self.sig_lev)).mean()


class test__get_confidence_estimate():

    """Testing the false positive rate for independent processes."""

    def __init__(self):
        print("\nChecking confidence tests analytic and shuffle")

        self.conf_lev = .9
        self.conf_samples = 1000
        self.T = 500
        self.links_coeffs = {0: [],
                             1: [],
                             }
        ax = 0.8
        ay = 0.9
        cxy = .7
        links_coeffs = {0: [((0, -1), ax)],
                        1: [((1, -1), ay), ((0, -1), -cxy)],
                        }
        self.rtol = .1
        self.measure_params = {'knn': 10}

        numpy.random.seed(42)
        data, links = pp.var_process(links_coeffs, T=self.T,
                                     use='inno_cov', verbosity=verbosity)
        self.array, self.xyz = te._construct_array(X=[(0, -1)], Y=[(1, 0)],
                                                   Z=[(0, -2), (1, -1)],
                                                   tau_max=2,
                                                   data=data, selector=False)

    def test_shuffle_vs_alpha(self):
        print("Checking that bootstrap conf intervals equal analytical ones "
              "for conf_lev = %.2f, conf_samples = %d" % (self.conf_lev,
                                                          self.conf_samples))
        for measure in ['reg', 'par_corr', 'cmi_gauss']:

            tmp = te._get_estimate(array=self.array,
                                   measure=measure, xyz=self.xyz,
                                   confidence='analytic',
                                   conf_samples=self.conf_samples,
                                   conf_lev=self.conf_lev,
                                   measure_params=self.measure_params,
                                   verbosity=0)
            expected = tmp['conf_lower'], tmp['conf_upper']

            tmp = te._get_estimate(array=self.array,
                                   measure=measure, xyz=self.xyz,
                                   confidence='bootstrap',
                                   conf_samples=self.conf_samples,
                                   conf_lev=self.conf_lev,
                                   measure_params=self.measure_params,
                                   verbosity=0)
            res = tmp['conf_lower'], tmp['conf_upper']

            print("bootstrap interval for %s = (%.3f, %.3f)  -- "
                  "analytic interval = (%.3f, %.3f)" % (
                      measure, res[0], res[1], expected[0], expected[1]))
            numpy.testing.assert_allclose(res, expected, rtol=self.rtol)


if __name__ == "__main__":
    # unittest.main()

    # Individual tests
    test_pc_algo_all()
    test_construct_array()
    test_get_lagfunctions()
    test_measures()
    sig = test__get_significance_estimate()
    sig.test_shuffle_vs_alpha()
    sig.test_alphas()
    conf = test__get_confidence_estimate()
    conf.test_shuffle_vs_alpha()
    print("\nAll passed!")

    # Nose...
    # result = nose.run()
