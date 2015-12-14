

import numpy
import tigramite_estimation as te
import tigramite_preprocessing as pp
import tigramite_plotting 

import nose
import nose.tools as nt


def assert_graphs_equal(actual, expected):
    """Check whether lists in dict are equal"""

    for j in expected.keys():
        nt.assert_items_equal(actual[j], expected[j])


def _get_parent_graph(nodes, exclude=None):
    """Returns parents"""

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
    return numpy.sqrt(1.-numpy.exp(-2.*cmi))


def var_process(parents_neighbors_coeffs, T=1000, use='inv_inno_cov',
                verbosity=0):

    max_lag = 0
    for j in parents_neighbors_coeffs.keys():
        for node, coeff in parents_neighbors_coeffs[j]:
            i, tau = node[0], -node[1]
            max_lag = max(max_lag, abs(tau))
    N = max(parents_neighbors_coeffs.keys()) + 1
    graph = numpy.zeros((N, N, max_lag))
    innos = numpy.zeros((N, N))
    innos[range(N), range(N)] = 1.
    true_parents_neighbors = {}
    # print graph.shape
    for j in parents_neighbors_coeffs.keys():
        true_parents_neighbors[j] = []
        for node, coeff in parents_neighbors_coeffs[j]:
            i, tau = node[0], -node[1]
            true_parents_neighbors[j].append((i, -tau))
            if tau == 0:
                innos[j, i] = innos[i, j] = coeff
            else:
                graph[j, i, tau - 1] = coeff

    if verbosity > 0:
        print("VAR graph =\n%s" % str(graph))
        if use == 'inno_cov':
            print("\nInnovation Cov =\n%s" % str(innos))
        elif use == 'inv_inno_cov':
            print("\nInverse Innovation Cov =\n%s" % str(innos))

    data = pp.var_network(graph=graph, inv_inno_cov=innos,
                          inno_cov=innos,
                          use=use, T=T)

    return data, true_parents_neighbors
##
# Test data:
# VAR process with given parents and neighbors and all coefficients equal
##
coeff = 0.3
T = 2000
numpy.random.seed(42)
# True graph
links_coeffs = {0: [((0, -1), coeff), ((2, -1), coeff), ((1, 0), coeff)],
                1: [((1, -1), coeff), ((0, -1), coeff),
                    ((0, 0), coeff), ((2, 0), coeff)],
                2: [((2, -1), coeff), ((1, -1), coeff),
                    ((0, -2), coeff), ((1, 0), coeff)],
                3: [((3, -1), 1.*coeff), ((2, -1), .9*coeff),
                    ((2, -4), -.8*coeff), ((1, -2), -.7*coeff)],
                }

fulldata, true_parents_neighbors = var_process(links_coeffs, T=T)
T, N = fulldata.shape

# fulldata_mask = numpy.ones(fulldata.shape, dtype='bool')
# fulldata_mask[:, :] = numpy.random.randint(0, 2, size=(T, N)).astype('bool')
# print fulldata_mask
# print graph
print("\n" + "-" * 60)
print("Testing tigramite estimation with graph")
for j in range(N):
    print("\n\tVariable %d: %s" % (j, true_parents_neighbors[j]))
print("\nand all coefficients = %.2f" % coeff)
print("\n" + "-" * 60)


verbosity = 1

#
#  Start
#


def test_pc_algo_all():

    max_trials = 5
    measure_params = {'knn': 100}
    significance = 'analytic'
    # parents_neighbors = te.pc_algo_all(
    #     fulldata, estimate_parents_neighbors='parents',
    #     tau_min=0, tau_max=4,
    #     initial_conds=2, max_conds=6, max_trials=max_trials,
    #     measure='par_corr',
    #     significance='alpha', sig_lev=0.995, sig_samples=100,
    #     fixed_thres=0.015,
    #     measure_params=measure_params,
    #     mask=False, mask_type=['y'], data_mask=numpy.array([0]),
    #     initial_parents_neighbors=None,
    #     verbosity=verbosity)
    # assert_graphs_equal(parents_neighbors,
    #                     get_parent_graph(true_parents_neighbors))
    # assert 1==2

    print("\nChecking different initial_conds =")
    for initial_conds in range(1, 4):
        print("%d" % initial_conds)

        parents_neighbors = te.pc_algo_all(
            fulldata, estimate_parents_neighbors='both',
            tau_min=0, tau_max=4,
            initial_conds=initial_conds, max_conds=6, max_trials=max_trials,
            measure='par_corr',
            significance=significance, sig_lev=0.995, sig_samples=100,
            fixed_thres=0.015,
            measure_params=measure_params,
            mask=False, mask_type=['y'], data_mask=numpy.array([0]),
            initial_parents_neighbors=None,
            verbosity=verbosity)
        assert_graphs_equal(parents_neighbors, true_parents_neighbors)

    print("\nChecking initial_parents_neighbors...")
    parents_neighbors = te.pc_algo_all(
        fulldata, estimate_parents_neighbors='both',
        tau_min=0, tau_max=4,
        initial_conds=1, max_conds=6, max_trials=max_trials,
        measure='par_corr',
        significance=significance, sig_lev=0.995, sig_samples=100,
        fixed_thres=0.015,
        measure_params=measure_params,
        mask=False, mask_type=['y'], data_mask=numpy.array([0]),
        initial_parents_neighbors=true_parents_neighbors,
        verbosity=verbosity)
    assert_graphs_equal(parents_neighbors, true_parents_neighbors)

    print("\nChecking estimate_parents_neighbors =")
    for estimate_parents_neighbors in ['parents', 'both']:
        print("%s" % estimate_parents_neighbors)
        parents_neighbors = te.pc_algo_all(
            fulldata,
            estimate_parents_neighbors=estimate_parents_neighbors,
            tau_min=0, tau_max=4,
            initial_conds=1, max_conds=6, max_trials=max_trials,
            measure='par_corr',
            significance='analytic', sig_lev=0.995, sig_samples=100,
            fixed_thres=0.015,
            measure_params=measure_params,
            mask=False, mask_type=['y'], data_mask=numpy.array([0]),
            initial_parents_neighbors=None,
            verbosity=verbosity)
        if estimate_parents_neighbors == 'both':
            assert_graphs_equal(parents_neighbors, true_parents_neighbors)
        elif estimate_parents_neighbors == 'parents':
            assert_graphs_equal(_get_parent_graph(parents_neighbors),
                                _get_parent_graph(true_parents_neighbors))

    # unittest.assertDictEqual(parents_neighbors, true_parents_neighbors)
    # nt.assert_dicts_almost_equal(parents_neighbors, true_parents_neighbors)


# def test_pc_algo():

#     for initial_conds in range(1, 4):
#         print("\n%d==================================== " %initial_conds)
#         parents_neighbors = te._pc_algo(fulldata, j=0,
#                 parents_or_neighbors='parents',
#                 initial_graph=None, all_parents=None,
#                 tau_min=0, tau_max=4,
#                 initial_conds=initial_conds, max_conds=6, max_trials=5,
#                 measure='par_corr',
#                 significance='alpha', sig_lev=0.995, sig_samples=100,
#                 fixed_thres=0.015,
#                 measure_params=measure_params,
#                 mask=False, mask_type=None, data_mask=numpy.array([0]),
#                 verbosity=2)

#         nt.assert_items_equal(parents_neighbors,
            # te._get_parents(true_parents_neighbors[0]))

def test_construct_array():

    data = numpy.array([[0, 10, 20, 30],
                        [1, 11, 21, 31],
                        [2, 12, 22, 32],
                        [3, 13, 23, 33],
                        [4, 14, 24, 34]])
    data_mask = numpy.array([[1, 0, 0, 1],
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
        mask=False,
        data=data,
        data_mask=data_mask,
        mask_type=None, verbosity=verbosity)
    numpy.testing.assert_almost_equal(res[0],
                                      numpy.array([[11.,  12.,  13.],
                                                   [2.,   3.,   4.],
                                                   [1.,   2.,   3.],
                                                   [10.,  11.,  12.],
                                                   [22.,  23.,  24.]]))
    numpy.testing.assert_almost_equal(res[1], numpy.array([0, 1, 2, 2, 2]))

    # masking y
    res = te._construct_array(
        X=X, Y=Y, Z=Z,
        tau_max=tau_max,
        mask=True,
        data=data,
        data_mask=data_mask,
        mask_type=['y'], verbosity=verbosity)

    numpy.testing.assert_almost_equal(res[0],
                                      numpy.array([[12.,  13.],
                                                   [3.,   4.],
                                                   [2.,   3.],
                                                   [11.,  12.],
                                                   [23.,  24.]]))

    numpy.testing.assert_almost_equal(res[1], numpy.array([0, 1, 2, 2, 2]))

    # masking all
    res = te._construct_array(
        X=X, Y=Y, Z=Z,
        tau_max=tau_max,
        mask=True,
        data=data,
        data_mask=data_mask,
        mask_type=['x', 'y', 'z'], verbosity=verbosity)

    numpy.testing.assert_almost_equal(res[0],
                                      numpy.array([[13.],
                                                   [4.],
                                                   [3.],
                                                   [12.],
                                                   [24.]]))

    numpy.testing.assert_almost_equal(res[1], numpy.array([0, 1, 2, 2, 2]))


def test_get_lagfunctions():

    significance = 'analytic'

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
    data, links = var_process(links_coeffs, T=5000, verbosity=verbosity)
    T, N = data.shape

    i = 1
    j = 2
    tau = 2

    # Test that cross correlation is correct
    cond_mode = 'none'
    expected_cmi = 0.5*numpy.log(1. + (cxy**2 * 1.**2)
                                 / (1. - ax**2) / (cwy**2 * 1.**2 + 1.**2))
    expected_parcorr = cmi2parcorr_trafo(expected_cmi)
    res = te.get_lagfunctions(data, parents_neighbors=links,
                              cond_mode=cond_mode,
                              measure='par_corr',
                              tau_max=2, verbosity=verbosity)
    numpy.testing.assert_almost_equal(res[0][i, j, tau],
                                      numpy.array([expected_parcorr]),
                                      decimal=2)

    # Test that ITY is correct
    cond_mode = 'parents_y'
    expected_cmi = 0.5*numpy.log(1. + (cxy**2 * 1.**2)
                                 / (1. - ax**2))
    expected_parcorr = cmi2parcorr_trafo(expected_cmi)
    res = te.get_lagfunctions(data, parents_neighbors=links,
                              cond_mode=cond_mode,
                              measure='par_corr',
                              tau_max=2, verbosity=verbosity)
    numpy.testing.assert_almost_equal(res[0][i, j, tau],
                                      numpy.array([expected_parcorr]),
                                      decimal=2)

    # Test that ITX is correct
    cond_mode = 'parents_x'
    expected_cmi = 0.5*numpy.log(1. + (cxy**2 * 1.**2)
                                 / (cwy**2 * 1.**2 + 1.**2))
    expected_parcorr = cmi2parcorr_trafo(expected_cmi)
    res = te.get_lagfunctions(data, parents_neighbors=links,
                              cond_mode=cond_mode,
                              measure='par_corr',
                              tau_max=2, verbosity=verbosity)
    numpy.testing.assert_almost_equal(res[0][i, j, tau],
                                      numpy.array([expected_parcorr]),
                                      decimal=2)

    # Test that MIT is correct
    cond_mode = 'parents_xy'
    expected_cmi = 0.5*numpy.log(1. + (cxy**2 * 1.**2)
                                 / (1.**2))
    expected_parcorr = cmi2parcorr_trafo(expected_cmi)
    res = te.get_lagfunctions(data, parents_neighbors=links,
                              cond_mode=cond_mode,
                              measure='par_corr',
                              tau_max=2, verbosity=verbosity)
    numpy.testing.assert_almost_equal(res[0][i, j, tau],
                                      numpy.array([expected_parcorr]),
                                      decimal=2)
    # All non-links should be almost zero
    for nj in range(N):
        for ni in range(N):
            for ntau in range(3):
                if (ni, -ntau) not in links[nj]:
                    # if abs(res[0][ni, nj, ntau]) > 0.02:
                    #     print ni, nj, ntau, res[0][ni, nj, ntau]
                    numpy.testing.assert_almost_equal(res[0][ni, nj, ntau],
                                                      numpy.array([0.]),
                                                      decimal=1)

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

    data, links = var_process(links_coeffs, T=5000,
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
    numpy.testing.assert_almost_equal(res[0][i, j, tau],
                                      numpy.array([expected_parcorr]),
                                      decimal=2)


def test_measures():

    measure_params = {'knn': 10, 'rescale_cmi': False}
    decimal = 2
    ax = 0.5
    ay = 0.6
    cxy = .7
    links_coeffs = {0: [((0, -1), ax)],
                    1: [((1, -1), ay), ((0, -1), cxy)],
                    }
    numpy.random.seed(42)
    data, links = var_process(links_coeffs, T=5000,
                              use='inno_cov', verbosity=verbosity)
    T, N = data.shape

    print("Checking precision up to %d decimals" % decimal)
    expected_cmi = 0.5*numpy.log(1. + (cxy**2 * 1.**2)
                                 / (1.**2))
    expected_parcorr = cmi2parcorr_trafo(expected_cmi)
    gamma_x = 1.**2 / (1. - ax**2)
    gamma_xy = (ax*cxy*gamma_x) / (1. - ax*ay)
    gamma_y = (cxy**2*gamma_x + 1.**2 + 2.*ay*cxy*gamma_xy) / (1. - ay**2)
    expected_reg = cxy * numpy.sqrt(gamma_x / gamma_y)
    for measure in ['par_corr', 'reg', 'cmi_knn', 'cmi_hybrid', 'cmi_gauss']:
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
            numpy.testing.assert_almost_equal(res[1],
                                              numpy.array([expected_parcorr]),
                                              decimal=decimal)
        elif measure == 'reg':
            print("%s = %.3f (expected = %.3f)"
                  % (measure, res[1], expected_reg))
            numpy.testing.assert_almost_equal(res[1],
                                              numpy.array([expected_reg]),
                                              decimal=decimal)
        elif (measure == 'cmi_knn'
              or measure == 'cmi_hybrid'
              or measure == 'cmi_gauss'):
            print("%s = %.3f (expected = %.3f)"
                  % (measure, res[1], expected_cmi))
            numpy.testing.assert_almost_equal(res[1],
                                              numpy.array([expected_cmi]),
                                              decimal=decimal)

    # binning estimator
    symb_data = pp.quantile_bin_array(data, bins=6)
    res = te._calculate_lag_function(
        measure='cmi_symb',
        data=symb_data,
        var_x=0, var_y=1,
        conds_x=links[0], conds_y=links[1],
        measure_params=measure_params,
        tau_max=1,
        selected_lags=[1],
        verbosity=verbosity)['cmi']
    print("%s = %.3f (expected = %.3f, here only 1 decimal checked)"
          % ('symb', res[1], expected_cmi))
    numpy.testing.assert_almost_equal(res[1],
                                      numpy.array([expected_cmi]),
                                      decimal=1)
    # ordinal pattern estimator
    symb_data = pp.ordinal_patt_array(data,
                                      array_mask=numpy.ones(data.shape,
                                                            dtype='int32'),
                                      dim=3, step=10)[0]
    res = te._calculate_lag_function(
        measure='cmi_symb',
        data=symb_data,
        var_x=0, var_y=1,
        conds_x=links[0], conds_y=links[1],
        measure_params=measure_params,
        tau_max=1,
        selected_lags=[1],
        verbosity=verbosity)['cmi']
    print("%s = %.3f (expected = %.3f, here only 1 decimal checked)"
          % ('ordinal-symb', res[1], expected_cmi))
    numpy.testing.assert_almost_equal(res[1],
                                      numpy.array([expected_cmi]),
                                      decimal=1)


class test__get_significance_estimate():

    """Testing the false positive rate for independent processes."""
    # array = numpy.array([[0, 10, 20, 30],
    #                      [1, 11, 21, 31],
    #                      [2, 12, 22, 32],
    #                      [3, 13, 23, 33],
    #                      [4, 14, 24, 34]])

    # print te._get_significance_estimate(
    #     array=array.T,
    #     xyz=numpy.array([0, 1, 1, 2]),
    #     significance='shuffle', measure='reg',
    #     sig_samples=4, sig_lev=.5, fixed_thres=.5,
    #     paras=None,
        # verbosity=verbosity)

    def __init__(self):
        print("\nChecking significance tests alpha and shuffle")

        self.samples = 10000
        self.sig_lev = .95
        self.sig_samples = 10000
        self.T = 500
        self.decimal = 2
        self.links_coeffs = {0: [],
                             1: [],
                             }

    def test_alphas(self):
        print("\nChecking whether sig_lev %.2f results in " % self.sig_lev +
              "%.2f false positives" % (1. - self.sig_lev))

        for measure in ['par_corr', 'reg']:
            res = self.get_fpr(measure, significance='analytic')
            print("%s = %.3f (expected = %.3f)"
                % (measure, res, 1.-self.sig_lev))
            numpy.testing.assert_almost_equal(res, 1.-self.sig_lev, 
                                              decimal=self.decimal)

    def test_shuffle_vs_alpha(self):
        print("\nChecking that shuffle sig_thres equals analytical thres "
              "for sig_lev = %.2f, sig_samples = %d" % (self.sig_lev, 
                                                        self.sig_samples))
        numpy.random.seed(42)
        array = numpy.random.randn(2, self.T)
        xyz = numpy.array([0, 1])
        for measure in ['par_corr', 'reg', 'cmi_gauss']:
            expected = te.get_sig_thres(
                                        sig_lev=self.sig_lev,
                                        df=self.T - len(xyz),
                                        measure=measure, array=array)
            res = te._get_significance_estimate(
                array=array, xyz=xyz, significance='shuffle',
                measure=measure,
                sig_samples=self.sig_samples, sig_lev=self.sig_lev,
                fixed_thres=None, measure_params=None,
                verbosity=0)
            print("%s = %.3f (expected = %.3f)"
                  % (measure, res, expected))
            numpy.testing.assert_almost_equal(res, expected,
                                              decimal=self.decimal)

    # def test_shuffle(self):
    #     for measure in ['reg']:   #, 'par_corr', reg', 'cmi_knn']:
    #         res = self.get_fpr(measure, significance='shuffle')
    #         print("%s = %.3f (expected = %.3f)"
    #             % (measure, res, 1.-self.sig_lev))
    #         numpy.testing.assert_almost_equal(res, 1.-self.sig_lev,
    #                                           decimal=self.decimal)

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
                            measure_params=None,
                            tau_max=1,
                            selected_lags=[1],
                            significance=significance,
                            sig_lev=self.sig_lev,
                            sig_samples=self.sig_samples,
                            verbosity=0)
            fpr[sam] = res['cmi'][1] > res['cmi_sig'][1]

        return fpr.mean()


class test__get_confidence_estimate():

    """Testing the false positive rate for independent processes."""

    def __init__(self):
        print("\nChecking significance tests alpha and shuffle")

        self.conf_lev = .95
        self.conf_samples = 10000
        self.T = 2000
        self.decimal = 2
        self.links_coeffs = {0: [],
                             1: [],
                             }
        ax = 0.8
        ay = 0.9
        cxy = .7
        links_coeffs = {0: [((0, -1), ax)],
                        1: [((1, -1), ay), ((0, -1), cxy)],
                        }
        numpy.random.seed(42)
        data, links = var_process(links_coeffs, T=self.T,
                                  use='inno_cov', verbosity=verbosity)
        self.array, self.xyz = te._construct_array(X=[(0, -1)], Y=[(1, 0)],
                                                   Z=[(0, -2), (1, -1)],
                                                   tau_max=2,
                                                   data=data, mask=False)
                                                   # .argsort(axis=0).argsort(axis=0)

    def test_shuffle_vs_alpha(self):
        print("\nChecking that bootstrap conf levels equal analytical ones "
              "for conf_lev = %.2f, conf_samples = %d" % (self.conf_lev,
                                                          self.conf_samples))
        for measure in ['reg', 'par_corr', 'cmi_gauss']:
            value = te._get_estimate(
                                array=self.array, measure=measure,
                                xyz=self.xyz,
                                measure_params=None,
                                verbosity=0)
            expected = te._get_confidence_estimate(
                            array=self.array, xyz=self.xyz,
                            confidence='analytic',
                            measure=measure,
                            value=value,
                            conf_samples=self.conf_samples,
                            conf_lev=self.conf_lev,
                            measure_params=None,
                            verbosity=0)

            res = te._get_confidence_estimate(
                            array=self.array, xyz=self.xyz,
                            confidence='bootstrap',
                            measure=measure,
                            value=value,
                            conf_samples=self.conf_samples,
                            conf_lev=self.conf_lev,
                            measure_params=None,
                            verbosity=0)
            print("%s = %.3f (+/- %.3f, %.3f) (expected = (%.3f, %.3f))"
                % (measure, value, res[0], res[1], expected[0], expected[1]))
            numpy.testing.assert_almost_equal(numpy.array([res]), numpy.array([expected]), 
                                              decimal=self.decimal)

if __name__ == "__main__":
    # unittest.main()
    # test_pc_algo_all()
    # test_construct_array()
    # test_get_lagfunctions()
    # test_measures()
    # sig = test__get_significance_estimate()
    # sig.test_alphas()
    # sig.test_shuffle_vs_alpha()
    # conf = test__get_confidence_estimate()
    # conf.test_shuffle_vs_alpha()
    ####sig.test_shuffle()

    # print("All passed!")
    result = nose.run()


# TODO: 
# documentation
# hybrid raus
# plotting bei asymetric contemp values (due to masking/conditions)
# Tigramite Icon fuer github page von Cosimo zeichnen lassen? Gegen Bezahlung...
# weave --> cython ?
# path analysis einbauen?