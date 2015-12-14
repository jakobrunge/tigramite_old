TIGRAMITE – TIME SERIES GRAPH BASED MEASURES OF INFORMATION TRANSFER

Version 2.0 (release date 2016-01-01)

(Python Package)


General Notes:

Tigramite is a time series analysis python module. With flexibly adaptable scripts it allows to detect and quantify causal dependencies from time series and create high-quality plots of the results.


Features:

- Analysis can be performed on one multivariate time series, in sliding windows of one multivariate time series, or an ensemble of multivariate time series

- Flexible Python framework for custom preprocessing like anomalization, high/lowpass filters, masking of samples (e.g. winter months only), and more

- Different (conditional) measures of association (partial correlation, conditional mutual information with different estimators)

- Estimation either from prescribed graph for testing hypotheses OR from iteratively estimated time series graph

- Fast computation through use of C-code via scipy.weave and fully parallelized script (mpi4py package necessary)

- Significance testing via prescribed threshold or sophisticated shuffle test

- Flexible specification of experiment and variable names and units

- Flexible plotting scripts for publication quality presentation of results



References:

Runge, J., Heitzig, J., Petoukhov, V., Kurths, J.: Escaping the Curse of Dimensionality in Estimating Multivariate Transfer Entropy, Physical Review Letters, 108, 2012, 258701
doi:10.1103/PhysRevLett.108.258701

Runge, J., Heitzig, J., Marwan, N., Kurths, J.: Quantifying Causal Coupling Strength: A Lag-specific Measure For Multivariate Time Series Related To Transfer Entropy, Physical Review E, 86, 2012, 061121
doi:10.1103/PhysRevE.86.061121

Runge, J., Petoukhov, V., Kurths, J.: Quantifying the strength and delay of climatic interactions, Journal of Climate, 27, 2014, 720-739
doi:10.1175/JCLI-D-13-00159.1

