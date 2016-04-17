# TIGRAMITE – TIME SERIES GRAPH BASED MEASURES OF INFORMATION TRANSFER

Version 2.0 beta (release date 2016-04-15)

(Python Package)


## General Notes

Tigramite is a time series analysis python module. With flexibly adaptable scripts it allows to reconstruct graphical models (conditional independence graphs) from discrete or continuously-valued time series based on a causal discovery algorithm, quantify interaction strengths with different measures, and create high-quality plots of the results.


## Features

- Analysis can be performed on multivariate time series. Further scripts allow sliding window or ensemble analyses

- Functions for custom preprocessing like anomalization, high/lowpass filters, masking of samples (e.g. winter months only), time-binning, ordinal pattern analysis, and more

- Different (conditional) measures of association (partial correlation, standardized regression, and conditional mutual information with different estimators)

- Fast computation through use of C-code via scipy.weave; also fully parallelized script (mpi4py package necessary) available

- Significance testing via analytical tests or a shuffle test for conditional mutual information

- Flexible plotting scripts for publication quality presentation of results


## References

Runge, J. et al. (2015). Identifying causal gateways and mediators in complex spatio-temporal systems. Nature Communications, 6, 8502. 
doi:10.1038/ncomms9502

Runge, J., Heitzig, J., Petoukhov, V., Kurths, J. (2012): Escaping the Curse of Dimensionality in Estimating Multivariate Transfer Entropy, Physical Review Letters, 108, 258701
doi:10.1103/PhysRevLett.108.258701

Runge, J., Heitzig, J., Marwan, N., Kurths, J. (2012): Quantifying Causal Coupling Strength: A Lag-specific Measure For Multivariate Time Series Related To Transfer Entropy, Physical Review E, 86, 061121
doi:10.1103/PhysRevE.86.061121

Runge, J., Petoukhov, V., Kurths, J. (2014): Quantifying the strength and delay of climatic interactions, Journal of Climate, 27, 720-739
doi:10.1175/JCLI-D-13-00159.1


## Required python packages

- numpy, tested with Version 1.10
- scipy, tested with Version 0.16
- matplotlib, tested with Version 1.5
- networkx, tested with Version 1.10
- basemap (only if plotting on a map is needed)
- mpi4py (optional, necessary for using the parallelized implementation)


## User Agreement

By downloading TiGraMITe you agree with the following points: The toolbox is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations following from application the toolbox.

You commit to cite TiGraMITe in your reports or publications if used.


## License

Copyright (C) 2012-2016 Jakob Runge

mpi4py wrapper module "mpi.py" Copyright (C) 2012 Jobst Heitzig

See license.txt for full text.

TiGraMITe is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version. TiGraMITe is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

## Usage

Edit one of the tigramite_analysis_*.py template files to load your data and choose parameters. Also create a folder for your results (the default folder is 'test/'). Then run as

$ python tigramite_analysis_template_beta.py

The results pickle file and figures will be saved in the specified folder.