# DMSP SSM NASA CDF Tools

  If your interested in the auroral boundary algorithm described in Kilcommons et al. 2017, see [this repository](https://github.com/lkilcommons/ssj_auroral_boundary)

This repository includes tools and information about the data from the Magnetometer (SSM) instrument 
aboard the Defense Meteorolgy Satellite Program (DMSP) spacecraft.

Specificially, it is a companion to the Level-2 DMSP SSM CDF files which were prepared as part of NASA grant #NNX13AG07G

You can get the data from the following locations: 

* [CDAWeb Virtual Observatory](http://cdaweb.gsfc.nasa.gov/istp_public/)
* [National Oceanic and Atmospheric Administration (NOAA NCEI)](http://satdat.ngdc.noaa.gov/dmsp/)

## Read the [Manual](DMSPSpaceWxSSJSSMSSIESATBDandUsersManual_v1_1.pdf)
* Section 2.1 has a table of CDF variable names for spacecraft locations
* To learn about the auroral boundaries included in this dataset, read section 2.1.1
* To learn about the SSM dataset, read section 2.3

To read additional details about the instrument and our files, visit the [DMSP NOAA NCEI (Formerly NGDC) Documentation](http://satdat.ngdc.noaa.gov/dmsp/docs/)

# Rules of the Road
This software is associated with the following publication.

[Kilcommons, L. M., R. J. Redmon, and D. J. Knipp (2017), A New DMSP Magnetometer & Auroral Boundary Dataset and Estimates of Field Aligned Currents in Dynamic Auroral Boundary Coordinates, J. Geophys. Res. Space Physics, 122, doi:10.1002/2016JA023342.
](http://dx.doi.org/10.1002/2016JA023342)

If you write a paper using this software, please acknowledge us or cite the above paper. We would also appreciate if you contact us via the corresponding author email associated with the above publication.

# Software

This software was written to simplify reading DMSP SSM CDF files in Python. Reading it should also help students and researchers understand what is in this data, and how to use it.

These tools are written in the Python language and require the following special libraries in addition to numpy, matplotlib and other tools included in the requirements of the installation script.

* [NASA CDF library](http://cdf.gsfc.nasa.gov/html/sw_and_docs.html)
* [spacepy](http://spacepy.lanl.gov/index.shtml)

These packages provide the tools nessecary to read the NASA CDF format data available from CDAWeb.

# Tools Currently in This Repository

### SSM Step Remover
DMSP SSM magnetometer suffers from occasional step discontinuities in the baseline due to currents from the spacecraft electronics. This tool attempts to automatically remove them.

Usage:
```
python ssm_step_remover.py /path/to/my/data/dmsp_ssm_magnetometer_data_file.cdf
```
To save changes to CDF file:
```
python ssm_step_remover.py /path/to/my/data/dmsp_ssm_magnetometer_data_file.cdf --modifycdf
```
To show what the algorithm is doing graphically:
```
python ssm_step_remover.py /path/to/my/data/dmsp_ssm_magnetometer_data_file.cdf --showplots
```
If the modifycdf flag is used this tool will modify the CDF file inplace creating two new CDF variables
'DELTA_B_SC_STEPCOR' and 'DELTA_B_APX_STEPCOR',
which correspond to the step corrected magnetic perturbations in spacecraft and Magnetic Apex coordinates.

This algorithm is not fully optmized, but it catches most step discontinuities correctly. It would provide a starting point for the interested researcher.
