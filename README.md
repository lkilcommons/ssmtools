# ssmtools
DMSP SSM NASA CDF Tools

This repository includes tools and information about the data from the Magnetometer (SSM) instrument 
aboard the Defense Meteorolgy Satellite Program spacecraft.

Specificially, it is a companion to the DMSP SSM CDF files which were prepared as part of NASA grant (#####) 
for inclusion in the [CDAWeb](http://cdaweb.gsfc.nasa.gov/istp_public/) Virtual Observatory.

> These tools are written in the Python language, and it is assumed that users have installed
> [spacepy](http://spacepy.lanl.gov/index.shtml)
> and the [NASA CDF library](http://cdf.gsfc.nasa.gov/html/sw_and_docs.html). 
> These packages provide the tools nessecary to read the CDF format data available from CDAWeb.

To read details about the instrument and our files, visit the [DMSP NOAA NCEI (Formerly NGDC) Documentation](http://satdat.ngdc.noaa.gov/dmsp/docs/)

A draft version of the User Manual including the SSM instrument is also available in the repo. 
This document will tell you what to watch out for so you don't comprimise your analysis, so it's worth reading.

Tools currently in the repo:

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
