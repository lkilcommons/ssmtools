# DMSP SSM NASA CDF Tools

This repository includes tools and information about the data from the Magnetometer (SSM) instrument 
aboard the Defense Meteorolgy Satellite Program spacecraft.

Specificially, it is a companion to the DMSP SSM CDF files which were prepared as part of NASA grant #NNX13AG07G

You can get the data from the following locations: 

* [CDAWeb Virtual Observatory](http://cdaweb.gsfc.nasa.gov/istp_public/)
* [National Oceanic and Atmospheric Administration (NOAA NCEI)](http://satdat.ngdc.noaa.gov/dmsp/)

## Read the [Friendly Manual](DMSPSpaceWxSSJSSMSSIESATBDandUsersManual_v1_1.pdf)
It's included is this repository. It has a list of all of the variables in the CDF files.
It will also tell you what to watch out for in terms of warts of the data. We've done our best to clean this data, but no dataset is perfect. Knowing the caveats will save you time.

To read additional details about the instrument and our files, visit the [DMSP NOAA NCEI (Formerly NGDC) Documentation](http://satdat.ngdc.noaa.gov/dmsp/docs/)

# Rules of the Road
This software is associated with an *in press* publication. A citation will be added to this readme when it is published.

If you write a paper using this software, please acknowledge us or cite the above paper.

# Software

This software was written to simplify reading DMSP SSM CDF files in Python. Reading it should also help students and researchers understand what is in this data, and how to use it.

These tools are written in the Python language and require the following special libraries in addition to numpy, matplotlib and other tools included in the requirements of the installation script.

* [NASA CDF library](http://cdf.gsfc.nasa.gov/html/sw_and_docs.html)
* [spacepy](http://spacepy.lanl.gov/index.shtml)

These packages provide the tools nessecary to read the NASA CDF format data available from CDAWeb.

# Tools currently in the repo

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
