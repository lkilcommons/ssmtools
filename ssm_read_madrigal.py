import madrigalWeb.madrigalWeb
import h5py, os
import numpy as np
import matplotlib.pyplot as pp
import matplotlib as mpl
from geospacepy import dmspcdf_tools,special_datetime,satplottools,omnireader
import datetime,sys,itertools,re
import urllib2, urllib, re, os, datetime
import seaborn as sns

# constants
user_fullname = 'Liam Kilcommons'
user_email = 'liam.kilcommons@colorado.edu'
user_affiliation = 'University of Colorado, Boulder'
madrigalUrl = 'http://cedar.openmadrigal.org'

def download_mfr(dmsp_number,year,month,day,outpath='/tmp',redownload=False):
	"""
	Trolls through the NOAA site to find the appropriate raw DMSP file and download it
	For SSM: Downloads MFR
	"""

	dmspnumstr = 'f%.2d' % (dmsp_number)
	yearstr = '%.4d' % (year) #4 digit zero-padded year
	monthstr = '%.2d' % (month) #2 digit zero-padded month
	daystr = '%.2d' % (day) #2 digit zero-padded day of year
	monthnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
	instrument = 'SSM'

	doy = datetime.datetime(year,month,day).timetuple().tm_yday
	doystr = '%.3d' % (doy) #3 digit zero-padded day of year

	if outpath is None:
		if not os.path.exists(outpath):
			print("Output directory did not exist. Creating %s\n" % (outpath))
			os.makedirs(outpath)
	
	noaaurl = 'http://satdat.ngdc.noaa.gov/dmsp/data/'+dmspnumstr+'/'+instrument.lower()+'/'+yearstr+'/'+monthstr+'/'
	
	filename_link_res =     ['href="PS.CKGWC_SC.U_DI.A_GP.SSMXX-F'+'%.2d' % (dmsp_number)+'-R99990-B9999090-APSM_AR.GLOBAL_DD.'+\
				yearstr+monthstr+daystr+'_TP.\d+-\d+_DF.MFR(.gz)?"',
			'href="mfr'+yearstr+doystr+'\d+_'+yearstr+doystr+'\d+_\d+.dat(.gz)?"',
			'href="m'+dmspnumstr[1:]+yearstr[2:]+doystr+'.dat(.gz)?"'] #\d matches digits, + is one or more times
	
	fpaths = [] 
	#I think the files are always gzipped on the NOAA site, but I'm making .gz optional for the regexp match just in case
	for filename_link_re in filename_link_res:
		fname_pattern = re.compile(filename_link_re, re.IGNORECASE)
		print('Connecting to %s' % (noaaurl))
		reqnoaa = urllib2.Request(noaaurl)
		responsenoaa = urllib2.urlopen(reqnoaa)
		fname_link = None
		for line in responsenoaa.readlines():
			#print str(line)
			#Find the first matching substring in this line
			fname_match = fname_pattern.search(line)
			#Returns a match object if found, None otherwise
			if fname_match is not None:
				fname_link = fname_match.group() #Group returns the substring matching the pattern
				break
		if fname_link is None:
			print("No file matching %s found!" % (filename_link_re))
			fpaths.append(None)
		else:
			print("Matched: "+filename_link_re)
			
			fname = fname_link[6:] #trim off leading href="
			fname = fname[:-1] #trim off trailing "
			fileurl = noaaurl+fname
			print("Attempting to download filename %s\n" % (fname))

			fpath = os.path.join(outpath,fname)
			if os.path.exists(fpath) and not redownload:
				print("File %s exists, will not download." % (fpath))
			elif os.path.exists(fpath) and redownload:
				print("File %s exists, will overwrite." % (fpath))
				urllib.urlretrieve(fileurl, fpath)
			else:
				urllib.urlretrieve(fileurl, fpath)
			fpaths.append(fpath)

	return fpaths

def get_noaa_mfr_data(satnums,dt,localdir='/tmp'):
	all_data = {}
	for satnum in satnums:
		fpath = download_mfr(satnum,dt.year,dt.month,dt.day,outpath=localdir)[0]
		if fpath is None:
			continue
		# Header:
		#  APSM output Magnetic Field Record file from the DMSP SSM instrument.  Version 4.16 - v12FEB2010  health 99
		#
		# Units are degrees (east), Km, nT.  X is down, Y is velocity, Z is orbit normal.  Ephemeris is geocentric terrestrial.
		#
		#    0 Date-time              2 MissnID      4 Lat               6 Alt  8 1st/Dif      10 TotY         12 CalDate         14 Meas-ModY     16 ModDate
		#                  1 Sec-of-day   3 Avg/Smp           5 Lon         7 EphSrc  9 TotX          11 TotZ          13 Meas-ModX     15 Meas-ModZ     17 ModTyp
		
		converters = {j:lambda x: float(x) for j in range(18)}
		converters[3] = lambda x: 1. if x == 'A' else 0.
		converters[7] = lambda x: 1. if x == 'I' else 0.
		converters[8] = lambda x: 1. if x == 'D' else 0.
		converters[17] = lambda x: 1. if x == 'I' else 0.
		print fpath
		data = np.genfromtxt(fpath,converters=converters,skip_header=6)
		
		#sod = data[:,1]
		#glat = data[:,4]
		#glon = data[:,5]
		#dbx,dby,dbz = data[:,13],data[:,14],data[:,15]
		all_data[satnum] = data

	return all_data

def get_madrigal_ssm_hdf5(satnums,dt,localdir='/tmp',redownload=False):
	"""
	Downloads madrigal SSM+SSIES files to localdir if they are not already present,
	otherwise just returns their filenames
	"""
	if not isinstance(satnums,list):
		satnums = [satnums]

	local_hdf5_files = {satnum:os.path.join(localdir,gen_madrigal_dmsp_localfile(satnum,dt)) for satnum in satnums}
	
	all_exist = True
	for satnum,fn in local_hdf5_files.iteritems():
		if os.path.exists(fn):
			print "%s exists" % (fn)
		else:
			print "%s does not exist" % (fn)
		all_exist = all_exist and os.path.exists(fn)

	if all_exist:
		print "All Madrigal files for %s already downloaded" % (str(satnums))
		return local_hdf5_files
	else:
		testData = madrigalWeb.madrigalWeb.MadrigalData(madrigalUrl)

		#Don't need to keep doing this
		instList = testData.getAllInstruments()
		dmsp_inst = [inst for inst in instList if 'Defense Meteorological Satellite Program' in inst.name][0]
		print dmsp_inst
		#DMSP code is 8100
		#dmsp_code = 8100

		expList = testData.getExperiments(dmsp_inst.code,dt.year,dt.month,dt.day,0,0,1,
										 dt.year,dt.month,dt.day,23,59,59)

		fileList = testData.getExperimentFiles(expList[0].id)

		downloaded_files = {}
		h5fs = {}
		for f in fileList:
			if 'magnetometer' in f.kindatdesc:
				#Check if it's local
				if f.category == 1:
					satkey = f.kindatdesc.split()[0] #first word is F{satnum}
					satkey = int(satkey.split('F')[-1]) #Only satnum
					if satkey in satnums:
						local_fname = local_hdf5_files[satkey]
						if not os.path.exists(local_fname) or redownload:
							
							result = testData.downloadFile(f.name,
														   local_fname, 
														   user_fullname, user_email, user_affiliation, "hdf5")
							print "Downloading %s result %s key is %s" % (local_fname,str(result),satkey)
							downloaded_files[satkey]=local_fname
						else:
							result = None
							print "Already exists %s, result %s, key is %s" % (local_fname,str(result),satkey)
							downloaded_files[satkey]=local_fname
					else:
						print "Not downloading %s because not in satnums input argument" % (f.kindatdesc)

		#if not return_filenames:
		#    h5fs = {satnum: h5py.File(fname,'r') for (satnum,fname) in downloaded_files.iteritems()}
		#    return h5fs
		#else:
		return downloaded_files

def gen_madrigal_dmsp_localfile(satnum,dt):
	return 'madrigal_ssm_ssies_f%.2d_%s.h5' % (satnum,dt.strftime('%Y%m%d'))

def get_madrigal_ssm_data(satnums,dt,localdir='/tmp'):
	"""
	Parses out magnetometer data as a numpy array from strangely formatted
	madrigal HDF5 files
	"""
	if not isinstance(satnums,list):
		satnums = [satnums]

	h5filenames = get_madrigal_ssm_hdf5(satnums,dt,localdir=localdir)
	all_data = {}
	for satnum in satnums:
		
		if satnum not in h5filenames:
			print "No hdf5 file for spacecraft F%.2d on %s" % (dt.strftime('%Y-%m-%d'))
			continue

		with h5py.File(h5filenames[satnum],'r') as h5f:

			data_params = ['YEAR','MONTH','DAY','HOUR','MIN','SEC',
					'GDLAT', 'GLON', 'GDALT', 'SAT_ID',
					'MLT', 'MLAT', 'MLONG',
					'BD', 'B_FORWARD', 'B_PERP',
					'DIFF_BD', 'DIFF_B_FOR', 'DIFF_B_PERP']

			#Split out the data
			h5_tablecolnames = h5f['Metadata']['Data Parameters'][:]
			h5_datatable = h5f['Data']['Table Layout'][:]

			#Make dictionary to translate between column names and numbers
			param_name_to_colnum = {param[0]:colind for colind,param in enumerate(h5_tablecolnames)}

			colinds = [param_name_to_colnum[param_name] for param_name in data_params]

			nrows = 86400
			ncols = len(data_params)
			data = np.zeros((nrows,ncols))
			data.fill(np.nan)

			maxrowind = 0
			for rowind,row in enumerate(h5_datatable):
				data[rowind,:] = [row[colind] for colind in colinds]
				maxrowind=rowind
				if rowind == 0:
					for icolind,colind in enumerate(colinds):
						print "First value from column %d, name %s is %f" % (colind,data_params[icolind],row[colind])   

			all_data[satnum] = data[:maxrowind,:]

	return all_data,data_params

def madrigal_ssm_geo_generator(satnums,dt,localdir='/tmp'):
	code = 'mad'
	if not isinstance(satnums,list):
		satnums = [satnums]

	all_data,data_params = get_madrigal_ssm_data(satnums,dt,localdir=localdir)
	for satnum in all_data:
		data = all_data[satnum]
		sod = data[:,3]*3600.+data[:,4]*60.+data[:,5]
		glat,glon = data[:,6],data[:,7]
		#mlat_mad = data[:,11]
		dbx,dby,dbz = data[:,-3]*1.0e9,data[:,-2]*1.0e9,data[:,-1]*1.0e9

		inday = np.logical_and(data[:,2]==float(dt.day),np.isfinite(data[:,2]))
		print data[:,2]
		print "Found %d extra points from a day other than %s" % (np.count_nonzero(np.logical_not(inday)),dt.strftime('%Y%m%d'))

		yield code,satnum,dt,sod[inday],glat[inday],glon[inday],dbx[inday],dby[inday],dbz[inday]

def cdf_ssm_geo_generator(satnums,dt,localdir=None):
	code = 'cdf'
	for satnum in satnums:
		cdf = dmspcdf_tools.get_cdf(satnum,dt.year,dt.month,dt.day,'ssm')
		sod = special_datetime.datetimearr2sod(cdf['Epoch'][:]).flatten()
		glat = cdf['SC_GEOCENTRIC_LAT'][:]
		glon = cdf['SC_GEOCENTRIC_LON'][:]
		#mlat = cdf['SC_AACGM_LAT'][:]
		db = cdf['DELTA_B_SC'][:]
		dbx,dby,dbz = db[:,0],db[:,1],db[:,2]
		#db_noaa = cdf['DELTA_B_SC_ORIG'][:]
		#dbx_noaa,dby_noaa,dbz_noaa = db_noaa[:,0],db_noaa[:,1],db_noaa[:,2]
		yield code,satnum,dt,sod,glat,glon,dbx,dby,dbz

def noaa_ssm_geo_generator(satnums,dt,localdir='/tmp'):
	code = 'noaa'
	if not isinstance(satnums,list):
		satnums = [satnums]

	all_data = get_noaa_mfr_data(satnums,dt,localdir=localdir)
	for satnum in all_data:
		data = all_data[satnum]
		sod = data[:,1]
		glat = data[:,4]
		glon = data[:,5]
		dbx,dby,dbz = data[:,13],data[:,14],data[:,15]
		yield code,satnum,dt,sod,glat,glon,dbx,dby,dbz

def ensure_hdf5_group_structure(h5f,code,satnum,dt):
	dt_group = dt.strftime('%Y%m%d')
	satnum_group = str(satnum)
	code_group = code

	if dt_group in h5f:
		if satnum_group in h5f[dt_group]:
			if code_group in h5f[dt_group][satnum_group]:
				raise RuntimeError('Data for DMSP %s on %s type %s already exists in HDF5 results' % (satnum_group,dt_group,code_group))
			else:
				h5f[dt_group][satnum_group].create_group(code_group)
		else:
			h5f[dt_group].create_group(satnum_group)
			h5f[dt_group][satnum_group].create_group(code_group)
	else:
		h5f.create_group(dt_group)
		h5f[dt_group].create_group(satnum_group)
		h5f[dt_group][satnum_group].create_group(code_group)
	return h5f[dt_group][satnum_group][code_group]

def ssm_geo_to_passdata(h5f,ssm_geo_generator):
	code,satnum,dt,sod,glat,glon,dbx,dby,dbz = next(ssm_geo_generator)
	print("Adding Pass Data to HDF5 file for %s %s %s" % (str(code),str(satnum),dt.strftime('%Y%m%d')))

	h5group = ensure_hdf5_group_structure(h5f,code,satnum,dt)

	finite_inds = np.flatnonzero(np.isfinite(glat))
	finglat = glat[np.isfinite(glat)]
	xings = satplottools.simple_passes(finglat)
	xings = finite_inds[xings]

	xing_sod = sod[xings]
	xing_glat = glat[xings]
	xing_glon = glon[xings]
	xing_dbx,xing_dby,xing_dbz = dbx[xings],dby[xings],dbz[xings]
	delta = 10
	xing_dbx_av,xing_dby_av,xing_dbz_av = np.zeros_like(xing_dbx),np.zeros_like(xing_dby),np.zeros_like(xing_dbz)
	for j,xing in enumerate(xings):
		xing_dbx_av[j] = np.nanmean(dbx[xing-delta:xing+delta])
		xing_dby_av[j] = np.nanmean(dby[xing-delta:xing+delta])
		xing_dbz_av[j] = np.nanmean(dbz[xing-delta:xing+delta])
	
	xing_db = np.column_stack((xing_dbx,xing_dby,xing_dbz))
	xing_db_av = np.column_stack((xing_dbx_av,xing_dby_av,xing_dbz_av))

	h5group.attrs['satnum'] = satnum
	h5group.attrs['code'] = str(code)
	h5group.attrs['jd'] = special_datetime.datetime2jd(dt)

	h5group.create_dataset('sod',data=xing_sod)
	h5group.create_dataset('glat',data=xing_glat)
	h5group.create_dataset('glon',data=xing_glon)
	h5group.create_dataset('dbx',data=xing_dbx)
	h5group.create_dataset('dbxav',data=xing_dbx_av)
	h5group.create_dataset('dby',data=xing_dby)
	h5group.create_dataset('dbyav',data=xing_dby_av)
	h5group.create_dataset('dbz',data=xing_dbz)
	h5group.create_dataset('dbzav',data=xing_dbz_av)

def ssm_store_xings(satnums,dts,datadir='/home/liamk/mirror/ssm',clobber=True):
	if not os.path.exists(datadir):
		print "Making data directory %s" % (datadir)
		os.makedirs(datadir)
	h5fn = os.path.join(datadir,'dmsp_ssm_comparison.h5')
	if os.path.exists(h5fn) and clobber:
		os.remove(h5fn)
	with h5py.File(h5fn,'w') as h5f:
		for dt in dts:
			gens = {}
			gens['noaa'] = noaa_ssm_geo_generator(satnums,dt,localdir=datadir)
			gens['cdf'] = cdf_ssm_geo_generator(satnums,dt,localdir=datadir)
			gens['mad'] = madrigal_ssm_geo_generator(satnums,dt,localdir=datadir)
			#Consume each generator
			for code in ['noaa','cdf','mad']:
				gen = gens[code]
				ifiles = 0
				while True:
					try:
						ssm_geo_to_passdata(h5f,gen)
					except StopIteration:
						print "Done with %s, read %d files" % (code,ifiles)
						break
					except:
						raise
					ifiles+=1
	return h5fn

def match_xings(sod1,sod2,tol=70):
	sod2to1 = np.zeros_like(sod1,dtype=int)
	sod2to1.fill(-999)
	for isod,sod in enumerate(sod1.flatten().tolist()):
		imin = np.nanargmin(np.abs(sod2-sod)) 
		if np.abs(sod2[imin]-sod) <= tol:
			sod2to1[isod] = imin
			print "Matched pass %d at UT sec %f with %d at %f" % (isod,sod,imin,sod2[imin])
		else:
			print "Failed to match pass %d at UT sec %f, closest match was %d at %f" % (isod,sod,imin,sod2[imin])
		
	return sod2to1

def ssm_compare_xings(h5fn,csvfn,outdir='/tmp'):
	
	all_data = []
	with open(csvfn,'w') as f:
		with h5py.File(h5fn,'r') as h5f:
			for dt_group in h5f:
				for satnum_group in h5f[dt_group]:
					if 'cdf' not in h5f[dt_group][satnum_group]:    
						print("No CDF data for %s %s, can't do comparisons" % (dt_group,satnum_group))
						continue
				
					h5group = h5f[dt_group][satnum_group]
					satday_data = []
					
					h5vars = ['sod','glat','glon','dbx','dby','dbz','dbxav','dbyav','dbzav']

					sod_cdf = h5group['cdf']['sod'][:]
					jd_cdf = sod_cdf/86400. + float(h5group['cdf'].attrs['jd'])
					
					satday_data.append(jd_cdf)

					satday_data.extend([h5group['cdf'][h5var][:] for h5var in h5vars])
					
					n_cdf_passes = len(sod_cdf.flatten())
					for code in ['noaa','mad']:
						if code in h5group:
							sod = h5group[code]['sod'][:]
							othertocdf = match_xings(sod_cdf,sod)
							for h5var in h5vars:
								var_other = h5group[code][h5var][:]
								var_cdf = np.zeros_like(sod_cdf)
								var_cdf.fill(np.nan)
								var_cdf[othertocdf!=-999] = var_other[othertocdf[othertocdf!=-999]]
								satday_data.append(var_cdf)
						else:
							for h5var in h5vars:
								var_cdf = np.zeros_like(sod_cdf)
								var_cdf.fill(np.nan)
								satday_data.append(var_cdf)
					
					satday_data = np.column_stack(satday_data)
					print("Writing %d crossings for sat %s date %s to file %s" % (n_cdf_passes,satnum_group,dt_group,csvfn))
					np.savetxt(f,satday_data)					
					
					#plot_comparison_data(satday_data,outfile='ssm_comparison_%s_%s.png' % (satnum_group,dt_group),
					#					outdir=outdir,offset=len(h5vars))

					all_data.append(satday_data)

		all_data = np.row_stack(all_data)
		plot_comparison_data(all_data,outdir=outdir)

def plot_comparison_data(all_data,outfile='ssm_comparison.png',outdir='/tmp',offset=9):
	from matplotlib.ticker import FuncFormatter,MaxNLocator
		
	fig = pp.figure(figsize=(6,6))

	gs = mpl.gridspec.GridSpec(7,2)
	a1 = pp.subplot(gs[:2,0])
	a2 = pp.subplot(gs[:2,1])
	a3 = fig.add_subplot(gs[3,:])
	a4 = fig.add_subplot(gs[4,:])
	a5 = fig.add_subplot(gs[5,:])
	a6 = fig.add_subplot(gs[6,:])

	gs.update(hspace=.35,wspace=.35)
	
	axletter_x,axletter_y = -.13,1.0
	axletter_xh,axletter_yh = -.33,1.15
	axlabel_x,axlabel_y = 0.,1.0	
	
	textkwargs = {'fontweight':'bold'}

	jd_cdf = all_data[:,0]

	db_inds = np.array([4,5,6])
	dbav_inds = np.array([7,8,9])

	sod_diff_noaa =  all_data[:,1]-all_data[:,(1+offset)]
	sod_diff_mad =  all_data[:,1]-all_data[:,(1+2*offset)]

	glat_cdf,glon_cdf = all_data[:,2],all_data[:,3]
	glat_noaa,glon_noaa =  all_data[:,(2+offset)],all_data[:,(3+offset)]
	glat_mad,glon_mad =  all_data[:,(2+2*offset)],all_data[:,(3+2*offset)]
	
	glon_cdf[glon_cdf<0] += 360.
	glon_cdf[glon_cdf>360.] -= 360.
	 
	glon_noaa[glon_noaa<0] += 360. 
	glon_noaa[glon_noaa>360.] -= 360.
	
	glon_mad[glon_mad<0.] += 360. 
	glon_mad[glon_mad>360.] -= 360. 
	
	glon_diff_noaa = glon_cdf-glon_noaa
	glon_diff_mad = glon_cdf-glon_mad

	gc_diff_noaa = satplottools.greatCircleDist(np.column_stack((glat_cdf,glon_cdf)),
														np.column_stack((glat_noaa,glon_noaa)),
														lonorlt='lon')
	gc_diff_mad = satplottools.greatCircleDist(np.column_stack((glat_cdf,glon_cdf)),
														np.column_stack((glat_mad,glon_mad)),
														lonorlt='lon')

	magnitude = lambda x: np.sqrt(x[:,0]**2+x[:,1]**2+x[:,2]**2)
	
	db_cdf,db_noaa,db_mad = all_data[:,dbav_inds],all_data[:,(dbav_inds+offset)],all_data[:,(dbav_inds+2*offset)]
	db_diff_noaa =  db_cdf-db_noaa
	db_diff_mad = db_cdf-db_mad

	db_diff_noaa_mag = magnitude(db_diff_noaa)
	db_diff_mad_mag = magnitude(db_diff_mad)
	
	#for a in [a1,a2,a3,a4,a5,a6]:
	#	a.cla()

	sod_bins = np.arange(-2.,2.,.1)
	sns.distplot(sod_diff_noaa[np.isfinite(sod_diff_noaa)],kde=False,bins=sod_bins,color='g',label='CDF-MFR',ax=a1)
	sns.distplot(sod_diff_mad[np.isfinite(sod_diff_mad)],kde=False,bins=sod_bins,color='r',label='CDF-MAD',ax=a1)
	#a1.plot(jd_cdf,sod_diff_noaa,'go')
	#a1.plot(jd_cdf,sod_diff_mad,'ro')
	#a1.set_yscale('log')
	a1.set_ylabel('Equator Crossings')
	a1.legend()
	a1.set_xlabel('Seconds')
	a1.text(axletter_xh,axletter_yh, 'a)',transform=a1.transAxes,**textkwargs)

	R = 6371.2+850.
	gc_bins = np.arange(0,.1,.01)*R/180.*np.pi
	sns.distplot(gc_diff_noaa[np.isfinite(gc_diff_noaa)]*R,kde=False,bins=gc_bins,color='g',label='CDF-MFR',ax=a2)
	sns.distplot(gc_diff_mad[np.isfinite(gc_diff_mad)]*R,kde=False,bins=gc_bins,color='r',label='CDF-MAD',ax=a2)
	#a2.plot(jd_cdf,glon_diff_noaa,'go')
	#a2.plot(jd_cdf,glon_diff_mad,'ro')
	#a2.set_yscale('log')
	a2.legend()
	a2.set_ylabel('Equator Crossings')
	a2.set_xlabel('Km')
	a2.text(axletter_xh,axletter_yh, 'b)',transform=a2.transAxes,**textkwargs)

	"""
	a3.plot(jd_cdf,glon_cdf,'bo')
	a3.plot(jd_cdf,glon_noaa,'go')
	a3.plot(jd_cdf,glon_mad,'ro')
	#a2.set_yscale('log')
	a3.set_ylabel('Degrees')
	a3.set_title('Longitude at Equator Crossing')

	a4.plot(jd_cdf,glat_cdf,'bo')
	a4.plot(jd_cdf,glat_noaa,'go')
	a4.plot(jd_cdf,glat_mad,'ro')
	#a2.set_yscale('log')
	a4.set_ylabel('Degrees')
	a4.set_title('Latitude at Equator Crossing')
	"""
	#alpha=.8
	a3.plot(jd_cdf,db_mad[:,0],'r^',label='MAD')
	a3.plot(jd_cdf,db_noaa[:,0],'g.',label='MFR')
	a3.plot(jd_cdf,db_cdf[:,0],'b.',label='CDF')
	a3.set_title('Magnetic Perturbation @ GEO Equator\n',fontweight='bold')
	a3.text(axletter_x,axletter_y, 'c)',
		transform=a3.transAxes,**textkwargs)
	a3.text(axlabel_x,axlabel_y, 'X (Down) [nT]',
		transform=a3.transAxes,**textkwargs)
	
	a4.plot(jd_cdf,db_mad[:,1],'r^',label='MAD')
	a4.plot(jd_cdf,db_noaa[:,1],'g.',label='MFR')
	a4.plot(jd_cdf,db_cdf[:,1],'b.',label='CDF')
	a4.text(axletter_x,axletter_y, 'd)',
		transform=a4.transAxes,**textkwargs)
	a4.text(axlabel_x,axlabel_y, 'Y (Along) [nT]',
		transform=a4.transAxes,**textkwargs)
	
	#a4.set_title('Y (Along-Track)')

	a5.plot(jd_cdf,db_mad[:,2],'r^',label='MAD')
	a5.plot(jd_cdf,db_noaa[:,2],'g.',label='MFR')
	a5.plot(jd_cdf,db_cdf[:,2],'b.',label='CDF')
	a5.text(axletter_x,axletter_y, 'e)',
		transform=a5.transAxes,**textkwargs)
	a5.text(axlabel_x,axlabel_y, 'Z (Across) [nT]',
		transform=a5.transAxes,**textkwargs)
	
	#a5.set_title('Z (Across-Track)')
	
	a3.legend(bbox_to_anchor=(0.,.8, 1., .102),ncol=3,loc=4)
	for a in [a3,a4,a5]:
		a.xaxis.set_ticklabels([])
		a.yaxis.set_major_locator(MaxNLocator(5))
		#a.set_ylim([-250.,250.])
		#a.set_xlim((jd_cdf[0],jd_cdf[-1]))
	#a5.set_xlabel('Time of Equator Crossing (from CDF data)')

	startdt,enddt = special_datetime.jd2datetime(jd_cdf[0]),special_datetime.jd2datetime(jd_cdf[-1])
	oi = omnireader.omni_interval(startdt,enddt,'hourly')
	dt_omni = oi['Epoch']
	jd_omni = special_datetime.datetimearr2jd(oi['Epoch'])
	dst = oi['DST']
	a6.plot(jd_omni,dst,'m.')
	a6.text(axletter_x,axletter_y, 'f)',
		transform=a6.transAxes,**textkwargs)
	a6.text(axlabel_x,axlabel_y, 'DST [nT]',
		transform=a6.transAxes,**textkwargs)
	#a6.set_title('Hourly OMNIWeb DST',fontweight='bold',ha='right')

	jd2str = lambda x,pos: special_datetime.jd2datetime(x).strftime('%m/%d\n%H:%M')
	a6.xaxis.set_major_formatter(FuncFormatter(jd2str))
	a6.set_xlabel('Date & Universal Time (2010)',fontweight='bold')

	fig.savefig(os.path.join(outdir,outfile),dpi=300.)
				
if __name__ == '__main__':
	satnums = [16,17,18]
	dts = [datetime.datetime(2010,5,27),datetime.datetime(2010,5,28),
	datetime.datetime(2010,5,29),datetime.datetime(2010,5,30),
	datetime.datetime(2010,5,31),datetime.datetime(2010,6,1)]
	
	outdir = '/home/liamk/mirror/ssm'
	#h5fn = ssm_store_xings(satnums,dts,datadir=outdir,clobber=True)
	csvfn = os.path.join(outdir,'ssm_compare_result.csv')
	h5fn = os.path.join(outdir,'dmsp_ssm_comparison.h5')
	ssm_compare_xings(h5fn,csvfn,outdir=outdir)

#print h5fs