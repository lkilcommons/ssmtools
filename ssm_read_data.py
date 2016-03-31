import numpy as np
import matplotlib
import matplotlib.pyplot as pp
from spacepy import pycdf
import os,time,datetime
#from scipy.ndimage.filters import gaussian_filter1d

#Python datetime to day of year
def datetime2doy(dt): 
	return dt.timetuple().tm_yday + dt.hour/24. + dt.minute/24./60. + dt.second/86400. + dt.microsecond/86400./1e6

#Python datetime to second of day
def datetime2sod(dt):
	return (dt-datetime.datetime.combine(dt.date(),datetime.time(0))).total_seconds() #Microsecond precision

def ymdhms2jd(year,mon,day,hr,mn,sc):
	#Takes UTC ymdhms time and returns julian date
	#FIXME: Add leap second support
	leapsecond = False
	if year < 1900:
		raise ValueError('Year must be 4 digit year')
	t1 = 367.*year

	t2 = int(7.*(year+int((mon+9.)/12.))/4.)
	t3 = int(275.*mon/9.)
	t4 = day + 1721013.5
	if not leapsecond:
		t5 = ((sc/60.+mn)/60+hr)/24
	else:
		t5 = ((sc/61.+mn)/60+hr)/24	 	
	#print t1,t2,t3,t4,t5
	return t1-t2+t3+t4+t5

#Julian Date (reasonably precise, but w/o leap seconds)
def datetime2jd(dt):
	return ymdhms2jd(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)

#J2000 Epoch (remember that J2000 is defined as starting at NOON on 1-1-2000 not 0 UT)
def datetime2j2000(dt):
	return ymdhms2jd(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)-ymdhms2jd(2000,1,1,12,0,0)

dup_last = lambda x: np.concatenate((x.flatten(),x.flatten()[-1:]),axis=1)


class ssm_cdf_reader(object):
	def __init__(self,ssmcdffn):
		self.cdffn = ssmcdffn # CDF file 
		self.cdf = pycdf.CDF(ssmcdffn) # Open CDF
		#Make a name for if we have anything to write
		#leaf = str(self.cdf.attrs['Logical_file_id']).replace('ssm','ssm_modified')+'.cdf'
		#self.modified_cdffn = os.path.join(,leaf)
		
		#Get UTC second of day
		self.ut = np.array([datetime2sod(dt) for dt in self.cdf['Epoch'][:].flatten().tolist()]).flatten()
		self.lat = self.cdf['SC_APEX_LAT'][:].flatten()
		self.lon = self.cdf['SC_APEX_LON'][:].flatten()
		self.glat = self.cdf['SC_GEOCENTRIC_LAT'][:].flatten()
		self.glon = self.cdf['SC_GEOCENTRIC_LON'][:].flatten()
		self.R = self.cdf['SC_GEOCENTRIC_R'][:].flatten()*1000.
		self.mlt = self.cdf['SC_APEX_MLT'][:].flatten()
		self.oi = self.cdf['ORBIT_INDEX'][:].flatten()
		self.n_orbits = int(np.max(np.abs(self.oi)))
		self.dBd1 = self.cdf['DELTA_B_APX'][:,0].flatten()
		self.dBd2 = self.cdf['DELTA_B_APX'][:,1].flatten()
		self.dBd3 = self.cdf['DELTA_B_APX'][:,2].flatten()
		#SSM coordinates x - down, y - along, z - across-right
		self.dB_along = self.cdf['DELTA_B_SC'][:,1].flatten()
		self.dB_across = -1*self.cdf['DELTA_B_SC'][:,2].flatten() # Across left
		self.dB_up = -1*self.cdf['DELTA_B_SC'][:,0].flatten()

	def get_orbit_mask(self,index,hemisphere='N',remove_nan=True,min_lat=0.,allow_irregular_dt=True):
		"""
		Get an index array into the main arrays that selects a particular orbit
		if index is 'all' then selects all data in the specified hemisphere
		"""
		oi = self.oi
		dt = dup_last(np.diff(self.ut)).flatten()

		hemisign = 1. if hemisphere=='N' else -1.
		if hemisphere=='NS':
			oi = np.abs(oi)
			in_hemi = np.abs(self.lat)>min_lat
		else:
			if hemisphere == 'N':
				in_hemi = self.lat>hemisign*np.abs(min_lat)
			elif hemisphere == 'S':
				in_hemi = self.lat<hemisign*np.abs(min_lat)

		if isinstance(index,str) and index=='all':
			in_orbit = np.isfinite(oi)
		else:
			in_orbit = oi == hemisign*index 
		
		in_orbit = np.logical_and(in_orbit.flatten(),in_hemi.flatten())

		if remove_nan:
			g = np.logical_and(np.isfinite(self.dBd1),np.isfinite(self.dBd2))
			in_orbit = np.logical_and(in_orbit,g)

		if not allow_irregular_dt:
			#Irregular cadence points can cause very wierd finite difference values
			#so removing them entirely can be useful
			irreg_dt = np.logical_or(dt > 1.01,dt < .99)
			in_orbit = np.logical_and(in_orbit,np.logical_not(irreg_dt))

		return in_orbit

	def get_orbit_data(self,index,coords='spacecraft',**kwargs):
		"""
		Set index to 'all' to get all data
		"""
		in_orbit = self.get_orbit_mask(index,**kwargs)
		if coords == 'apex':
			return self.ut[in_orbit],self.lat[in_orbit],self.lon[in_orbit],self.mlt[in_orbit],self.dBd1[in_orbit],self.dBd2[in_orbit]
		elif coords == 'spacecraft':
			return self.ut[in_orbit],self.glat[in_orbit],self.glon[in_orbit],self.dB_along[in_orbit],self.dB_across[in_orbit],self.dB_up[in_orbit]
	
	def iter_all_orbit_indices(self,**kwargs):
		"""
		Simply return the index of each orbit
		in the order in which that occured
		(handles which hemisphere is the first pass of the day
		ambiguity)
		"""
		for i in range(self.n_orbits):
			first_north = np.flatnonzero(self.oi==i)[0]
			first_south = np.flatnonzero(self.oi==-1*i)[0]
			if first_north < first_south:
				hemi_order = ['N','S']
			else:
				hemi_order = ['S','N']
				
			for hemi in hemi_order:
				yield i,hemi



