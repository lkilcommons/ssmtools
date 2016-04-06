import numpy as np
import matplotlib
import matplotlib.pyplot as pp
from spacepy import pycdf
import os,time,argparse
#from scipy.ndimage.filters import gaussian_filter1d
import ssm_read_data

angbetween = lambda v1,v2: np.arccos((v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])/(np.sqrt(v1[0]**2+v1[1]**2+v1[2]**2)*np.sqrt(v2[0]**2+v2[1]**2+v2[2]**2)))

class ssm_step_remover(object):
	"""
	Attempt to remove steps using Welches/Student's T test
	"""
	def __init__(self,ssmcdffn,plot_correction=False,modifycdf=False,validation_plots=False,reader=None):
		
		#Option to load prexisting reader
		if reader is None:
			self.reader = ssm_read_data.ssm_cdf_reader(ssmcdffn)
			self.cdffn = ssmcdffn
		else:
			print "Warning: Ignoring call to use cdffile %s, and using cdf reader instance passed as kwargs['reader'] instead." % (ssmcdffn)
			self.reader = reader
			self.cdffn = reader.cdffn
		self.plot_correction = plot_correction
		self.modifycdf = modifycdf
		self.validation_plots = validation_plots
		#Make copies so we can compare
		self.orig_dB_along,self.orig_dB_across,self.orig_dB_up = self.reader.dB_along.copy(),self.reader.dB_across.copy(),self.reader.dB_up.copy()
		self.orig_dBd1,self.orig_dBd2,self.orig_dBd3 = self.reader.dBd1.copy(),self.reader.dBd2.copy(),self.reader.dBd3.copy()
 		if self.plot_correction:
			self.pauselen = 1
			self.f = pp.figure()
			self.a1 = self.f.add_subplot(111)
			pp.ion()

	def plot_difference(self,oi,hemi):
		ms=3
		inpass = self.reader.get_orbit_mask(oi,hemisphere=hemi,remove_nan=False)

		t = self.reader.ut[inpass]
		dB_along,dB_across,dB_up = self.reader.dB_along[inpass],self.reader.dB_across[inpass],self.reader.dB_up[inpass]
		odB_along,odB_across,odB_up = self.orig_dB_along[inpass],self.orig_dB_across[inpass],self.orig_dB_up[inpass]
		
		dBd1,dBd2,dBd3 = self.reader.dBd1[inpass],self.reader.dBd2[inpass],self.reader.dBd3[inpass]
		odBd1,odBd2,odBd3 = self.orig_dBd1[inpass],self.orig_dBd2[inpass],self.orig_dBd3[inpass]
		
		#Spacecraft coordinates
		fsc = pp.figure(figsize=(11,8))
		asc1 = fsc.add_subplot(311)
		asc2 = fsc.add_subplot(312)
		asc3 = fsc.add_subplot(313)

		asc1.plot(t,odB_along,'b.',label='Along, Before',ms=ms)
		asc1.plot(t,dB_along,'r.',label='Along, After',ms=ms)
		asc1.legend()

		asc2.plot(t,odB_across,'b.',label='Across, Before',ms=ms)
		asc2.plot(t,dB_across,'r.',label='Across, After',ms=ms)
		asc2.legend()

		asc3.plot(t,odB_up,'b.',label='Up, Before',ms=ms)
		asc3.plot(t,dB_up,'r.',label='Up, After',ms=ms)
		asc3.legend()

		asc1.set_title('Magnetic Pertrubations Before and After Step Removal')
		
		fapx = pp.figure(figsize=(11,8))
		apx1 = fapx.add_subplot(311)
		apx2 = fapx.add_subplot(312)
		apx3 = fapx.add_subplot(313)
		
		apx1.plot(t,odBd1,'b.',label='dBd1, Before')
		apx1.plot(t,dBd1,'r.',label='dBd1, After')
		apx1.legend()

		apx2.plot(t,odBd2,'b.',label='dBd2, Before')
		apx2.plot(t,dBd2,'r.',label='dBd2, After')
		apx2.legend()

		apx3.plot(t,odBd3,'b.',label='dBd3, Before')
		apx3.plot(t,dBd3,'r.',label='dBd3, After')
		apx3.legend()

		for ax in [asc1,asc2,asc3,apx1,apx2,apx3]:
			ax.set_ylabel('nT')

		apx1.set_title('Apex Coordinates Magnetic Pertrubations Before and After Step Removal')
		
		parts = os.path.split(self.cdffn)
		filename = parts[-1]
		path = parts[0]
		stem = os.path.splitext(filename)[0]

		plotfolder = os.path.join(path,stem+'_validation')
		if not os.path.exists(plotfolder):
			os.makedirs(os.path.join(plotfolder,'sc'))
			os.makedirs(os.path.join(plotfolder,'apx'))
			
		fsc.suptitle('%s, Orbit %d, Hemisphere: %s' % (filename,oi,hemi))
		fapx.suptitle('%s, Orbit %d, Hemisphere: %s' % (filename,oi,hemi))

		fsc.savefig(os.path.join(plotfolder,'sc','%s_orbit%.2d_%s_sc.png' % (stem,oi,hemi)))
		fapx.savefig(os.path.join(plotfolder,'apx','%s_orbit%.2d_%s_apx.png' % (stem,oi,hemi)))

	def repair_pass(self,oi,hemi):
		"""
		oi - orbit index
		hemi - 'N' or 'S', hemisphere
		"""
		ms = 3
		inpass = self.reader.get_orbit_mask(oi,hemisphere=hemi,remove_nan=False)

		passinds = np.flatnonzero(inpass)

		t = self.reader.ut[inpass]
		dB_along,dB_across,dB_up = self.reader.dB_along[inpass],self.reader.dB_across[inpass],self.reader.dB_up[inpass]
		dB_d1,dB_d2,dB_d3 = self.reader.dBd1[inpass],self.reader.dBd2[inpass],self.reader.dBd3[inpass]

		#Create the scalar field perturbation
		dB = np.sqrt(dB_along**2+dB_across**2+dB_up**2)
		
		if np.count_nonzero(np.isfinite(dB)) < 100.:
			print "Less than 100 finite values in this pass (orbit %d, %s hemisphere)...skipping" % (oi,hemi)
			return

		#Determine what is valid data
		g = np.isfinite(dB)
		ginds = np.flatnonzero(g)
		
		#Remove step-up discontinuities
		done = False
	
		origdB = dB.copy()
		istep = 0
		max_iters = 15 
		jump_inds = None
		while not done:

			if self.plot_correction:
				#Plot where the jump was found
				self.a1.cla()
				#self.a1.plot(t,dB,'k.',label='dB',ms=ms)
				self.a1.plot(t,dB_along,'m.',label='dB_along',ms=ms)
				self.a1.plot(t,dB_across,'g.',label='dB_across',ms=ms)
				self.a1.plot(t,dB_up,'b.',label='dB_up',ms=ms)
				self.a1.legend()
				self.a1.set_title("Step Detection Iteration %d, Orbit %d, %s Hemisphere" % (istep,oi,
						hemi))
				self.f.canvas.draw()
				pp.pause(self.pauselen)

			#Detect steps using the field-aligned component of apex coordinates perturbations
			#theoretically there will be no perturbations in this component
			jumped,done = self.find_one_step_t_test(dB_d3)
			
			#Apply the correction to each component
			if not done:
				
				jump_inds = np.flatnonzero(jumped)
			
				#Get the corrections for this particular jump
				this_d1_correction = self.compute_correction(dB_d1,jump_inds)
				this_d2_correction = self.compute_correction(dB_d2,jump_inds)
				this_fa_correction = self.compute_correction(dB_d3,jump_inds)
				this_along_correction = self.compute_correction(dB_along,jump_inds)
				this_across_correction = self.compute_correction(dB_across,jump_inds)
				this_up_correction = self.compute_correction(dB_up,jump_inds)
				
				jump_size_d3 = this_fa_correction[np.nanargmax(np.abs(this_fa_correction))]
				jump_size_along = np.nanmean(this_along_correction)
				jump_size_across = np.nanmean(this_across_correction)
				jump_size_up = np.nanmean(this_up_correction)
				
				print "(Orbit %d)(Hemisphere: %s)(Iter %d): Removing %.3f minute long %.3f nT (field-aligned direction) step." % (oi,
					hemi,istep,
					(t[jump_inds[-1]]-t[jump_inds[0]])/60.,
					jump_size_d3)
				
				#Apply the correction to THIS PASS's data
				
				dB_d1 = dB_d1 - this_d1_correction
				dB_d2 = dB_d2 - this_d2_correction
				dB_d3 = dB_d3 - this_fa_correction
				
				dB_along = dB_along - this_along_correction
				dB_across = dB_across - this_across_correction
				dB_up = dB_up - this_up_correction

				#Show Pass Correction
				if self.plot_correction:
					#Plot and draw
					self.a1.cla()
					#self.a1.plot(t,origdB,'k.',label='Scalar Pertrb',ms=ms)
					#self.a1.plot(t,dB,'r.',label='Corr Scalar by %.3f nT' % (jump_size_scalar),ms=ms)
					self.a1.plot(t,dB_along,'m.',label='Corr dB_along by %.3fnT' % (jump_size_along),ms=ms)
					self.a1.plot(t,dB_across,'g.',label='Corr dB_across by %.3fnT' % (jump_size_across),ms=ms)
					self.a1.plot(t,dB_up,'b.',label='Corr dB_up by %.3fnT' % (jump_size_up),ms=ms)
					self.a1.axvspan(t[jump_inds[0]],t[jump_inds[-1]],alpha=.3,color='red')
					self.a1.legend()
					self.a1.set_title("Step Removal Iteration %d, Orbit %d, %s Hemisphere" % (istep,oi,
						hemi))
					self.f.canvas.draw()
					pp.pause(self.pauselen) 
				
				#Apply the correction to the WHOLE DAY's data
				self.reader.dB_along[passinds] = self.reader.dB_along[passinds] - this_along_correction
				self.reader.dB_across[passinds] = self.reader.dB_across[passinds] - this_across_correction
				self.reader.dB_up[passinds] = self.reader.dB_up[passinds] - this_up_correction
				
				#Apply the correction to the WHOLE DAY's data
				self.reader.dBd1[passinds] = self.reader.dBd1[passinds] - this_d1_correction
				self.reader.dBd2[passinds] = self.reader.dBd2[passinds] - this_d2_correction
				self.reader.dBd3[passinds] = self.reader.dBd3[passinds] - this_fa_correction
				istep+=1
				#Prevent possible infinite loop
				if istep >= max_iters:
					done = True

	def repair_all_passes(self):
		
		for oi,hemi in self.reader.iter_all_orbit_indices():
			self.repair_pass(oi,hemi)
			if self.validation_plots:
				self.plot_difference(oi,hemi)

		#We are done, update the data in the CDF unless we are doing a dry run
		#Recall that SSM coordinates are a little counter-intuitive because the instrument boom is on the side
		#of the spacecraft
		#SSM coordinates x - down, y - along, z - across-right
		if self.modifycdf:
			#Create the final results array
			DELTA_B_SC = np.column_stack((-1*self.dB_up,self.dB_along,-1*self.dB_across))
			DELTA_B_APX = np.column_stack((self.dB_d1,self.dB_d2,self.dB_d3))
			
			self.reader.cdf.readonly(False)
			self.reader.cdf['DELTA_B_SC_STEPCOR'] = DELTA_B_SC
			self.reader.cdf['DELTA_B_APX_STEPCOR'] = DELTA_B_SC
			self.reader.cdf.save()

	def compute_correction(self,dB,jumpinds,mode='interpolate'):
		"""
		Determines how to correct baseline for jump up and corresponding jump down
		Gist is that you determine how much the jump up was relative to the mean value before the jump
		and correspondly, how much the jump back down was relative the the mean value after the jump.

		dB is magnetic perturbation with NaN's NOT removed
		jumpinds is the indices of all points in dB between the jump up and jump down

		Returns a correction array of length dB, but with zeros outside the range of the jump
		The amount of the correction is linearly interpolated between the amount of the jump up
		and the amount of the jump down
		"""
		#Get change amount (start of jump region)
		deltadB_start =  np.nanmean(dB[jumpinds[0]+1:jumpinds[0]+10]) - np.nanmean(dB[jumpinds[0]-10:jumpinds[0]-1])
		
		#Get change amount (end of jump region)
		deltadB_end = np.nanmean(dB[jumpinds[-1]-10:jumpinds[-1]-1]) - np.nanmean(dB[jumpinds[-1]+1:jumpinds[-1]+10])

		#print "Start of step: %3.f nT" % (deltadB_start)
		#print "End of step: %3.f nT" % (deltadB_end)
		
		correction = np.zeros_like(dB)
		if mode == 'interpolate':
			#Compute difference in start and end deltas
			se_diff = deltadB_end - deltadB_start

			#Compute slope of linear interpolation
			slope = se_diff/len(jumpinds)

			x = jumpinds-jumpinds[0]+1
			correction[jumpinds] = slope*x + deltadB_start
		elif mode == 'smallest':
			#Just use the smallest of the two possible corrections on the theory that it's better
			#to undercorrect instead of overcorrect
			correction[jumpinds] = deltadB_start if deltadB_start > deltadB_end else deltadB_end

		return correction

	def find_one_step_t_test(self,x,window_width=40):
		"""Attempt to find step in some perturbation data by applying the student-t test to see if 
		one half of a sliding window is distributed differently than the other half, indicating that there
		is a change in the mean.

		Note:
			Is resiliant to NaN in x
		"""

		jumped = np.zeros_like(x,dtype=bool)

		#g  = np.flatnonzero(np.isfinite(x))
		#x = x[g]
		half_width = int(window_width/2)+1
		ts,dmeans = np.zeros_like(x),np.zeros_like(x)
		jumps,jumpsizes = [],[]
		i = half_width+1 
		
		while i + half_width < len(x):
			wind_x = x[i-half_width:i+half_width]
			
			#Don't do computation if more than half points in window are NaN
			if np.count_nonzero(np.isfinite(wind_x)) < window_width/2:
				ts[i] = np.nan
				dmeans[i] = np.nan
				i+=1
				continue
			else:
				left_half = wind_x[:half_width-1]
				right_half = wind_x[half_width+1:]
				#Compute the T statistic
				Xbarl,Xbarr = np.nanmean(left_half),np.nanmean(right_half)
				sl,sr = np.nanstd(left_half),np.nanstd(right_half)
				#Have to be careful, because nanmean and nanstd implicitly reduce
				#N, the sample size if there are missing values
				Nl,Nr = np.count_nonzero(np.isfinite(left_half)),np.count_nonzero(np.isfinite(right_half))
				DeltaXbar = Xbarr-Xbarl
				#Welch's T statistic
				t = -1*DeltaXbar/np.sqrt(sl**2/Nl+sr**2/Nr)
				#Results
				ts[i] = t
				dmeans[i] = DeltaXbar
				i+=1

		#This termination condition is a bit arbitrary. We
		#could do better if we knew the approximate minimum size of the jump
		
		if np.count_nonzero(ts) < window_width:
			#Just return all false and done if there isn't any data 
			return jumped,True

		#Find the index of the worst down and up steps
		isplit_up = np.nanargmin(ts)
		isplit_down = np.nanargmax(ts)
		
		if isplit_up > isplit_down:
			x_down = np.abs(np.nanmean(x[isplit_down-5:isplit_down-1])) # Mean of x before down
			x_up = np.abs(np.nanmean(x[isplit_up+1:isplit_up+5])) # Mean of x after up
			injump = np.arange(isplit_down,isplit_up)
		elif isplit_down > isplit_up:
			x_up = np.abs(np.nanmean(x[isplit_up-5:isplit_up-1])) # Mean of x before up
			x_down = np.abs(np.nanmean(x[isplit_down+1:isplit_down+5])) # Mean of x after down
			injump = np.arange(isplit_up,isplit_down)

		up_unambigous = np.abs(ts[isplit_up]) > 3*x_up
		down_unambigous = np.abs(ts[isplit_down]) > 3*x_down

	 	if up_unambigous and down_unambigous:
	 		jumped[injump] = True
	 		done = False
	 	else:
	 		done = True
	
		if self.plot_correction:
			self.a1.cla()
			t = np.arange(len(x))
			self.a1.plot(t,x,'b.',label='Original',ms=3)
			#self.a1.plot(t,xcorr,'g.',label='Roughly Corrected',ms=3)
			self.a1.plot(t,ts,'r.',label='t-Statistic',ms=3)
			self.a1.plot(t,dmeans,'k.',label='Window Mean Diff',ms=3)
			self.a1.set_title("Welch's t-statistic step detection (%d point window)" % (window_width))
			self.a1.axvspan(t[injump[0]],t[injump[-1]],alpha=.3,color='red')
			self.a1.plot(t[isplit_down],ts[isplit_down],'go')
			self.a1.plot(t[isplit_up],ts[isplit_up],'go')
			self.f.canvas.draw()
			self.a1.legend()
			pp.pause(self.pauselen)    

		return jumped,done

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="DMSP SSM step discontinuite detection and removal")

	parser.add_argument("cdffile", help='Remove step discontinuities from this DMSP SSM CDF file',type=str,default=None)
	parser.add_argument("--showplots",help="Show the step removal in action",action='store_true',default=False)
	parser.add_argument("--modifycdf",help="Modify the CDF (create 2 new variables for corrected apex and spacecraft dB",default=False)
	args = parser.parse_args()

	if not os.path.exists(args.cdffile):
		raise IOError("Sorry, but the specified CDF file %s appears to not exist!" % (args.cdffile))

	remover = ssm_step_remover(args.cdffile,plot_correction=args.showplots,modifycdf=args.modifycdf)
	remover.repair_all_passes()