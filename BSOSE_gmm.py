import os,sys,gsw
import numpy as np
import netCDF4 as nc
import matplotlib as mpl

import matplotlib.pyplot as plt
from matplotlib import patches as patches
from scipy.io import loadmat
from scipy import interpolate

from datetime import datetime

# run clustering algorithms for T/S, using PCA as dimension reduction
from sklearn import preprocessing#, mixture
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import train_test_split

# import colormaps
# import colormaps
from palettable.colorbrewer.qualitative import Accent_4 as acc
from palettable.colorbrewer.qualitative import Accent_5 as acc5
from palettable.colorbrewer.qualitative import Paired_4 as paired

# ~~~~~~~~~~~~~~~~~
grid_file  = '/data/soccom/GRID_6/grid.nc'
dir_path   = '/data/irosso/data/BSOSE/DFe'
dir_bgt    = '/data/SOSE/SOSE/SO6/ITER122/budgets'
dir_bgt2   = '/data/SOSE/SOSE/SO6/ITER122'
HOMEdir    = '/data/irosso'
plotdir    = os.path.join(HOMEdir,'plots/BSOSE/gmm')

# Compression level for the dimensionality reduction 
maxvar      = 99.9 # in %

# if using T and S combined
combined    = True

# for T/S plots
ss        = np.linspace(33.5,36.5,15)
tt        = np.linspace(-3,25,15)
ss2, tt2  = np.meshgrid(ss,tt)
dens      = gsw.sigma0(ss2,tt2)
csig      = [26.4,26.9,27.2,27.6,27.8]

# list of colors
colS        = ['green','red','royalblue','orange','m','wheat','yellowgreen','gray','pink',
				'lime','royalblue','slateblue','crimson','yellowgreen','peru','mediumpurple',
				'skyblue','greenyellow','lightcoral','teal']
				
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def compute_PCA(Xn,scaler,reducer,train,comp,pr):
	if train:
		# Compute the EOFs from X normalized
		reducer = PCA(n_components=maxvar/100,svd_solver='full')
		reducer.fit(Xn)
	# Reduce the dataset (compute the y):
	Xr     = reducer.transform(Xn) # Here we compute: np.dot(Xn - reducer.mean_, np.transpose(reducer.components_))

	# Variables of the reduced space:
	Nc     = reducer.n_components_ # Number of components retained
	EOFs   = reducer.components_   # [Nc , Nz], the P matrix
	V      = reducer.explained_variance_ratio_ # Explained variance, with 0 to 1 values
	#printXr.shape,Xn.shape
	
	"""
	# compute whole VE for the original data
	VE = 100. * (1- np.nanvar( Xn - Xr)/np.nanvar(Xn))
	#print"Xr size and VE: ",Xr.shape, VE
	
	# compute first 2 modes VE for the original data
	VE = 100. * (1- np.nanvar( Xn - Xr[:2] * EOFs[:2,:])/np.nanvar(Xn))
	#print"Xr size and VE: ",Xr[:2].shape, VE
	"""
	
	# We can also compute EOFs with real units this way:
	S      = np.sqrt(reducer.explained_variance_*Xn.shape[0]) # These are the singular values
	Z      = np.dot(Xn - reducer.mean_, np.transpose(reducer.components_)) # This is simply Xr or the principal components
	Ztilde = Z/np.sqrt(S) # Normalized PCs
	EOFs_real = np.dot(np.transpose(Ztilde),Xn)/Xn.shape[0] # Regression on any collection of profiles
	#printEOFs_real.shape

	# Compute the RMS difference between the reconstructed and original dataset:
	Xn_reconstructed = reducer.inverse_transform(Xr)
	X_reconstructed  = scaler.inverse_transform(Xn_reconstructed)
	rms = np.sqrt(np.mean(np.square(X_reconstructed-Xn),axis=0))

	Xr_3PCAs      = np.dot(Xn, EOFs[:3,:].T)
	Xn_reconstructed_3PCAs = np.dot(Xr_3PCAs, EOFs[:3,:])
		
	# compute whole VE for the original data
	VE = 100. * (1- np.nanvar( Xn - Xn_reconstructed)/np.nanvar(Xn))
	#print"Xr size and VE: ",Xn_reconstructed.shape, VE
	
	# compute first 2 modes VE for the original data
	VE = 100. * (1- np.nanvar( Xn - Xn_reconstructed_3PCAs)/np.nanvar(Xn))
	#print"Xr size and VE: ",Xn_reconstructed_3PCAs .shape, VE
	
	#print"The variance explained by each mode, computed by reducer.explained_variance_ratio_, is ", V 

	##print'The RMS of the difference between the reconstruceted and the original dataset is: ', rms
	#print"\nWe reduced the dimensionality of the problem from 2000 depth levels down to %i PCs\n"%(Nc)
	
	col         = ['green','red','royalblue','orange','m','orange']
	if train:
		fig, ax = plt.subplots(figsize=(7,7))#, dpi=300, facecolor='w', edgecolor='k', sharey='row')
		xl = np.max(np.abs([np.min(EOFs_real),np.max(EOFs)]))
		for ie in range(0,min(3,EOFs.shape[0])):
			plt.plot(np.transpose(EOFs[ie,:]),pr,linestyle='-',c=col[ie],linewidth=1,label="PCA-%i"%(ie+1))
		#ax[iax].set_xlim(1.1*np.array([-xl,xl]))
		if 'Temp' in comp:
			ax.legend(loc=2,fancybox=True)
		ax.set_xlabel('PCAs', fontsize=14)# (no units)')
		ax.set_ylabel('pressure [db]', fontsize=14)
		ax.grid(True)
		ax.set_title('PCA components of %s' %comp,fontsize=16)
		#ax.set_xlim(-0.2,0.2)
		ax.invert_yaxis()
		ax.set_ylim(-900,-300)
		#plt.suptitle('%s' %comp)
	
		outfile = os.path.join(plotdir,'gmm_BSOSE_PCAs_depth_%s_300_900m_3PCAs.png' %comp[:3])
		#printoutfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)
		#.show()
		plt.close()
		
	return [Nc,V,EOFs,EOFs_real,Xr,reducer]	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def run_gmm(Xr,gmm,train,K):
	if train:
		gmm = GMM(n_components=K,\
					covariance_type='full',\
					init_params='kmeans',\
					max_iter=1000,\
					random_state=1000,\
					tol=1e-6)
		gmm.fit(Xr) # Training on reduced data

		# Extract GMM parameters:
		priors = gmm.weights_ # [K,1]
		centers= gmm.means_   # [K,Nc]
		covars = gmm.covariances_ # [K,Nc,Nc] if 'full'

	# Classify the dataset:
	LABELS = gmm.predict(Xr) # [Np,1]
	POST   = gmm.predict_proba(Xr) # [Np,Nc]
	
	return [LABELS, POST,gmm]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def plot_BIC(setF,lfield):
	lowest_bic = np.infty
	bic        = []
	n_components_range = range(1, 30)
	cv_types = ['full']
	for cv_type in cv_types:
		for n_components in n_components_range:
			print('computing gmm with %i components..' %n_components)
			# Fit a Gaussian mixture with EM
			gmm = GMM(n_components=n_components,covariance_type=cv_type)
			gmm.fit(setF)
			bic.append(gmm.bic(setF))
			if bic[-1] < lowest_bic:
				lowest_bic = bic[-1]
				best_gmm = gmm

	bic = np.array(bic)

	# Plot the BIC scores
	plt.figure(figsize=(8, 6))
	plt.plot(n_components_range,bic,'k',linewidth=4,alpha=0.5)
	plt.scatter(n_components_range,bic,c='k',edgecolors='face',s=100,alpha=0.4)
	#plt.scatter(n_components_range[8],bic[8],c='r',s=200,marker='*')
	plt.xlabel('# clusters')
	plt.ylabel('BIC score')
	plt.title(r"GMM BIC score, using the first 2 PCAs of %s" %(lfield))
	plt.grid("on")
	plt.xlim(-0.01,50.01)

	outfile = os.path.join(plotdir,'gmm_BSOSE_BIC_score_%s.png' %(lfield))
	print(outfile)
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def test_small_sample_BIC(PTn,SAn,scalerPT,scalerSA):
	bicV        = np.nan*np.ones((10,29))
	perc75      = len(PTn)*75//100
	for kk in range(10):
		# create a list of indexes of size equal to 75% of PTn (and SAn) (not sure I will have 10 different sets)
		idx     = np.random.randint(len(PTn), size=perc75)
		PTn_tr  = PTn[idx,:]
		SAn_tr  = SAn[idx,:]  
		# reduce the set
		[Nc_PT,V_PT,EOFs_PT,EOFs_real_PT,PT_r,reducerPT]   = compute_PCA(PTn,scalerPT,None,1,'Potential Temperature',RC[idz1+1:idz2+1])
		[Nc_SA,V_SA,EOFs_SA,EOFs_real_SA,SA_r,reducerSA]   = compute_PCA(SAn,scalerSA,None,1,'Practical Salinity',RC[idz1+1:idz2+1])
		# select only the first 2 PCAs of PT and SA to build the array for the gmm algorithm
		# 2 PCAs
		set = np.zeros((PT_r.shape[0],4))
		set[...,0] = PT_r[:,0]
		set[...,1] = PT_r[:,1]
		set[...,2] = SA_r[:,0]
		set[...,3] = SA_r[:,1]

		# BIC score
		lowest_bic = np.infty
		bic        = []
		n_components_range = range(1, 30)
		cv_types = ['full']
		for cv_type in cv_types:
			for n_components in n_components_range:
				# Fit a Gaussian mixture with EM
				gmm = GMM(n_components=n_components,covariance_type=cv_type)
				gmm.fit(set)
				bic.append(gmm.bic(set))
				if bic[-1] < lowest_bic:
					lowest_bic = bic[-1]
					best_gmm = gmm

		bic = np.array(bic)
		bicV[kk,:] = bic

	# Plot the BIC scores
	plt.figure(1,figsize=(8, 6))	
	plt.errorbar(n_components_range,np.nanmean(bicV,0),yerr=np.nanstd(bicV,0),ecolor='m',elinewidth=3, capsize=2)
	plt.scatter(n_components_range,np.nanmean(bicV,0),c='k',marker='o',edgecolors='face',s=10,alpha=0.4)	
	plt.xlabel('# clusters')
	plt.ylabel('BIC score')
	plt.title(r"GMM BIC score, 75% of data")
	plt.grid("on")
	plt.xlim(-0.01,30.01)

	outfile = os.path.join(plotdir,'gmm_BSOSE_BIC_score_training_10samples.png')
	print(outfile)
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def run_gmm_ff(FFm_flat,msk,zfield,K,lfield,colS,bathy,XC,YC,XCmsk,YCmsk,SSHm):	
	# 4. normalize the chopped and flat profiles
	scalerFF   = preprocessing.StandardScaler()
	scalerFF   = scalerFF.fit(FFm_flat[msk,:])
	FFn        = scalerFF.transform(FFm_flat[msk,:])  

	# 5. reduce the dataset using PCAs (putting T and S together? to be decided)
	[Nc_ff,V_ff,EOFs_ff,EOFs_real_ff,ff_r,reducerFF]  = compute_PCA(FFn,scalerFF,None,1,lfield,zfield)

	# 6. Use 2 PCAs
	set_r = np.zeros((ff_r.shape[0],4))
	set_r[...,0] = ff_r[:,0]
	set_r[...,1] = ff_r[:,1]

	# 7. find the number of clusters by computing the BIC score
	#plot_BIC(set_r,lfield) # --> maybe 12?

	# 8. find the clusters --> run the gmm
	[labels,post,gmm]    = run_gmm(set_r,'',1,K)
	
	# 9. extract the optimal parameters
	lambda_k             = gmm.weights_ 		# dimension = n_components = 9
	mean_k               = gmm.means_			# dimension = n_components x n_features = 9 x 4
	cov_k                = gmm.covariances_	    # dimension = n_components x n_features x n_features = 9 x 4 x 4

	# 10. plot maps and means 
	fig,ax   = plt.subplots(figsize=(12,7))
	#im       = ax.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
	ax.contour(XC,YC,bathy,[0,10,20],colors='k')
	ax.set_xlim(XC[0],XC[-1])
	ax.set_ylim(YC[0],YC[-1])
	ax.set_title('Clusters in the Indian sector of the Southern Ocean (BSOSE)',fontsize=16)
	#cax     = fig.add_axes([0.65, 0.15, 0.2, 0.01])
	#cbar    = plt.colorbar(im,cax=cax,orientation='horizontal')
	#cbar.ax.set_title('[km]')
	#cbar.ax.xaxis.set_tick_params(color='k')
	for ii in range(K):
		idx = np.where(labels==ii)[0][:]
		ax.scatter(XCmsk[idx],YCmsk[idx],c=colS[ii],s=5,edgecolor='None',alpha=0.9)
		ax.plot(np.nan,np.nan,c=colS[ii],linewidth=2,label='%i' %(ii+1))
	#ax.scatter(lon_dataSO,lat_dataSO,c=labels,s=100,edgecolor='None',alpha=0.8)
	ax.contour(XC,YC,bathy,[2500,3500],colors='k',linewidths=0.5)
	ax.contour(XC,YC,SSHm,[-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8],colors='w')
	leg = ax.legend(loc=3,ncol=5,fancybox=True,framealpha=0.8)

	outfile = os.path.join(plotdir,'gmm_Indian_Ocean_BSOSE_%i_clusters_map_%s.png' %(K,lfield))
	print(outfile)
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	plt.show()
	
	return [labels,post]

# ~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~~ #
data       = nc.Dataset(grid_file)
XC         = data.variables['XC'][:]
YC         = data.variables['YC'][:]
DXC        = data.variables['DXC'][:]
DYC        = data.variables['DYC'][:]
RC         = data.variables['RC'][:]
DRF        = data.variables['DRF'][:]
bathy      = data.variables['Depth'][:]

idx_1000   = np.max(np.where(RC>=-1000))
x_KP       = np.min(np.where(XC[0,:]>=180))

# use the monthly means for the moment, since I'm going to use the temporal mean below:
fSSH       = os.path.join(dir_bgt2,'bsose_i122_2013to2017_monthly_SSH.nc')
#fTHETA     = os.path.join(dir_bgt,'bsose_i122_2013to2017_MonthlySnapShots_THETA.nc')
fTHETA     = os.path.join(dir_bgt2,'bsose_i122_2013to2017_monthly_Theta.nc')
#fSALT      = os.path.join(dir_bgt,'bsose_i122_2013to2017_MonthlySnapShots_SALT.nc')
fSALT      = os.path.join(dir_bgt2,'bsose_i122_2013to2017_monthly_Salt.nc')
fDFe       = os.path.join(dir_bgt2,'bsose_i122_2013to2017_monthly_Fe.nc')
fO2        = os.path.join(dir_bgt2,'bsose_i122_2013to2017_monthly_O2.nc')
fNO3       = os.path.join(dir_bgt2,'bsose_i122_2013to2017_monthly_NO3.nc')

# 1. I want to work with profiles only between 300 and 900m, as for the Argo profiles --> extract the indexes:
idz1       = np.max(np.where(RC>=-300))
idz2       = np.max(np.where(RC>=-900))

SSH_data   = nc.Dataset(fSSH)
SSH        = SSH_data.variables['ETAN'][:,:,:x_KP]
del SSH_data

PT_data    = nc.Dataset(fTHETA)
PT         = PT_data.variables['THETA'][:,idz1:idz2,:,:x_KP]
del PT_data

SALT_data  = nc.Dataset(fSALT)
SALT       = SALT_data.variables['SALT'][:,idz1:idz2,:,:x_KP]
del SALT_data

# sigma0
CT         = gsw.CT_from_pt(SALT,PT)
s0         = gsw.sigma0(SALT,CT)
sigma0     = np.ma.masked_less(s0,0.)

# temporal means
SSHm       = np.nanmean(SSH,0)
PTm        = np.nanmean(PT,0)
SALTm      = np.nanmean(SALT,0)
sigma0m    = np.nanmean(sigma0,0)

# 2. indexes of bathymetry >= 900 m (otherwise I will have some profiles with NaNs)
XC         = XC[0,:x_KP]
YC         = YC[:,0]
bathy      = bathy[:,:x_KP]
bathy_flat = bathy.flatten()

msk        = np.where(bathy_flat>=900)[0][:]
mskY, mskX = np.where(bathy>=900)

XCmsk      = XC[mskX]
YCmsk      = YC[mskY]

# I still need to extract a training set to reduce the calculation and bias (like one grid point every 3.. something like that..)

# ~~~~~~~~ T and S ~~~~~~~~~ #
if combined:
	# 3a. combine T and S
	Q       = np.zeros((2*(idz2-idz1),PTm.shape[1],PTm.shape[2]),'>f4')
	Q[:idz2-idz1,...] = PTm
	Q[idz2-idz1:,...] = SALTm

	# 3b. flatten the Q
	Q_flat   = Q.reshape(-1,Q.shape[1]*Q.shape[2]).transpose()
	
	# normalize Q
	scalerQ = preprocessing.StandardScaler()
	scalerQ = scalerQ.fit(Q_flat[msk,:])
	Qn      = scalerQ.transform(Q_flat[msk,:])

	# build a new pressure array, with pressure levels repeated (since T and S are concatenated)
	RC2     = np.concatenate([RC[idz1+1:idz2+1]]*2)

	# reduce the set
	[Nc_Q,V_Q,EOFs_Q,EOFs_real_Q,Q_r,reducerQ] = compute_PCA(Qn,scalerQ,None,1,'Q = [T S]',RC2)

	# variance explained by the first 2 components: np.sum(V_Q[:2]) = 0.9854175174759008

	# 2 PCAs
	#set_r = Q_r[:,:2]
	# 3 PCAs
	set_r = Q_r[:,:3]

else:
	# 3. flatten the mean fields (I need samples x features dimensions for the code below):
	PTm_flat   = PTm.reshape(-1,PTm.shape[1]*PTm.shape[2]).transpose()
	SAm_flat   = SALTm.reshape(-1,SALTm.shape[1]*SALTm.shape[2]).transpose()

	# 4. normalize the chopped and flat profiles
	scalerPT   = preprocessing.StandardScaler()
	scalerPT   = scalerPT.fit(PTm_flat[msk,:])
	scalerSA   = preprocessing.StandardScaler()
	scalerSA   = scalerSA.fit(SAm_flat[msk,:])
	PTn        = scalerPT.transform(PTm_flat[msk,:])  
	SAn        = scalerSA.transform(SAm_flat[msk,:])     

	# 5. reduce the dataset using PCAs (putting T and S together? to be decided)
	[Nc_PT,V_PT,EOFs_PT,EOFs_real_PT,PT_r,reducerPT]   = compute_PCA(PTn,scalerPT,None,1,'Potential Temperature',RC[idz1+1:idz2+1])
	[Nc_SA,V_SA,EOFs_SA,EOFs_real_SA,SA_r,reducerSA]   = compute_PCA(SAn,scalerSA,None,1,'Practical Salinity',RC[idz1+1:idz2+1])

	# 6. Use 2 PCAs for both PT and SA
	set_r = np.zeros((PT_r.shape[0],4))
	set_r[...,0] = PT_r[:,0]
	set_r[...,1] = PT_r[:,1]
	set_r[...,2] = SA_r[:,0]
	set_r[...,3] = SA_r[:,1]

# 7. find the number of clusters by computing the BIC score
plot_BIC(set_r,'T and S') # --> maybe 12?
# TO DO!!! test the robustness of the results
#test_small_sample_BIC(PTn,SAn,scalerPT,scalerSA)

# 8. find the clusters --> run the gmm
K = 20
[labels,post,gmm]    = run_gmm(set_r,'',1,K)
	
# 9. extract the optimal parameters
lambda_k                   = gmm.weights_ 		# dimension = n_components = 9
mean_k                     = gmm.means_			# dimension = n_components x n_features = 9 x 4
cov_k                      = gmm.covariances_	# dimension = n_components x n_features x n_features = 9 x 4 x 4

# 10. plot maps and means 
fig,ax   = plt.subplots(figsize=(12,7))
#im       = ax.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
ax.contour(XC,YC,bathy,[0,10,20],colors='k')
ax.set_xlim(XC[0],XC[-1])
ax.set_ylim(YC[0],YC[-1])
ax.set_title('Clusters in the Indian sector of the Southern Ocean (BSOSE)',fontsize=16)
#cax     = fig.add_axes([0.65, 0.15, 0.2, 0.01])
#cbar    = plt.colorbar(im,cax=cax,orientation='horizontal')
#cbar.ax.set_title('[km]')
#cbar.ax.xaxis.set_tick_params(color='k')
for ii in range(K):
	idx = np.where(labels==ii)[0][:]
	ax.scatter(XCmsk[idx],YCmsk[idx],c=colS[ii],s=5,edgecolor='None',alpha=0.9)
	ax.plot(np.nan,np.nan,c=colS[ii],linewidth=2,label='%i' %(ii+1))
#ax.scatter(lon_dataSO,lat_dataSO,c=labels,s=100,edgecolor='None',alpha=0.8)
ax.contour(XC,YC,bathy,[2500,3500],colors='k',linewidths=0.5)
ax.contour(XC,YC,SSHm,[-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8],colors='w')
leg = ax.legend(loc=3,ncol=5,fancybox=True,framealpha=0.8)

outfile = os.path.join(plotdir,'gmm_Indian_Ocean_BSOSE_%i_clusters_map_T_S_combined_3PCAs.png' %K)
print(outfile)
plt.savefig(outfile, bbox_inches='tight',dpi=200)
plt.close()

# check profiles
xtest = np.min(np.where(XC>=60))
ytest = np.min(np.where(YC>=-60))

# full profile
PT_data    = nc.Dataset(fTHETA)
PTfull     = PT_data.variables['THETA'][:,:,ytest,xtest]
PTfullm    = np.nanmean(PTfull,0)
PTfullm    = np.ma.masked_equal(PTfullm,0)

SALT_data  = nc.Dataset(fSALT)
SALTfull   = SALT_data.variables['SALT'][:,:,ytest,xtest]
SAfullm    = np.nanmean(SALTfull,0)
SAfullm    = np.ma.masked_equal(SAfullm,0)

plt.figure()
plt.plot(SAfullm,PTfullm)
cs  = plt.contour(ss2,tt2,dens,csig,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
plt.clabel(cs,inline=0,fontsize=10)
plt.xlabel('Practical salinity [PSU]')
plt.ylabel('Potential temperature [$^{\circ}$C]')
plt.title('%.2fE-%.2fS' %(np.abs(XC[xtest]),np.abs(YC[ytest])))

outfile = os.path.join(plotdir,'Indian_Ocean_BSOSE_diagram_T_S.png')
print(outfile)
plt.savefig(outfile, bbox_inches='tight',dpi=200)
plt.close()


# ~~~~~~~~ BGC ~~~~~~~~~ #
idz1       = np.min(np.where(RC<=0))
idz2       = np.max(np.where(RC>=-200))

DFe_data   = nc.Dataset(fDFe)
DFe        = DFe_data.variables['TRAC06'][:,idz1:idz2,:,:x_KP]
del DFe_data

O2_data    = nc.Dataset(fO2)
O2         = O2_data.variables['TRAC03'][:,idz1:idz2,:,:x_KP]
del O2_data

NO3_data   = nc.Dataset(fNO3)
NO3        = NO3_data.variables['TRAC04'][:,idz1:idz2,:,:x_KP]
del NO3_data

DFem       = np.nanmean(DFe,0)
O2m        = np.nanmean(O2,0)
NO3m       = np.nanmean(NO3,0)

msk        = np.where(bathy_flat>=200)[0][:]
mskY, mskX = np.where(bathy>=200)

XCmsk      = XC[mskX]
YCmsk      = YC[mskY]

DFe_flat   = DFem.reshape(-1,DFem.shape[1]*DFem.shape[2]).transpose()
O2_flat    = O2m.reshape(-1,DFem.shape[1]*DFem.shape[2]).transpose()
NO3_flat   = NO3m.reshape(-1,DFem.shape[1]*DFem.shape[2]).transpose()

K = 20
# DFe
run_gmm_ff(DFe_flat,msk,RC[idz1+1:idz2+1],K,'Dissolved Iron',colS,bathy,XC,YC,XCmsk,YCmsk,SSHm)	

# NO3
run_gmm_ff(NO3_flat,msk,RC[idz1+1:idz2+1],K,'Nitrate',colS,bathy,XC,YC,XCmsk,YCmsk,SSHm)	

# O2
run_gmm_ff(O2_flat,msk,RC[idz1+1:idz2+1],K,'Oxygen',colS,bathy,XC,YC,XCmsk,YCmsk,SSHm)	

