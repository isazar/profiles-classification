import os,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import netCDF4 as nc

from palettable.colorbrewer.qualitative import Accent_4 as acc
from palettable.colorbrewer.qualitative import Accent_5 as acc5
from palettable.colorbrewer.qualitative import Paired_4 as paired

from load_folders import fold_paths
fold_paths()

# import cm colorbars
sys.path.append(fold_paths.cmocean)
import cm

# import gsw library
sys.path.append(fold_paths.gsw)
import gsw

# import modules for PCM
sys.path.append(fold_paths.classification)
import run_PCM as PCM
import plot_maps

plotdir     = fold_paths.plotdir
ETOPO       = fold_paths.ETOPO
RTopo       = fold_paths.RTopo
floatdir    = fold_paths.floatdir
dirRdm      = fold_paths.dirRdm
folder_argo = fold_paths.folder_argo

# map of clusters
#bathy_data  = [XC,YC,bathy,x1,x2,y1,y2,xtix,xtix_l,ytix,ytix_l]


def plot_clusters(labels,lon,lat,tit,K,bathy_data,plotdir):
	XC       = bathy_data[0]
	YC       = bathy_data[1]
	bathy    = bathy_data[2]
	x1       = bathy_data[3]
	x2       = bathy_data[4]
	y1       = bathy_data[5]
	y2       = bathy_data[6]
	col      = ['r','c','pink','green','b','orange','yellowgreen','gray','magenta','lime','royalblue','slateblue','crimson','yellowgreen','peru']

	fig,ax   = plt.subplots(figsize=(7,7))
	if 'Seals' in plotdir:
		data        = nc.Dataset(RTopo)
		ice_topo    = data.variables['ice_base_topography'][:]
		ax.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[100,500,1000,2000,3000,4000,5000],cmap=cm.deep)
		ax.contour(XC[x1:x2], YC[y1:y2],-1.*ice_topo[y1:y2,x1:x2],[10],colors='k')
	else:
		ax.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
	ax.set_xlim(XC[x1],XC[x2])
	ax.set_ylim(YC[y1],YC[y2])
	ax.set_title('Clusters',fontsize=16)
	for ii in range(K):
		idx = np.where(labels==ii)[0][:]
		ax.scatter(lon[idx],lat[idx],c=col[ii],s=5,edgecolor='None',alpha=0.9)
		ax.plot(np.nan,np.nan,c=col[ii],linewidth=2,label='%i' %(ii+1))
	leg = ax.legend(loc=3,ncol=3,fancybox=True,framealpha=0.8)

	outfile = os.path.join(plotdir,'gmm_clusters_map_%s.png' %(tit))
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	plt.show()

    
def plot_map_prob(lon,lat,labels,post,K,bathy_data,tit,plotdir):
	XC       = bathy_data[0]
	YC       = bathy_data[1]
	bathy    = bathy_data[2]
	x1       = bathy_data[3]
	x2       = bathy_data[4]
	y1       = bathy_data[5]
	y2       = bathy_data[6]
	xtix     = bathy_data[7]
	xtix_l   = bathy_data[8]
	ytix     = bathy_data[9]
	ytix_l   = bathy_data[10]
	# map of clusters with post probab
	fig      = plt.figure(figsize=(25,12))
	for ii in range(K):
		ax       = plt.subplot(3,3,ii+1)
		if 'Seals' in plotdir:
			data        = nc.Dataset(RTopo)
			ice_topo    = data.variables['ice_base_topography'][:]
			ax.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[100,500,1000,2000,3000,4000,5000],cmap="Greys",alpha=0.3)
			ax.contour(XC[x1:x2], YC[y1:y2],-1.*ice_topo[y1:y2,x1:x2],[10],colors='k')
		else:
			ax.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
		ax.set_xlim(XC[x1],XC[x2])
		ax.set_ylim(YC[y1],YC[y2])
		ax.set_title('k = %i' %(ii+1),fontsize=16)
		idx = np.where(labels==ii)[0][:]
		im = ax.scatter(lon[idx],lat[idx],c=post[idx,ii]*100.,s=20,edgecolor='None',alpha=0.7,cmap=acc.mpl_colormap) #acc.mpl_colormap)
		im.set_clim(60,100)
		if ii == K-1:
			cax     = fig.add_axes([0.15, 0.7, 0.1, 0.01])
			cbar    = plt.colorbar(im,cax=cax,orientation='horizontal',extend='min')
			cbar.ax.xaxis.set_tick_params(color='k')
			cbar.set_ticks(np.linspace(60,100,5))  
		ax.set_xticks(xtix)
		ax.set_xticklabels(xtix_l,fontsize=14)
		ax.set_yticks(ytix[::2])
		ax.set_yticklabels(ytix_l[::2],fontsize=14) 

	outfile = os.path.join(plotdir,'gmm_clusters_map_post_%s.png' %(tit))
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	plt.show()
	
	
def plot_histograms(labels,K,post,tit,plotdir):
	cmpR     = mpl.colors.ListedColormap(acc5.mpl_colors)
	prob_tot = []
	bins     = np.arange(0,1.1,0.1)
	bins_mid = (bins[:-1]+bins[1:])/2.
	prob_bin = np.nan*np.ones((len(bins_mid)),'>f4')
	prob_binT= np.nan*np.ones((len(bins_mid)),'>f4')
	colors   = cmpR(np.linspace(0, 1, 5))
	plt.figure(figsize=(15,8))
	for ii in range(K):
		idx = np.where(labels==ii)[0][:]
		for jj,bb in enumerate(bins[:-1]):
			ff  = post[idx,ii]
			idj = np.where(np.logical_and(ff>=bb,ff<bb+0.1))[0][:]
			prob_bin[jj] = np.nansum(ff[idj])
		#print ii+1, np.nansum(prob_bin[:8]/prob_bin[-1])*100
		prob_tot = np.append(prob_tot,post[idx,ii])
		plt.subplot(2,5,ii+2)
		plt.bar(bins_mid*100.,prob_bin/prob_bin[-1],width=0.1*100,color=colors,align='center')		
		plt.grid(True)
		plt.xticks(np.arange(30,120,20))
		plt.title("k = %i" %(ii+1))
		if ii in [6]:
			plt.xlabel('probability [%]',fontsize=14)
		if ii == 4:
			plt.ylabel('normalized number of profiles',fontsize=14)
		plt.xlim(30,100)
	for jj,bb in enumerate(bins[:-1]):
		idj = np.where(np.logical_and(prob_tot>=bb,prob_tot<bb+0.1))[0][:]
		prob_binT[jj] = np.nansum(prob_tot[idj])
	plt.subplot(2,5,1)
	plt.title("total")
	plt.grid(True)
	plt.xticks(np.arange(30,120,20))
	plt.ylabel('normalized number of profiles',fontsize=14)
	plt.bar(bins_mid*100.,prob_binT/prob_binT[-1],width=0.1*100,color='#1D98AE',alpha=0.7,align='center')		
	plt.xlim(30,100)

	outfile = os.path.join(plotdir,'gmm_clusters_prob_hist_%s.png' %(tit))
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	plt.show()
	
	
def plot_fuzzy(K,lon,lat,labels,post,bathy_data,tit,plotdir):
	XC       = bathy_data[0]
	YC       = bathy_data[1]
	bathy    = bathy_data[2]
	x1       = bathy_data[3]
	x2       = bathy_data[4]
	y1       = bathy_data[5]
	y2       = bathy_data[6]
	xtix     = bathy_data[7]
	xtix_l   = bathy_data[8]
	ytix     = bathy_data[9]
	ytix_l   = bathy_data[10]
	cmp      = mpl.colors.ListedColormap(paired.mpl_colors)
	fig      = plt.figure(figsize=(12,7))
	ax       = plt.subplot(111)
	if 'Seals' in plotdir:
		data        = nc.Dataset(RTopo)
		ice_topo    = data.variables['ice_base_topography'][:]
		ax.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[100,500,1000,2000,3000,4000,5000],cmap="Greys",alpha=0.3)
		ax.contour(XC[x1:x2], YC[y1:y2],-1.*ice_topo[y1:y2,x1:x2],[10],colors='k')
	else:
		plt.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20,3000],colors='k')
	for ii in range(K):
		tmp = np.delete(range(K),ii)
		for jj in tmp:
			id_k = np.where(np.logical_and(labels==ii,post[:,ii]<=0.7))[:][0]  
			idx  = np.where(post[id_k,jj]>=0.3)[:][0]
			im=plt.scatter(lon[id_k][idx],lat[id_k][idx],c=post[id_k,ii][idx]*100,s=50,edgecolor='None',alpha=0.8,cmap=cmp)
			im.set_clim(50,70)
			ax.set_xlim(XC[x1],XC[x2])
			ax.set_ylim(YC[y1],YC[y2])
		if ii == K-1:
			cax     = fig.add_axes([0.2, 0.25, 0.1, 0.01])
			cbar    = plt.colorbar(im,cax=cax,orientation='horizontal',extend='min')
			cbar.ax.xaxis.set_tick_params(color='k')
			cbar.set_ticks(np.linspace(50,70,5))  
			cbar.ax.set_title('posterior probability [%]')
			ax.set_xticks(xtix)
			ax.set_xticklabels(xtix_l,fontsize=14)
			ax.set_yticks(ytix[::2])
			ax.set_yticklabels(ytix_l[::2],fontsize=14) 

	outfile = os.path.join(plotdir,'gmm_clusters_map_all_post_fuzzy_%s.png' %(tit))
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	plt.show()
