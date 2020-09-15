import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing#, mixture
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

# Compression level for the dimensionality reduction
maxvar      = 99.9

def compute_PCA(Xn,scaler,reducer,train,comp,pr,plotdir,showplt):
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

    """
    # compute whole VE for the original data
    VE = 100. * (1- np.nanvar( Xn - Xr)/np.nanvar(Xn))
    print "Xr size and VE: ",Xr.shape, VE

    # compute first 2 modes VE for the original data
    VE = 100. * (1- np.nanvar( Xn - Xr[:2] * EOFs[:2,:])/np.nanvar(Xn))
    print "Xr size and VE: ",Xr[:2].shape, VE
    """

    # We can also compute EOFs with real units this way:
    S      = np.sqrt(reducer.explained_variance_*Xn.shape[0]) # These are the singular values
    Z      = np.dot(Xn - reducer.mean_, np.transpose(reducer.components_)) # This is simply Xr or the principal components
    Ztilde = Z/np.sqrt(S) # Normalized PCs
    EOFs_real = np.dot(np.transpose(Ztilde),Xn)/Xn.shape[0] # Regression on any collection of profiles

    # Compute the RMS difference between the reconstructed and original dataset:
    Xn_reconstructed = reducer.inverse_transform(Xr)
    X_reconstructed  = scaler.inverse_transform(Xn_reconstructed)
    rms = np.sqrt(np.mean(np.square(X_reconstructed-Xn),axis=0))

    Xr_2PCAs      = np.dot(Xn, EOFs[:2,:].T)
    Xn_reconstructed_2PCAs = np.dot(Xr_2PCAs, EOFs[:2,:])

    # compute whole VE for the original data
    VE = 100. * (1- np.nanvar( Xn - Xn_reconstructed)/np.nanvar(Xn))
    print "The Variance Explained computed by me for all the components is ", VE

    # compute first 2 modes VE for the original data
    VE = 100. * (1- np.nanvar( Xn - Xn_reconstructed_2PCAs)/np.nanvar(Xn))
    print "The Variance Explained computed by me for the first 2 modes of the original data is ", VE

    print "The variance explained by each mode, computed by reducer.explained_variance_ratio_, is ", V

    #print 'The RMS of the difference between the reconstruceted and the original dataset is: ', rms
    print "\nWe reduced the dimensionality of the problem from 2000 depth levels down to %i PCs\n"%(Nc)

    style = ['-','--','.']
    if train and showplt and 'Q' not in comp :
        fig, ax = plt.subplots(figsize=(7,7))#, dpi=300, facecolor='w', edgecolor='k', sharey='row')
        xl = np.max(np.abs([np.min(EOFs_real),np.max(EOFs)]))
        for ie in range(0,min(2,EOFs.shape[0])):
        	ax.plot(np.transpose(EOFs[ie,:]),pr,linestyle=style[ie],c='k',label="PCA-%i"%(ie+1),linewidth=2)
        #ax[iax].set_xlim(1.1*np.array([-xl,xl]))
        if 'Temp' in comp:
            ax.legend(loc=2,fancybox=True)
        ax.set_xlabel('PCAs', fontsize=14)# (no units)')
        ax.set_ylabel('pressure [db]', fontsize=14)
        ax.grid(True)
        ax.set_title('PCA components of %s' %comp,fontsize=16)
        ax.set_xlim(-0.1,0.1)
        ax.invert_yaxis()
        ax.set_ylim(1000,200)

        outfile = os.path.join(plotdir,'gmm_Indian_Ocean_PCAs_depth_%s_chopped_300_900m.png' %comp[:3])
        plt.savefig(outfile, bbox_inches='tight',dpi=200)
        plt.show()
        plt.close()

    return [Nc,V,EOFs,EOFs_real,Xr,reducer]
    
def plot_BIC(setM,tit,K,plotdir):
    lowest_bic = np.infty
    bic        = []
    n_components_range = range(1, 50)
    cv_types = ['full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GMM(n_components=n_components,covariance_type=cv_type)
            gmm.fit(setM)
            bic.append(gmm.bic(setM))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    plt.plot(n_components_range,bic,'k',linewidth=4,alpha=0.5)
    plt.scatter(n_components_range,bic,c='k',edgecolors='face',s=100,alpha=0.4)
    plt.scatter(n_components_range[K-1],bic[K-1],c='r',s=200,marker='*')
    plt.xlabel('# clusters')
    plt.ylabel('BIC score')
    plt.title(r"GMM BIC score, using the first 2 PCAs of $\theta$ and salinity (training set)")
    plt.grid("on")
    plt.xlim(-0.01,50.01)

    outfile = os.path.join(plotdir,'gmm_Indian_Ocean_BIC_score_training_%s.png' %(tit))
    plt.savefig(outfile, bbox_inches='tight',dpi=200)
    
    
def test_small_sample_BIC(lon,lat,PT,SA,PTn,SAn,PTtr_r,SAtr_r,scalerPT,scalerSA,press,idz1,idz2,plotdir):
    rdm = np.nan*np.ones((10,2650))
    degRdm = 2.5
    ii = 0
    for ix in np.arange(0,180,degRdm):
        for iy in np.arange(-70,-30,degRdm):
            idx = np.where(np.logical_and(np.logical_and(np.logical_and(lon>=ix,lon<ix+degRdm),lat>=iy),lat<iy+degRdm))[0][:]
            try:
                for zz in range(10):
                    tmp = np.random.choice(idx,1)
                    rdm[zz,ii] = tmp
                ii += 1
            except:
                continue
                #print 'no Argo in %.1fiE-%.1fE, %.1fS-%.1fS' %(ix,ix+degRdm,iy,iy+degRdm)

    col         = ['r','c','m','green','b','orange','yellowgreen','gray','pink','lime','royalblue','slateblue','crimson','yellowgreen','peru']
    ## to check that it does produce always the same list, run the code twice, plotting the results on top of each other
    for kk in range(10):
        i1 = np.where(~np.isnan(rdm[kk,:]))[:][0]
        irdm = [int(rr) for rr in rdm[kk,i1]]
        plt.figure(1)
        plt.scatter(lon[irdm],lat[irdm],c=col[kk],s=5,marker='*',edgecolor='None',alpha=0.5)
        plt.title('coverage of randomly chosen locations')
    plt.show()

    bicV        = np.nan*np.ones((10,29))
    for kk in range(10):
        i1 = np.where(~np.isnan(rdm[kk,:]))[:][0]
        irdm = [int(rr) for rr in rdm[kk,i1]]
        PTn_tr  = PTn[irdm,:]
        SAn_tr  = SAn[irdm,:]

        # reduce the training set
        [Nc_PT,V_PT,EOFs_PT,EOFs_real_PT,PTtr_r,reducerPT] = compute_PCA(PTn_tr,scalerPT,None,1,'Potential Temperature',press[idz1:idz2],plotdir,False)
        [Nc_SA,V_SA,EOFs_SA,EOFs_real_SA,SAtr_r,reducerSA] = compute_PCA(SAn_tr,scalerSA,None,1,'Absolute Salinity',press[idz1:idz2],plotdir,False)

        # select only the first 2 PCAs of PT and SA to build the array for the gmm algorithm
        # 2 PCAs
        # training
        setM = np.zeros((PTtr_r.shape[0],4))
        setM[...,0] = PTtr_r[:,0]
        setM[...,1] = PTtr_r[:,1]
        setM[...,2] = SAtr_r[:,0]
        setM[...,3] = SAtr_r[:,1]

        # BIC score
        lowest_bic = np.infty
        bic        = []
        n_components_range = range(1, 30)
        cv_types = ['full']
        for cv_type in cv_types:
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = GMM(n_components=n_components,covariance_type=cv_type)
                gmm.fit(setM)
                bic.append(gmm.bic(setM))
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
    plt.title(r"GMM BIC score, 1% of data")
    plt.grid("on")
    plt.xlim(-0.01,30.01)

    outfile = os.path.join(plotdir,'gmm_Indian_Ocean_BIC_score_training_10samples.png')
    plt.savefig(outfile, bbox_inches='tight',dpi=200)
    plt.show()   
    
    
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