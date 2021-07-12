import os, sys
import numpy as np
import netCDF4 as nc

from load_folders import fold_paths
fold_paths()
# import gsw library
sys.path.append(fold_paths.gsw)
import gsw

def IO_argo(floatdir):
    from datetime import datetime,date, timedelta
    programs = os.listdir(floatdir)
    programs = programs[1:]
    # initialize
    tot_fl      = 0
    maxNprof    = 0
    ll          = 0
    ll1         = 0
    l0          = 0
    dateFlNum   = []
    iniDate     = datetime(1950,1,1)
    minDate     = datetime(2019,1,1).toordinal()

    # initialize big arrays
    PT_dataSO   = np.nan*np.ones((822*310,2000),'>f4')
    SA_dataSO   = np.nan*np.ones((822*310,2000),'>f4')
    lon_dataSO  = np.nan*np.ones((822*310),'>f4')
    lat_dataSO  = np.nan*np.ones((822*310),'>f4')
    pr_dataSO   = np.nan*np.ones((822*2000*310),'>f4')
    yySO        = np.zeros((822*310),'int32')
    IDSO        = np.zeros((822*2000*310),'int32')

    for pp in programs:
        SO_floats = os.listdir(os.path.join(floatdir,'%s' %pp))
        for ff_SO in SO_floats:
            print ff_SO
            # @hidden_cell
            SO_prof = [f for f in os.listdir(os.path.join(floatdir,'%s/%s' %(pp,ff_SO))) if 'prof.nc' in f]
            
            tot_fl += 1
            file    = os.path.join(floatdir,'%s/%s/%s' %(pp,ff_SO,SO_prof[0]))
            data    = nc.Dataset(file)
            time    = data.variables['JULD'][:]
            lon     = data.variables['LONGITUDE'][:]
            lat     = data.variables['LATITUDE'][:]
            pressPre= data.variables['PRES_ADJUSTED'][:].transpose()
            tempPre = data.variables['TEMP_ADJUSTED'][:].transpose()
            tempQC  = data.variables['TEMP_ADJUSTED_QC'][:].transpose()
            psaPre   = data.variables['PSAL_ADJUSTED'][:].transpose()
            psaQC    = data.variables['PSAL_ADJUSTED_QC'][:].transpose()
            # have to add the good QC: [1,2,5,8]  (not in ['1','2','5','8'])
            psaPre[np.where(psaQC!='1')] = np.nan
            tempPre[np.where(tempQC!='1')] = np.nan
            # let's get rid of some profiles that are out of the indian sector
            msk1    = np.where(np.logical_and(lon>=0,lon<180))[0][:]
            msk2    = np.where(np.logical_and(lat[msk1]>-70,lat[msk1]<-30))[0][:]

            lon     = lon[msk1][msk2]
            lat     = lat[msk1][msk2]
            time    = time[msk1][msk2]
            pressPre= pressPre[:,msk1[msk2]]
            tempPre = tempPre[:,msk1[msk2]]
            psaPre  = psaPre[:,msk1[msk2]]
            if time[0] + iniDate.toordinal() < minDate:
                minDate = time[0]
            if len(lon) > maxNprof:
                maxNprof = len(lon[~np.isnan(lon)])
                flN      = ff_SO

            # interpolate
            lat       = np.ma.masked_less(lat,-90)
            lon       = np.ma.masked_less(lon,-500)
            lon[lon>360.] = lon[lon>360.]-360.
            Nprof     = np.linspace(1,len(lat),len(lat))
            # turn the variables upside down, to have from the surface to depth and not viceversa
            if any(pressPre[:10,0]>500.):
                pressPre = pressPre[::-1,:]
                psaPre   = psaPre[::-1,:]
                tempPre  = tempPre[::-1,:]
            # interpolate data on vertical grid with 1db of resolution (this is fundamental to then create profile means)
            fields    = [psaPre,tempPre]
            press     = np.nan*np.ones((2000,pressPre.shape[1]))
            for kk in range(press.shape[1]):
                press[:,kk] = np.arange(2,2002,1)
            psa   = np.nan*np.ones((press.shape),'>f4')
            temp  = np.nan*np.ones((press.shape),'>f4')

            for ii,ff in enumerate(fields):
                for nn in range(pressPre.shape[1]):
                    # only use non-nan values, otherwise it doesn't interpolate well
                    try:
                        f1 = ff[:,nn][ff[:,nn].mask==False] #ff[:,nn][~np.isnan(ff[:,nn])]
                        f2 = pressPre[:,nn][ff[:,nn].mask==False]
                    except:
                        f1 = ff[:,nn]
                        f2 = pressPre[:,nn]
                    if len(f1)==0:
                        f1 = ff[:,nn]
                        f2 = pressPre[:,nn]
                    try:
                        sp = interpolate.interp1d(f2[~np.isnan(f1)],f1[~np.isnan(f1)],kind='linear', bounds_error=False, fill_value=np.nan)
                        ff_int = sp(press[:,nn])
                        if ii == 0:
                            psa[:,nn]   = ff_int
                        elif ii == 1:
                            temp[:,nn] = ff_int
                    except:
                    	continue
                        #print 'At profile number %i, the float %s has only 1 record valid'

            # To compute theta, I need absolute salinity [g/kg] from practical salinity (PSS-78) [unitless] and conservative temperature.
            sa = np.nan*np.ones((press.shape),'>f4')
            for kk in range(press.shape[1]):
                sa[:,kk] = gsw.SA_from_SP(psa[:,kk], press[:,0], lon[kk], lat[kk])
            ptemp     = gsw.pt_from_CT(sa, temp)

            # mask out the profiles with :
            msk       = np.where(lat<-1000)
            lat[msk]  = np.nan
            lon[msk]  = np.nan
            sa[msk]   = np.nan
            sa[sa==0.]= np.nan
            ptemp[msk] = np.nan
            ptemp[temp==0.]= np.nan

            # save the nprofiles
            NN        = np.ones((temp.shape),'int32')
            for ii in range(len(Nprof)):
                NN[:,ii]=Nprof[ii]

            lon_dataSO[ll1:ll1+len(lon)]     = lon
            lat_dataSO[ll1:ll1+len(lon)]     = lat
            PT_dataSO[ll:ll+len(sa[0,:]),:]  = ptemp.T
            SA_dataSO[ll:ll+len(sa[0,:]),:]  = sa.T
            floatID                          = int(ff_SO)*np.ones((sa.shape[1]),'int32')
            IDSO[ll:ll+len(sa[0,:])]         = floatID

            # separate seasons
            dateFl   = []
            for dd in time:
                floatDate = iniDate + timedelta(float(dd))
                dateFl.append(floatDate)
                dateFlNum = np.append(dateFlNum,floatDate.toordinal())

            yearsSO   = np.array([int(dd.year) for dd in dateFl])
            yySO[ll:ll+len(sa[0,:])] = yearsSO

            ll  = ll + len(sa[0,:])
            ll1 = ll1 + len(lon)

    # chop away the part of the array with no data
    PT_dataSO  = PT_dataSO[:ll,:]
    SA_dataSO  = SA_dataSO[:ll,:]
    IDSO       = IDSO[:ll]
    lat_dataSO = lat_dataSO[:ll]
    lon_dataSO = lon_dataSO[:ll]
    mmSO       = mmSO[:ll]
    yySO       = yySO[:ll]

    # remove entire columns of NaNs
    idBad  = []
    for ii in range(PT_dataSO.shape[0]):
        f0 = PT_dataSO[ii,:]
        f1 = f0[~np.isnan(f0)]
        if len(f1) == 0:
            idBad.append(ii)
    PT_dataSO  = np.delete(PT_dataSO,idBad,0)
    SA_dataSO  = np.delete(SA_dataSO,idBad,0)
    IDSO       = np.delete(IDSO,idBad,0)
    lon_dataSO = np.delete(lon_dataSO,idBad,0)
    lat_dataSO = np.delete(lat_dataSO,idBad,0)
    mmSO       = np.delete(mmSO,idBad,0)
    yySO       = np.delete(yySO,idBad,0)

    idBad   = []
    # Interpolate again in depth.. for some reason, some profiles have still wholes in the middle:
    for ii in range(PT_dataSO.shape[0]):
        f0 = PT_dataSO[ii,:]
        try:
            sp = interpolate.interp1d(press[~np.isnan(f0),0],f0[~np.isnan(f0)],kind='linear', bounds_error=False, fill_value=np.nan)
            PT_dataSO[ii,:]= sp(press[:,0])
            f0 = SA_dataSO[ii,:]
            sp = interpolate.interp1d(press[~np.isnan(f0),0],f0[~np.isnan(f0)],kind='linear', bounds_error=False, fill_value=np.nan)
            SA_dataSO[ii,:]= sp(press[:,0])
        except:
            idBad.append(ii)
            #print ii, ' has only 1 number'
    PT_dataSO  = np.delete(PT_dataSO,idBad,0)
    SA_dataSO  = np.delete(SA_dataSO,idBad,0)
    IDSO       = np.delete(IDSO,idBad,0)
    lon_dataSO = np.delete(lon_dataSO,idBad,0)
    lat_dataSO = np.delete(lat_dataSO,idBad,0)
    mmSO       = np.delete(mmSO,idBad,0)
    yySO       = np.delete(yySO,idBad,0)
    
    IO_argo.profSO    = [PT_dataSO,SA_dataSO,lon_dataSO,lat_dataSO,IDSO]
    IO_argo.temporalSO= [yySO,mmSO]

    return [IO_argo.profSO,IO_argo.temporalSO]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
  
def load_seals(sealdir,bathy_data):
	from datetime import datetime,date, timedelta
	# initialize
	maxNprof    = 0
	ll          = 0
	ll1         = 0
	l0          = 0
	dateFlNum   = []	
	
	XC          = bathy_data[0]
	YC          = bathy_data[1]
	x1          = bathy_data[3]
	x2          = bathy_data[4]
	y1          = bathy_data[5]
	y2          = bathy_data[6]
	
	# initialize big arrays
	CT_dataSO   = np.nan*np.ones((100*310,1000),'>f4')
	SA_dataSO   = np.nan*np.ones((100*310,1000),'>f4')
	SP_dataSO   = np.nan*np.ones((100*310,1000),'>f4')
	lon_dataSO  = np.nan*np.ones((100*310),'>f4')
	lat_dataSO  = np.nan*np.ones((100*310),'>f4')
	pr_dataSO   = np.nan*np.ones((100*1000*310),'>f4')
	yySO        = np.zeros((100*310),'int32')
	mmSO        = np.zeros((100*310),'int32')
	IDSO        = np.zeros((100*310),'int32')

	data_list = [g for g in os.listdir(seadir) if g.endswith('nc') and not g.startswith('WOD')]
	for ff_SO in data_list:
		#printff_SO
		tot_fl      = 0
		iniDate     = datetime(1950,1,1)
		minDate     = datetime(2019,1,1).toordinal()
		#printff_SO
		file    = os.path.join(sealdir,'%s' %ff_SO)
		# load data
		data    = nc.Dataset(file)
		cruise  = data.variables['PLATFORM_NUMBER'][:]
		if "CTD2020" in ff_SO:
			cruise = np.array(['202020%0.3i' %(int(f)) for f in cruise])
		elif "CTD2019" in ff_SO:
			cruise = np.array(['191919%0.3i' %(int(f)) for f in cruise])
		else:
			cruise = np.array(['%0.9i' %f for f in cruise])
		time    = data.variables['JULD'][:]
		lon     = data.variables['LONGITUDE'][:]
		lat     = data.variables['LATITUDE'][:]
		pressPre= data.variables['PRES_ADJUSTED'][:].transpose()
		tempPre = data.variables['TEMP_ADJUSTED'][:].transpose()
		tempQC  = data.variables['TEMP_ADJUSTED_QC'][:].transpose()
		psaPre   = data.variables['PSAL_ADJUSTED'][:].transpose()
		psaQC    = data.variables['PSAL_ADJUSTED_QC'][:].transpose()
		psaPre[np.where(psaQC=='4')]   = np.nan
		tempPre[np.where(tempQC=='4')] = np.nan
	
		# let's get rid of some profiles that are out of the Amudsen Sea
		tot_fl += 1
		msk1    = np.where(np.logical_and(lon>=XC[x1],lon<XC[x2]))[0][:]
		msk2    = np.where(np.logical_and(lat[msk1]>YC[y1],lat[msk1]<YC[y2]))[0][:]

		lon     = lon[msk1][msk2]
		lat     = lat[msk1][msk2]
		time    = time[msk1][msk2]
		pressPre= pressPre[:,msk1[msk2]]
		tempPre = tempPre[:,msk1[msk2]]
		psaPre  = psaPre[:,msk1[msk2]]
		cruise  = cruise[msk1][msk2]
		if time[0] + iniDate.toordinal() < minDate:
			minDate = time[0]
		if len(lon) > maxNprof:
			maxNprof = len(lon[~np.isnan(lon)])
			flN      = ff_SO
		# remove data below 1000m (there are just so few seal profiles with data below 1000m anyway)
		msk           = np.where(pressPre>1000)
		pressPre[msk] = np.nan
		tempPre[msk]  = np.nan
		psaPre[msk]   = np.nan
		
		# interpolate and prepare the data
		lat     = np.ma.masked_less(lat,-90)
		lon     = np.ma.masked_less(lon,-500)
		lon[lon>360.] = lon[lon>360.]-360.
		Nprof   = np.linspace(1,len(lat),len(lat))
		# turn the variables upside down, to have from the surface to depth and not viceversa
		if any(pressPre[:10,0]>500.):
			pressPre = pressPre[::-1,:]
			psaPre   = psaPre[::-1,:]
			tempPre  = tempPre[::-1,:]
		# interpolate data on vertical grid with 1db of resolution (this is fundamental to then create profile means)
		fields    = [psaPre,tempPre]
		press     = np.nan*np.ones((1000,pressPre.shape[1]))
		for kk in range(press.shape[1]):
			press[:,kk] = np.arange(2,1002,1)
		psa        = np.nan*np.ones((press.shape),'>f4')
		inst_temp  = np.nan*np.ones((press.shape),'>f4')	

		for ii,ff in enumerate(fields):
			for nn in range(pressPre.shape[1]):
				# only use non-nan values, otherwise it doesn't interpolate well
				try:
					f1 = ff[:,nn][ff[:,nn].mask==False] #ff[:,nn][~np.isnan(ff[:,nn])]
					f2 = pressPre[:,nn][ff[:,nn].mask==False]
				except:
					f1 = ff[:,nn]
					f2 = pressPre[:,nn]
				if len(f1)==0:
					f1 = ff[:,nn]
					f2 = pressPre[:,nn]
				try:
					sp = interpolate.interp1d(f2[~np.isnan(f1)],f1[~np.isnan(f1)],kind='linear', bounds_error=False, fill_value=np.nan)
					ff_int = sp(press[:,nn]) 
					if ii == 0:
						psa[:,nn]   = ff_int
					elif ii == 1:
						inst_temp[:,nn] = ff_int
				except:
					print('At profile number %i, the float %s has only 1 record valid: len(f2)=%i' %(nn,ff_SO,len(f2)))
				
		# To compute theta, I need absolute salinity [g/kg] from practical salinity (PSS-78) [unitless] and conservative temperature.
		sa = np.nan*np.ones((press.shape),'>f4') 
		for kk in range(press.shape[1]):
			sa[:,kk] = gsw.SA_from_SP(psa[:,kk], press[:,0], lon[kk], lat[kk])	
		temp     = gsw.CT_from_t(sa, inst_temp, press)
		ptemp    = gsw.pt_from_CT(sa, temp)			

		# mask out the profiles with :
		msk         = np.where(lat<-1000)
		lat[msk]    = np.nan
		lon[msk]    = np.nan
		cruise[msk] = np.nan
		psa[msk]    = np.nan
		sa[msk]     = np.nan
		psa[sa==0.] = np.nan
		sa[sa==0.]  = np.nan
		# use CT instead of PT
		temp[msk]   = np.nan
		temp[temp==0.]= np.nan

		# save the nprofiles
		NN        = np.ones((temp.shape),'int32')
		for ii in range(len(Nprof)):
			NN[:,ii]=Nprof[ii]
	
		lon_dataSO[ll1:ll1+len(lon)]     = lon
		lat_dataSO[ll1:ll1+len(lon)]     = lat
		CT_dataSO[ll:ll+len(sa[0,:]),:]  = temp.T
		SA_dataSO[ll:ll+len(sa[0,:]),:]  = sa.T
		SP_dataSO[ll:ll+len(sa[0,:]),:]  = psa.T
		IDSO[ll1:ll1+len(lon)]           = [int(f) for f in cruise]

		# separate seasons
		dateFl   = []
		for dd in time:
			floatDate = iniDate + timedelta(float(dd))
			dateFl.append(floatDate)
			dateFlNum = np.append(dateFlNum,floatDate.toordinal())

		yearsSO   = np.array([int(dd.year) for dd in dateFl])
		monthsSO  = np.array([int(dd.month) for dd in dateFl])
		SPR       = np.where(np.logical_and(monthsSO >= 9,monthsSO <=11))
		SUM       = np.where(np.logical_or(monthsSO == 12,monthsSO <=2))
		AUT       = np.where(np.logical_and(monthsSO >= 3,monthsSO <=5))
		WIN       = np.where(np.logical_and(monthsSO >= 6,monthsSO <=8))
		mmFlSO    = monthsSO.copy()
		mmFlSO[SPR] = 1
		mmFlSO[SUM]  =2
		mmFlSO[AUT] = 3
		mmFlSO[WIN] = 4	

		mmSO[ll:ll+len(sa[0,:])] = mmFlSO
		yySO[ll:ll+len(sa[0,:])] = yearsSO
	
		ll  = ll + len(sa[0,:])
		ll1 = ll1 + len(lon)

		minArgoDate = timedelta(float(minDate))+iniDate
		minArgoDate.strftime('%Y/%m/%d %H:%M:%S%z')
	# chop away the part of the array with no data
	CT_dataSO  = CT_dataSO[:ll,:]
	SA_dataSO  = SA_dataSO[:ll,:]
	SP_dataSO  = SP_dataSO[:ll,:]
	IDSO       = IDSO[:ll]
	lat_dataSO = lat_dataSO[:ll]
	lon_dataSO = lon_dataSO[:ll]
	mmSO       = mmSO[:ll]
	yySO       = yySO[:ll]

	# remove entire columns of NaNs
	idBad  = []
	for ii in range(CT_dataSO.shape[1]):
		f0 = CT_dataSO[:,ii]
		f1 = f0[~np.isnan(f0)]
		if len(f1) == 0:
			idBad.append(ii)

	CT_dataSO  = np.delete(CT_dataSO,idBad,0)
	SA_dataSO  = np.delete(SA_dataSO,idBad,0)
	SP_dataSO  = np.delete(SP_dataSO,idBad,0)
	IDSO       = np.delete(IDSO,idBad,0)
	lon_dataSO = np.delete(lon_dataSO,idBad,0)
	lat_dataSO = np.delete(lat_dataSO,idBad,0)
	mmSO       = np.delete(mmSO,idBad,0)
	yySO       = np.delete(yySO,idBad,0)

	load_seals.profSO     = [CT_dataSO,SA_dataSO,lon_dataSO,lat_dataSO,IDSO,SP_dataSO]
	load_seals.temporalSO = [yySO,mmSO]
	
	return [load_seals.profSO,load_seals.temporalSO]