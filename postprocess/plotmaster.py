import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cmocean
import os
import pandas as pd
import scipy.signal as sig
from datetime import datetime
try:
    from parallel import simple as sparallel
except ModuleNotFoundError:
    print("parallel.py not found, continue in serial\n")

# Dispersion relationship framed as a root problem
def f(k,w,h,g):
    return g*k*np.tanh(k*h) - w**2 

# Derivative of dispersion relationship with respect to k
def dfdk(k,w,h,g):
    return g*(np.tanh(k*h) + h*k*np.cosh(k*h)**-2)


# determine minimum water depth to meet d/L criteria for shortest period (T=2)
def ldis(T,h,k=1,g=9.81):
    # constants
    errorTolerance = 10**-12
    maxIterations = 20

    w = 2*np.pi/T
    for i in range(maxIterations):
            
        correction = -f(k,w,h,g)/dfdk(k,w,h,g)
        
        k += correction
        
        # Exiting loop if solution is found within specified error tolerance. 
        error = correction/k
        if ( abs(error) < errorTolerance):
            break
            
        l = 2*np.pi/k
    return l

def computeCelerity(depth,wavelength):
    g = 9.81
    relative_depth = depth/wavelength
    if relative_depth > 0.05 and relative_depth < 0.5:
        regime = 'Intermediate water'
        celerity = np.sqrt((g*wavelength*np.tanh((2*np.pi*depth)/wavelength))/(2*np.pi))
    elif relative_depth < 0.05:
        regime = 'Shallow water'
        celerity = np.sqrt(g*depth)
    else:
        regime = 'Unknown'
        celerity = 0
    return celerity, regime

def load_bathy(path,d):
    # init depth arr
    depoutfile = os.path.join(path,"output", "dep.out")  # attempt to find dep.out
    if os.path.exists(depoutfile):  # if dep.out exists, use it to create depth data
        if "binary" in d["FIELD_IO_TYPE"]:
            depth = np.fromfile(depoutfile).reshape((d["Nglob"], d["Mglob"])) * -1
        else:
            depth = np.loadtxt(depoutfile) * -1
    return depth

def load_var(var,opath,fnum,d):
    match var.lower():
        case 'eta' | 'u' | 'v' | 'umean' | 'vmean' | 'etamean' | 'mask':
            var_in = var.lower()
        case 'hsig' | 'hrms' | 'havg':
            var_in = var.capitalize()

    ffile = os.path.join(opath, var_in + '_' + fnum)

    match d['FIELD_IO_TYPE'].lower():
        case 'ascii':
            var_out = np.loadtxt(ffile)
        case 'binary':
            var_out = np.fromfile(ffile).reshape(d["Nglob"],d["Mglob"])
            
    return var_out

def get_station_bathy(simpath,stpath,stfname,station,d):
    dep = load_bathy(simpath,d)
    st = np.loadtxt(os.path.join(stpath,stfname))
    stx = st[:,0]
    sty = st[:,1]
    stdep = [dep[int(sty[n]),int(stx[n])] for n in (station-1)]
    stxm = [stx[n] for n in (station-1)]
    stym = [sty[n] for n in (station-1)]
    return stdep, stxm, stym

def get_station_xdist_1d(stpath,stfname,stations,d):
    st = np.loadtxt(os.path.join(stpath,stfname))
    stx = st[:,0]
    sts = []
    for i in stations:
        sts.append(stx[(i-1)])
    # stx = stx[(i-1 for i in stations)]
    dist = np.diff(sts) * d["DX"]
    return dist    

def readStationsData(outputDir, stations):
    """Read in the desired station(s)."""
    prefix = 'sta_'
    filePrefix = os.path.join(outputDir, prefix)
    stationsDataList = []
    
    for station in stations:
        fileName = filePrefix+'{:0>4d}'.format(station)
        timeSeries = np.loadtxt(fileName)
        stationsDataList.append(timeSeries)

    return stationsDataList


def calculateHsig(f,Spec_density):
    """Calculate Hmo (Hsig) for narrow-banded spectrum."""
    dfreq=f[1]-f[0]
    totalE=np.sum(Spec_density*dfreq)
    Hrms=np.sqrt(totalE*8)
    Hmo = np.sqrt(2.0) * Hrms    
    return Hmo, Hrms, totalE

def plot_PSD(ppath,data,stations,steady,xfactor=16.):
    hdata = []
    tdata = []
    edata = []
    for index, stationData in enumerate(data):
        fig,ax = plt.subplots(figsize=(15,10),facecolor='white')
        timeS = stationData[:,0] # first row in sta_000# is time stamp
        etaS = stationData[:,1] # second row is time series of eta (third and fourth are U and V)

        dt = round(np.diff(timeS).mean(),2)
        shiftby = int(steady / dt) # number of recordings during steady_time

        eta = etaS[shiftby:]
        time = timeS[shiftby:]
        f, Pxx_spec_den = compute_PSD(time, eta)
        pltrange = int(len(f)/xfactor)

        # send to calculate Hsig
        Hmo, Hrms, totalE = calculateHsig(f,Pxx_spec_den)
        Tp = 1/f[np.argmax(Pxx_spec_den[0:pltrange])]

        fnum = '%.4d' %(stations[index])
        ax.plot(f[0:pltrange],Pxx_spec_den[0:pltrange],linewidth=2.0,label='station '+str(fnum)+' with '+r'$H_{sig}$'+'={0:6.4f} m'.format(Hmo))
        ax.grid()
        ax.set_xlabel('frequency $[Hz]$', fontsize=12)
        ax.set_ylabel('Spectral Density $[m^2/Hz]$', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.tick_params(axis='both', which='minor', labelsize=13)

        ax.legend()
        plt.savefig(os.path.join(ppath,'spectra_sta'+str(fnum)+'_st'+str(int(steady))+'.png'), dpi=600)
        plt.close()

        hdata.append(Hmo)
        tdata.append(Tp)
        edata.append(totalE)

    return hdata, tdata, edata

def compare_PSD(snum,ppath, data, stations,steady, labels,xfactor,fnames):
 
    fig,ax = plt.subplots(figsize=(15,10),facecolor='white')
    for index, stationData in enumerate(data):
        timeS = stationData[:,0] # first row in sta_000# is time stamp
        etaS = stationData[:,1] # second row is time series of eta (third and fourth are U and V)

        dt = round(np.diff(timeS).mean(),2)
        shiftby = int(steady / dt) # number of recordings during steady_time

        eta = etaS[shiftby:]
        time = timeS[shiftby:]
        f, Pxx_spec_den = compute_PSD(time, eta)
        Hmo, Hrms, totalE = calculateHsig(f,Pxx_spec_den)
        pltrange = int(len(f)/xfactor)
        
        PSD = np.sqrt(Pxx_spec_den)
        ax.plot(f[0:pltrange],Pxx_spec_den[0:pltrange],linewidth=2.0,label=labels[index]+' with '+r'$H_{sig}$'+'={0:6.4f} m'.format(Hmo))

    ax.grid()
    ax.set_xlabel('frequency $[Hz]$', fontsize=12)
    ax.set_ylabel('Spectral Density $[m^2/Hz]$', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    ax.legend()
    fnum = '%.4d' %(stations[index])
    var01 = ''.join(['_'+fnames[i] for i in range(len(fnames))])
    plt.savefig(os.path.join(ppath,'spectra_sta'+str(fnum)+'_st'+str(int(steady))+var01+'.png'), dpi=600)
#    if len(fnames)==2: plt.savefig(os.path.join(ppath,'spectra_sta'+str(fnum)+'_st'+str(int(steady))+'_'+fnames[0]+'_'+fnames[1]+'.png'), dpi=600)
#    if len(fnames)==3: plt.savefig(os.path.join(ppath,'spectra_sta'+str(fnum)+'_st'+str(int(steady))+'_'+fnames[0]+'_'+fnames[1]+'_'+fnames[2]+'.png'), dpi=600)
    plt.close()
    

def compute_PSD(time,eta):
    fftwindows = [2048,1024,512,256,128,64]
    dt = round(np.diff(time).mean(),2)
    freqS = 1.0/dt # is a sampling rate (Hz)
    twindow = len(eta)
    nFFT = fftwindows[ np.where([twindow > fwin for fwin in fftwindows])[0][0] ]
    nOverlap = nFFT / 2 #1024 #256 #64 #32
    myWindow=np.bartlett(nFFT)        

    f, Pxx_spec_den = sig.welch(x=eta, fs=freqS, window=myWindow,
                                nperseg=nFFT,
                                noverlap=nOverlap,
                                nfft=nFFT,
                                scaling='density')
    return f, Pxx_spec_den
    
def filter_var(var,freq_threshold,which_pass):
    """
    freq_threshold: frequency (Hz) threshold for highpass or lowpass butterworth filter
    which_pass: filter type; lowpass ('lp') or highpass ('hp') accepted
    """
    sos = sig.butter(10,freq_threshold,which_pass,fs=10,output='sos')
    var_filt = sig.sosfilt(sos,var)
    return var_filt

def compute_PSD_filt(time,var,freq,wpass):
    var_filt = filter_var(var,freq,wpass)
    f,psd = compute_PSD(time,var_filt)
    return f, psd


def plot_station_eta(ppath,data,stations,steady):
    for x in range(len(data)):
        fig,ax  = plt.subplots(figsize=(11,4), dpi=600, facecolor="white")
        timeS = data[x][:,0] # first row in sta_000# is time stamp
        etaTS = data[x][:,1] # second row is time series of eta (third and fourth are U and V)
            
        dt = round(np.diff(timeS).mean(),2)
        shiftby = int(steady / dt) # number of recordings during steady_time

        eta = etaTS[shiftby:]
        time = timeS[shiftby:]

        # print figure
        ax.plot(timeS,etaTS)
        ax.plot(time,eta)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('water level (m)')
        ax.grid()
        fnum = '%.4d' %(stations[x])
        fileName = ppath+'/'+'debug_eta_st'+fnum+'.png'
        plt.savefig(fileName, bbox_inches='tight')
        plt.close() 

def plot_station_vel(ppath,data,stations,steady):
    udata = []
    vdata = []
    for x in range(len(data)):
        fig,ax  = plt.subplots(figsize=(11,4), dpi=600, facecolor="white")
        time = data[x][:,0]
        u = data[x][:,2]
        v = data[x][:,3]
            
        dt = round(np.diff(time).mean(),2)
        shiftby = int(steady / dt)
        
        us = u[shiftby:]
        vs = v[shiftby:]
        ts = time[shiftby:]

        R = np.sqrt(us**2 + vs**2)
        theta = np.rad2deg(np.arctan(vs/us))

        # print figure
        ax.plot(ts,us,label='u')
        ax.plot(ts,vs,label='v')
        ax.plot(ts,R,color='k',label='total',alpha=0.5,linestyle='--')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('velocity (m/s)')
        ax.legend()
        ax.grid()
        fnum = '%.4d' %(stations[x])
        fileName = ppath+'/'+'debug_vel_st'+fnum+'.png'
        plt.savefig(fileName, bbox_inches='tight')
        plt.close()     

        udata.append(np.median(us))
        vdata.append(np.median(vs))
    return udata, vdata 

def run_stations(paths,station,spath,sfname,steady,labels,xfactor):
    pnum = len(paths)
    data = []
    for ind,bpath in enumerate(paths):
        opath = os.path.join(bpath,'output')
        ppath = os.path.join(bpath,'postprocessing')
        if not os.path.exists(ppath): os.makedirs(ppath)
        print('running station data: %s' % bpath)

        d = readLOG(bpath)
        sdata = readStationsData(opath,station)
        plot_station_eta(ppath,sdata,station,steady)
        udata, vdata = plot_station_vel(ppath,sdata,station,steady)
        hdata, tdata, edata = plot_PSD(ppath,sdata,station,steady,xfactor)
        stdep,_,_ = get_station_bathy(bpath,spath[ind],sfname[ind],station,d)
        print(stdep)

        fname=os.path.basename(os.path.dirname(ppath))
        funwaveStationPath =os.path.join(ppath,'station_summary_'+fname+'.xlsx')
        df = pd.DataFrame({'Hsig':hdata,'Tp':tdata,'Energy':edata,'Umean':udata,'Vmean':vdata,'Depth':stdep},index=np.arange(1,(len(hdata)+1)))
        df.to_excel(funwaveStationPath)
        #with open(funwaveStationPath,'w') as f:
        #    for row in range(len(sdata)):
        #        f.write(f'{row+1}    {hdata[row]}    {tdata[row]}    {edata[row]}    {udata[row]}    {vdata[row]}    {stdep[row]}\n')
        #f.close()

        data.append(sdata)

    cpath = os.path.commonpath(paths)
    DATA = []
    PATH = []
    for n in range(len(sdata)):
        DATA.append([data[i][n] for i in range(pnum)])
        PATH.append([os.path.basename(paths[i]) for i in range(pnum)])
        compare_PSD(n,cpath,DATA[0],station,steady,labels,xfactor,fnames=PATH)
#        match pnum:
#            case 2: compare_PSD(n,cpath,[data[0][n],data[1][n]],steady,labels,xfactor,fnames=[os.path.basename(paths[0]),os.path.basename(paths[1])])
#            case 3: compare_PSD(n,cpath,[data[0][n],data[1][n],data[2][n]],steady,labels,xfactor,fnames=[os.path.basename(paths[0]),os.path.basename(paths[1]),os.path.basename(paths[2])])

def plot_grid1D(path,field='eta',n_procs=30):
    opath = os.path.join(path,'output')
    ppath = os.path.join(path,'postprocessing')
    if not os.path.exists(ppath): os.makedirs(ppath)

    d = readLOG(path)
    dep = load_bathy(path,d)

    i = np.arange(0, d["Mglob"]*d["DX"], d["DX"])
    j = np.arange(0, d["Nglob"]*d["DY"], d["DY"])
    
    nfile = int(d["TOTAL_TIME"]/d["PLOT_INTV"])

    match field:
        case 'eta':
            try:
                args = list([(n, i, j, opath, ppath, d, dep) for n in range(nfile)])
                sparallel(plot_eta1D, n_procs, args)
            except:
                for n in range(nfile):
                    plot_eta1D(n, i, opath, ppath, d, dep)

def plot_grid2D(path,field,stride,gridtype,cmax,flip=0,n_procs=30,P=50):
    opath = os.path.join(path,'output')
    ppath = os.path.join(path,'postprocessing')
    if not os.path.exists(ppath): os.makedirs(ppath)

    d = readLOG(path)

    i = np.arange(0, d["Mglob"]*d["DX"], d["DX"])
    j = np.arange(0, d["Nglob"]*d["DY"], d["DY"])

    i = i[::stride]
    j = j[::stride]
    
    nfile = int(d["TOTAL_TIME"]/d["PLOT_INTV"])

    match field:
        case 'eta':
            try:
                args = list([(n, i, j, opath, ppath, d, gridtype,stride,cmax, flip) for n in range(nfile)])
                sparallel(plot_eta2D, n_procs, args)
            except:
                for n in range(nfile):
                    plot_eta2D(n, i, j, opath, ppath, d, gridtype,stride,cmax, flip)
                #for num in range(nfile):
                #    print(f'Plot {str(num)} of {str(nfile)} done.')

        case 'vel':
            try:
                args = list([(n, i, j, opath, ppath, d, gridtype, stride,cmax,flip,P) for n in range(nfile)])
                sparallel(plot_vel2D, n_procs, args)
            except:
                for n in range(nfile):
                    plot_vel2D(n, i, j, opath, ppath, d, gridtype, stride,cmax,flip,P)

def plot_eta1D(num, i, opath, ppath, d, dep, c=1):

    xmin = 0
    xmax = d["Mglob"]
                
    font = {'size'   : 12} 
    tstep = d["PLOT_INTV"]*num
    fnum = '%.5d' %num
    ffile = os.path.join(opath, 'eta_' + fnum)

    if not os.path.exists(ffile):
        print("Field file doesn't exist: %s" % ffile)
        return

    var = load_var('eta',opath,fnum,d)

    maxvar = np.max(var[c,:])

    fig, axs = plt.subplots(1, 1, facecolor='w', constrained_layout=True, sharex=True) 
    matplotlib.rc('font', **font) 
    fig.set_size_inches(w=9,h=7) 

    axs.plot(i[xmin:xmax],var[c,xmin:xmax],'-b',linewidth=2)
    axs.plot(i[xmin:xmax],dep[c,xmin:xmax],'-k',linewidth=2)

    # axs.plot(i[xmin:xmax],var[c,xmin:xmax], '-c', linewidth=0.2)
    # axs.fill_between(i[xmin:xmax], dep[c,xmin:xmax], var[c,xmin:xmax],
    #                 where = var[c,xmin:xmax] > dep[c,xmin:xmax],
    #                 facecolor = 'cyan', interpolate = True)
    # axs.fill_between(i[xmin:xmax], min(dep[c,xmin:xmax])-1, dep[c,xmin:xmax], 
    #                 where= dep[c,xmin:xmax] > (dep[c,xmin:xmax]-2),       
    #                 facecolor = '0.35', hatch = 'X')

    axs.set_ylim((min(dep[c,:]) - 1, maxvar + 1))
    axs.set_xlim(xmin * d["DX"], xmax * d["DX"])

    axs.set_xlabel('FUNWAVE X')
    axs.set_ylabel('Water level (m)')
    axs.set_title('%.1f secs' % tstep)

    plt.savefig(os.path.join(ppath,'surface_'+fnum+'.png'), dpi=600, bbox_inches='tight')
    plt.close()



def plot_eta2D(num, i, j, opath, ppath, d, gridtype, stride, cmax, flip):
                
    cmin=-cmax
    xmin,xmax,ymin,ymax,imin,imax,jmin,jmax=plot_boundary(gridtype,d,flip)
    font = {'size'   : 12} 
    tstep = d["PLOT_INTV"]*num
    fnum = '%.5d' %num
    ffile = os.path.join(opath, 'eta_' + fnum)

    if not os.path.exists(ffile):
        print("Field file doesn't exist: %s" % ffile)
        return

    eta = load_var('eta',opath,fnum,d) #np.fromfile(ffile).reshape(d["Nglob"],d["Mglob"])
    mask = load_var('mask',opath,fnum,d) #np.fromfile(os.path.join(opath,'mask_' + fnum)).reshape(d["Nglob"],d["Mglob"])
    eta_ma = np.ma.masked_where(mask==0,eta)
    eta_ma = eta_ma[::stride,::stride]
    if flip==1: eta_ma = np.fliplr(np.flipud(eta_ma))

    fig, axs = plt.subplots(1, 1, facecolor='w', constrained_layout=True, sharex=True) 
    matplotlib.rc('font', **font) 
    fig.set_size_inches(w=9,h=7) 
    #fig.tight_layout(pad=5.0)
    mappable = axs.pcolormesh(i,j,eta_ma,cmap=cmocean.cm.balance,vmin=cmin,vmax=cmax)
    fig.colorbar(mappable,ax=axs,label='elevation (m)')
    axs.set_ylim(ymin,ymax)
    axs.set_xlim(xmin,xmax)
    axs.set_aspect('equal')
    axs.set_xlabel('FUNWAVE X')
    axs.set_ylabel('FUNWAVE Y')
    axs.set_title('%.1f secs' % tstep)

    plt.savefig(os.path.join(ppath,'surface_'+fnum+'.png'), dpi=600, bbox_inches='tight')

    plt.close()

def plot_vel2D(num, i, j,opath, ppath, d, gridtype, stride, cmax, flip, P):

    cmin=0.0
    xmin,xmax,ymin,ymax,imin,imax,jmin,jmax=plot_boundary(gridtype,d,flip)
    font = {'size'   : 12} 
    tstep = d["PLOT_INTV"]*num
    fnum = '%.5d' % num
    ffile = os.path.join(opath, 'u_' + fnum)

    if not os.path.exists(ffile):
        print("Field file doesn't exist: %s" % ffile)
        return

    N = max(d["Mglob"],d["Nglob"])
    qstride = N//P
    if qstride==0: qstride=1

    mask = load_var('mask',opath,fnum,d) #np.fromfile(os.path.join(opath,'mask_' + fnum)).reshape(d["Nglob"],d["Mglob"])
    
    u = load_var('u',opath,fnum,d) #np.fromfile(ffile).reshape(d["Nglob"],d["Mglob"])
    uma = np.ma.masked_where(mask==0,u)
    u_ma = uma[::stride,::stride] 

    v = load_var('v',opath,fnum,d) #np.fromfile(os.path.join(opath, 'v_' + fnum)).reshape(d["Nglob"],d["Mglob"])
    vma = np.ma.masked_where(mask==0,v)
    v_ma = vma[::stride,::stride]
    velma = np.sqrt(u_ma**2 + v_ma**2)
    if flip==1: velma = np.fliplr(np.flipud(velma))
    #vel_ma = np.ma.masked_where(mask==0,vel)
    #vel_ma = vel_ma[::stride,::stride]
    
        
    fig, axs = plt.subplots(1, 1, facecolor='w', constrained_layout=True, sharex=True) 
    matplotlib.rc('font', **font) 
    fig.set_size_inches(w=9,h=7) 
    mappable = axs.pcolormesh(i,j,velma,cmap=cmocean.cm.amp,vmin=cmin,vmax=cmax)
    fig.colorbar(mappable,ax=axs,label='total velocity (m/s)')
    axs.set_ylim(ymin,ymax)
    axs.set_xlim(xmin,xmax)
    axs.set_aspect('equal')
    axs.set_xlabel('FUNWAVE X')
    axs.set_ylabel('FUNWAVE Y')
    axs.set_title('%.1f secs' % tstep)

    plt.savefig(os.path.join(ppath,'vel_'+fnum+'.png'), dpi=600, bbox_inches='tight')
    plt.close()

def mask_mean(opath,d,n):
    start = d["STEADY_TIME"] / d["PLOT_INTV"] - 1
    end = d["TOTAL_TIME"] / d["PLOT_INTV"]
    intv = d["T_INTV_mean"] / d["PLOT_INTV"]
    indices = np.arange(start,end,intv)
    nfile = intv
    shape = (d["Nglob"],d["Mglob"])
    masksum = np.zeros(shape=shape)
    for num in np.arange(indices[(n-1)],indices[n],1):
        fnum = '%.5d' %num
        mask = load_var('mask',opath,fnum,d) #np.fromfile(os.path.join(opath,'mask_'+fnum)).reshape(shape)
        masksum += mask
    maskmean = masksum / nfile
    return maskmean


def globalWaveEnergy(hsig):
    rho = 1000
    g = 9.81
    f = lambda x: (1/16)*rho*g*x**2
    return f(hsig)

def plot_mean(bpath,field,n0,n1,gridtype,cfactor=1.0,P=50,q=0,flip=0):
    ## PLOT MEAN GLOBAL OUTPUTS
    opath = os.path.join(bpath,'output')
    ppath = os.path.join(bpath,'postprocessing')
    if not os.path.exists(ppath): os.makedirs(ppath)

    d = readLOG(bpath)

    adcp1_fun = (2657, 1468)
    keyword = field.lower()

    for num in np.arange(n0,n1+1):
        fnum = '%.5d' %num
        mask = mask_mean(opath,d,num)
        # mask = load_var('mask',opath,'00000',d) #np.fromfile(os.path.join(opath,'mask_00000')).reshape(d["Nglob"],d["Mglob"])

        # quiver
        N = max(d["Mglob"],d["Nglob"])
        qstride = N//P
        if qstride==0: qstride=1
        umean = load_var('umean',opath,fnum,d) #np.fromfile(os.path.join(opath,'umean_'+fnum)).reshape(d['Nglob'],d['Mglob'])
        vmean = load_var('vmean',opath,fnum,d) #np.fromfile(os.path.join(opath,'vmean_'+fnum)).reshape(d['Nglob'],d['Mglob'])
        uma = np.ma.masked_where(mask==0,umean)
        vma = np.ma.masked_where(mask==0,vmean)

        match keyword:
            case "umean":
                varma = uma
                if flip==1: varma = -uma
                cmap = cmocean.cm.curl
                lab = 'velocity (m/s)'
                vmin = round(cfactor*-1.0)
                vmax = round(cfactor*1.0)
                col = 'k'
            case "vmean":
                varma = vma
                if flip==1: varma = -vma
                cmap = cmocean.cm.curl
                lab = 'velocity (m/s)'
                vmin = round(cfactor*-1.0)
                vmax = round(cfactor*1.0)
                col = 'k'
            case "vel":
                velmean = np.sqrt(umean**2 + vmean**2)
                varma = np.ma.masked_where(mask==0,velmean)
                cmap = cmocean.cm.rain
                lab = 'velocity (m/s)'
                vmin = 0
                vmax = round(cfactor*1.0)
                col = 'k'
            case "eta":
                #keyword = type.lower()+'mean'
                var = load_var(keyword+'mean',opath,fnum,d) #np.fromfile(os.path.join(opath,keyword+'mean_'+fnum)).reshape(d["Nglob"],d["Mglob"])
                varma = np.ma.masked_where(mask==0,var)
                cmap = cmocean.cm.dense
                lab = 'setup (m)'
                vmin = 0
                vmax = d["Hmo"]/16
                col = 'k'
            case "hsig" | "hrms" | "havg":
                var = load_var(keyword,opath,fnum,d) #np.fromfile(os.path.join(opath,keyword.capitalize()+'_'+fnum)).reshape(d["Nglob"],d["Mglob"])
                varma = np.ma.masked_where(mask==0,var)
                cmap = cmocean.cm.haline
                lab = keyword+ ' (m)'
                vmin = 0
                vmax = round(cfactor*d["Hmo"])
                col = 'k'
            case "energy":
                var = load_var('hsig',opath,fnum,d) #np.fromfile(os.path.join(opath,'Hsig_'+fnum)).reshape(d["Nglob"],d["Mglob"])
                varma = np.ma.masked_where(mask==0,var)
                varma = globalWaveEnergy(varma)
                cmap = cmocean.cm.solar
                lab = keyword+ '$J/m^2$'
                vmin = 0
                vmax = round(cfactor*globalWaveEnergy(d["Hmo"]))
                col = 'w'


        i = np.arange(0, d["Mglob"]*d["DX"], d["DX"])
        j = np.arange(0, d["Nglob"]*d["DY"], d["DY"])
        xmin,xmax,ymin,ymax,imin,imax,jmin,jmax=plot_boundary(gridtype,d,flip)
        boxes = ['e','d','a','b','c']
        if flip==1: 
            varma = np.fliplr(np.flipud(varma))
            uma = -np.fliplr(np.flipud(uma))
            vma = -np.fliplr(np.flipud(vma))

        fig, ax = plt.subplots(1,1,facecolor='w',constrained_layout=True, sharex=True)
        fig.set_size_inches(w=9,h=7)                
        mappable = ax.pcolormesh(i,j,varma,cmap=cmap,vmin=vmin,vmax=vmax)
        fig.colorbar(mappable,ax=ax,label=lab)
        #ax.plot(adcp1_fun[0],adcp1_fun[1],'rx')
        if q==1:
            Q = plt.quiver(i[::qstride],j[::qstride],uma[::qstride,::qstride],vma[::qstride,::qstride],color=col,scale=8,alpha=0.5)
            qk = plt.quiverkey(Q, 0.46, 0.1, 0.5, r'$0.5 \frac{m}{s}$', labelpos='E',labelcolor=col,coordinates='figure',color=col)
        for box in boxes:
            try:
                p1,p2,p3,p4,px1,px2,px3,px4,c = add_boxes(box,d,flip)
                epatch = matplotlib.patches.Polygon([px1,px2,px3,px4],linestyle='--',linewidth=2,edgecolor=c,facecolor='None')
                ax.add_patch(epatch)
            except:
                pass

        ax.set_ylim(ymin,ymax)
        ax.set_xlim(xmin,xmax)
        ax.set_aspect('equal')
        ax.set_xlabel('FUNWAVE X')
        ax.set_ylabel('FUNWAVE Y')

        fname = os.path.basename(bpath)
        dmy = datetime.now().strftime('%d%b%Y').upper()
        plt.savefig(os.path.join(ppath,keyword.lower()+'_'+gridtype+'_'+fnum+'_'+fname+'_'+dmy+'.png'), dpi=600, bbox_inches='tight')
        # plt.savefig(os.path.join(ppath,'vel_'+fnum+'.png'), dpi=600)
        plt.close()



def flipx(m,points):
    if isinstance(points,int):
        fpoints = m-points
    elif isinstance(points,list):
        fpoints = [m-x for x in points]
    elif isinstance(points,tuple):
        fpoints = m-points[0]
    return fpoints
def flipy(n,points):
    if isinstance(points,int):
        fpoints = n-points
    elif isinstance(points,list):
        fpoints = [n-y for y in points]
    elif isinstance(points,tuple):
        fpoints = n-points[1]
    return fpoints

def plot_boundary(gridtype,d,flip):
    match gridtype:
        case 'extended':
            xmin = 1100
            xmax = int(d["Mglob"]*d["DX"])
            ymin = 650
            ymax = 5251

            imin = int(1100/d["DX"])
            imax = d["Mglob"]
            jmin = int(650/d["DY"])
            jmax = int(5251/d["DY"])
        case 'project':
            xmin = 2300
            xmax = int(d["Mglob"]*d["DX"])
            ymin = 1800
            ymax = 3800

            imin = int(2300/d["DX"])
            imax = d["Mglob"]
            jmin = int(1800/d["DY"])
            jmax = int(3800/d["DY"])
        case _:
            xmin = 0
            xmax = int(d["Mglob"]*d["DX"])
            ymin = 0
            ymax = int(d["Nglob"]*d["DY"])

            imin = 0
            imax = d["Mglob"]
            jmin = 0
            jmax = d["Nglob"]

    if flip==1:
        imin=flipx(d["Mglob"],imin)
        imax=flipx(d["Mglob"],imax)
        jmin=flipy(d["Nglob"],jmin)
        jmax=flipy(d["Nglob"],jmax)

        xmin = int(imax*d["DX"])
        xmax = int(imin*d["DX"])
        ymin = int(jmax*d["DY"])
        ymax = int(jmin*d["DY"])

    return xmin, xmax, ymin, ymax, imin, imax, jmin, jmax


def readLOG(path):
    # logfilename = os.path.join(args.path, 'LOG.txt')
    logfilename = os.path.join(path, 'LOG.txt')

    # parse log txt file
    print("--------------------")
    # print("reading: " + logfilename)
    # default output file type
    outfiletype = 'ASCII'
    
    parse = ''
    with open(logfilename) as file:
        for line in file:
            if "STATISTICS" in line:
                break
            if not "---" in line:
                parse = [i for i in line.split() if i]
                for i in range(len(parse)):
                    if 'Mglob' in parse[i]:
                        Mglob = int(parse[i + 1])
                        # print("parsed Mglob: " + str(Mglob))
                    elif "Nglob" in parse[i]:
                        Nglob = int(parse[i + 1])
                        # print("parsed Nglob: " + str(Nglob))
                    elif "DX=" in parse[i]:
                        dx = float(parse[i + 1])
                        # print("parsed dx: " + str(dx))
                    elif "DY=" in parse[i]:
                        dy = float(parse[i + 1])
                        # print("parsed dy: " + str(dy))
                    elif "DEPTH_FILE" in parse[i]:
                        depthtype = "data"
                        depthfilepath = parse[i].split(':')[1]
                        # print("parsed depth, depth type: " + depthtype)
                        # print("parsed depthfilepath: "+ depthfilepath)
                    elif "DEPTH_FLAT" in parse[i]:
                        depthflat = float(parse[i+1])
                        depthtype = "flat"
                        # print("parsed depth, depth type: flat")
                        # print("parsed depthflat: " + str(depthflat))
                    elif "SLP" in parse[i]:
                        depthtype = "slope"
                        slp = float(parse[i + 1])
                        # print("parsed depth type: slope")
                        # print("parsed slp: " + str(slp))
                    elif "Xslp" in parse[i]: 
                        xslp = float(parse[i + 1])
                        # print("parsed xslp: " + str(xslp))
                    elif "DEP_WK" in parse[i]:
                        try:
                            depth = float(parse[i + 2])
                        except IndexError:
                            depth = 99.
                        print("parsed water depth: " + str(depth))
                    elif "Xc_WK" in parse[i]:
                        x_wavemaker = float(parse[i + 2])
                        # print("parse wavemaker location: " + str(x_wavemaker))
                    elif "TOTAL_TIME" in parse[i]:
                        totaltime = float(parse[i + 1])
                        print("parse total time: " + str(totaltime))
                    elif "PLOT_INTV" in parse[i]:
                        plotint = float(parse[i + 1])
                        print("parsed plot intv: " + str(plotint))
                    elif "STEADY_TIME" in parse[i]:
                        try:
                            steady = float(parse[i + 2])
                        except:
                            steady = 0.
                    elif "T_INTV_mean" in parse[i]:
                        try:
                            tmean = float(parse[i + 2])
                        except:
                            tmean = 0.
                    elif "Tperiod" in parse[i]:
                        period = float(parse[i + 2])
                        print("parsed wave period: " + str(period))
                    elif "AMP_WK" in parse[i]:
                        amp = float(parse[i + 2])
                    elif "FreqPeak" in parse[i]: 
                        # period = float(parse[i + 2])
                        freq = float(parse[i + 1])
                        period = 1/freq
                        print("parsed wave period: " + str(period))
                    elif "FreqMin" in parse[i]:
                        freqmin = float(parse[i + 2])
                    elif "FreqMax" in parse[i]:
                        freqmax = float(parse[i + 2])
                    elif "Hmo" in parse[i]:
                        hmo = float(parse[i+2])
                    elif "BINARY" in parse[i]:
                        outfiletype = 'binary'
                        print("parsed file i/o type: binary")
                    elif "WAVEMAKER:" in parse[i]:
                        parse = parse[i].split(':')
                        if "INI_SOL" or "WK_DATA2D" in parse[i+1]:
                            wk_flag = -1
                        elif "WK_REG" in parse[i+1]:
                            wk_flag = 0
                        elif "WK_IRR" or "JON_2D" or "JON_1D" or "TMA_1D" in parse[i+1]:
                            wk_flag = 1
                    
    
    # check for normal termination
    with open(logfilename) as file:
        for line in file:
            if "PRINTING FILE NO. 99999" in line:
                errflag = 1
                break
            if "Normal Termination" in line:
                errflag = 0
                break
            else:
                errflag = 9

    # print("closing: " + logfilename)
    print("--------------------")

    if wk_flag == 0 or wk_flag == 1:
        # wavelength
        wavelength = ldis(period,depth)
        # wave celerity (crest speed)
        celerity, regime = computeCelerity(depth,wavelength)
    else: 
        wavelength = 999
        celerity = 999


    d = {
        "TOTAL_TIME" : totaltime,
        "PLOT_INTV"  : plotint,
        "STEADY_TIME" : steady,
        "T_INTV_mean" : tmean,
        "Mglob" : Mglob,
        "Nglob" : Nglob,
        "DX" : dx,
        "DY" : dy,
        "DEP_WK" : depth,
        "Xc_WK" : x_wavemaker,
        "Wavelength" : wavelength,
        "Celerity" : celerity,
        "ErrChk" : errflag,
        "FIELD_IO_TYPE" : outfiletype
    }
    
    if wk_flag == 0:
        d["AMP_WK"] = amp
        d["Tperiod"] = period
    elif wk_flag == 1:
        d["Hmo"] = hmo
        d["FreqPeak"] = freq
        d["FreqMin"] = freqmin
        d["FreqMax"] = freqmax
    else:
        d["Hmo"] = 999.
        d["FreqPeak"] = 999.
    return d


##################
## PLOT STATIONS
#################
# simpaths = ['/p/work/mtorres/projects/cirp/fy25/test/waveref/surface_wave_1d_o6455477']
# stations = np.arange(1,106)
# spath = simpaths[0]
# sfname = 'gauges.txt'
# steadytime = 0.
# xfactor=16.
# label1=''
# label2=''
# labels=[label1,label2]
# run_stations(paths = simpaths, station=stations, spath=spath, sfname=sfname, steady=steadytime, labels=labels,xfactor=xfactor)

#################
## PLOT GLOBAL 
#################
# simpath = '/p/work/mtorres/projects/districts/nae/phase1a/spur/storm/yr100_o6427706'
# gridtype = 'project'
# field = 'eta'
# stride = 5
# cmax = 5.0
# P = 50
# flip = 0 
# n_procs = 30

# opath = os.path.join(simpath,'output')
# ppath = os.path.join(simpath,'postprocessing')

# if not os.path.exists(simpath):
#     print("Invalid Sim Path: %s" % simpath)

# if not os.path.exists(ppath):
#     os.makedirs(ppath)


# d = readLOG(simpath)
# plot_grid2D(simpath,d,field,stride,gridtype,cmax=cmax,flip=flip,P=P)

# plot_grid1D(simpath,d,field)


#############################
## PROJECT SPECIFIC FUNCTIONS
#############################

# def add_boxes(box,d,flip):
    # E1 = (1444,1812)
    # E2 = (1656,1757)
    # E3 = (1605,1574)
    # E4 = (1390,1636)

    # D1 = (1648,1718)
    # D2 = (1805,1676)
    # D3 = (1764,1531)
    # D4 = (1605,1574)

    # A1 = (1798,1645)
    # A2 = (1900,1604)
    # A3 = (1885,1395)
    # A4 = (1740,1435)

    # B1 = (1791,1420)
    # B2 = (1900,1378)
    # B3 = (1888,1208)
    # B4 = (1744,1248)

    # C1 = (1777,1238)
    # C2 = (1923,1195)
    # C3 = (1832,875)
    # C4 = (1686,914)
    # if int(d["DX"]) == 1:
    #     E1 = [int(e*2) for e in E1]
    #     E2 = [int(e*2) for e in E2]
    #     E3 = [int(e*2) for e in E3]
    #     E4 = [int(e*2) for e in E4]
        
    #     D1 = [int(d*2) for d in D1]
    #     D2 = [int(d*2) for d in D2]
    #     D3 = [int(d*2) for d in D3]
    #     D4 = [int(d*2) for d in D4]
        
    #     A1 = [int(a*2) for a in A1]
    #     A2 = [int(a*2) for a in A2]
    #     A3 = [int(a*2) for a in A3]
    #     A4 = [int(a*2) for a in A4]
        
    #     B1 = [int(b*2) for b in B1]
    #     B2 = [int(b*2) for b in B2]
    #     B3 = [int(b*2) for b in B3]
    #     B4 = [int(b*2) for b in B4]
        
    #     C1 = [int(c*2) for c in C1]
    #     C2 = [int(c*2) for c in C2]
    #     C3 = [int(c*2) for c in C3]
    #     C4 = [int(c*2) for c in C4]
    
    # E1xy = (int(E1[0]*d["DX"]), int(E1[1]*d["DY"]))
    # E2xy = (int(E2[0]*d["DX"]), int(E2[1]*d["DY"]))
    # E3xy = (int(E3[0]*d["DX"]), int(E3[1]*d["DY"]))
    # E4xy = (int(E4[0]*d["DX"]), int(E4[1]*d["DY"]))
           
    # D1xy = (int(D1[0]*d["DX"]), int(D1[1]*d["DY"]))
    # D2xy = (int(D2[0]*d["DX"]), int(D2[1]*d["DY"]))
    # D3xy = (int(D3[0]*d["DX"]), int(D3[1]*d["DY"]))
    # D4xy = (int(D4[0]*d["DX"]), int(D4[1]*d["DY"]))

    # A1xy = (int(A1[0]*d["DX"]), int(A1[1]*d["DY"]))
    # A2xy = (int(A2[0]*d["DX"]), int(A2[1]*d["DY"]))
    # A3xy = (int(A3[0]*d["DX"]), int(A3[1]*d["DY"]))
    # A4xy = (int(A4[0]*d["DX"]), int(A4[1]*d["DY"]))

    # B1xy = (int(B1[0]*d["DX"]), int(B1[1]*d["DY"]))
    # B2xy = (int(B2[0]*d["DX"]), int(B2[1]*d["DY"]))
    # B3xy = (int(B3[0]*d["DX"]), int(B3[1]*d["DY"]))
    # B4xy = (int(B4[0]*d["DX"]), int(B4[1]*d["DY"]))

    # C1xy = (int(C1[0]*d["DX"]), int(C1[1]*d["DY"]))
    # C2xy = (int(C2[0]*d["DX"]), int(C2[1]*d["DY"]))
    # C3xy = (int(C3[0]*d["DX"]), int(C3[1]*d["DY"]))
    # C4xy = (int(C4[0]*d["DX"]), int(C4[1]*d["DY"]))

    # if flip==1:
    #     E1 = (flipx(d["Mglob"],E1[0]),flipy(d["Nglob"],E1[1]))
    #     E2 = (flipx(d["Mglob"],E2[0]),flipy(d["Nglob"],E2[1]))
    #     E3 = (flipx(d["Mglob"],E3[0]),flipy(d["Nglob"],E3[1]))
    #     E4 = (flipx(d["Mglob"],E4[0]),flipy(d["Nglob"],E4[1]))

    #     E1xy = (int(E1[0]*d["DX"]), int(E1[1]*d["DY"]))
    #     E2xy = (int(E2[0]*d["DX"]), int(E2[1]*d["DY"]))
    #     E3xy = (int(E3[0]*d["DX"]), int(E3[1]*d["DY"]))
    #     E4xy = (int(E4[0]*d["DX"]), int(E4[1]*d["DY"]))

    #     D1 = (flipx(d["Mglob"],D1[0]),flipy(d["Nglob"],D1[1]))
    #     D2 = (flipx(d["Mglob"],D2[0]),flipy(d["Nglob"],D2[1]))
    #     D3 = (flipx(d["Mglob"],D3[0]),flipy(d["Nglob"],D3[1]))
    #     D4 = (flipx(d["Mglob"],D4[0]),flipy(d["Nglob"],D4[1]))

    #     D1xy = (int(D1[0]*d["DX"]), int(D1[1]*d["DY"]))
    #     D2xy = (int(D2[0]*d["DX"]), int(D2[1]*d["DY"]))
    #     D3xy = (int(D3[0]*d["DX"]), int(D3[1]*d["DY"]))
    #     D4xy = (int(D4[0]*d["DX"]), int(D4[1]*d["DY"]))

    #     A1 = (flipx(d["Mglob"],A1[0]),flipy(d["Nglob"],A1[1]))
    #     A2 = (flipx(d["Mglob"],A2[0]),flipy(d["Nglob"],A2[1]))
    #     A3 = (flipx(d["Mglob"],A3[0]),flipy(d["Nglob"],A3[1]))
    #     A4 = (flipx(d["Mglob"],A4[0]),flipy(d["Nglob"],A4[1]))

    #     A1xy = (int(A1[0]*d["DX"]), int(A1[1]*d["DY"]))
    #     A2xy = (int(A2[0]*d["DX"]), int(A2[1]*d["DY"]))
    #     A3xy = (int(A3[0]*d["DX"]), int(A3[1]*d["DY"]))
    #     A4xy = (int(A4[0]*d["DX"]), int(A4[1]*d["DY"]))

    #     B1 = (flipx(d["Mglob"],B1[0]),flipy(d["Nglob"],B1[1]))
    #     B2 = (flipx(d["Mglob"],B2[0]),flipy(d["Nglob"],B2[1]))
    #     B3 = (flipx(d["Mglob"],B3[0]),flipy(d["Nglob"],B3[1]))
    #     B4 = (flipx(d["Mglob"],B4[0]),flipy(d["Nglob"],B4[1]))

    #     B1xy = (int(B1[0]*d["DX"]), int(B1[1]*d["DY"]))
    #     B2xy = (int(B2[0]*d["DX"]), int(B2[1]*d["DY"]))
    #     B3xy = (int(B3[0]*d["DX"]), int(B3[1]*d["DY"]))
    #     B4xy = (int(B4[0]*d["DX"]), int(B4[1]*d["DY"]))

    #     C1 = (flipx(d["Mglob"],C1[0]),flipy(d["Nglob"],C1[1]))
    #     C2 = (flipx(d["Mglob"],C2[0]),flipy(d["Nglob"],C2[1]))
    #     C3 = (flipx(d["Mglob"],C3[0]),flipy(d["Nglob"],C3[1]))
    #     C4 = (flipx(d["Mglob"],C4[0]),flipy(d["Nglob"],C4[1]))

    #     C1xy = (int(C1[0]*d["DX"]), int(C1[1]*d["DY"]))
    #     C2xy = (int(C2[0]*d["DX"]), int(C2[1]*d["DY"]))
    #     C3xy = (int(C3[0]*d["DX"]), int(C3[1]*d["DY"]))
    #     C4xy = (int(C4[0]*d["DX"]), int(C4[1]*d["DY"]))

    # match box.upper():
    #     case 'E':
    #         return E1, E2, E3, E4, E1xy, E2xy, E3xy, E4xy, 'k'
    #     case 'D':
    #         return D1, D2, D3, D4, D1xy, D2xy, D3xy, D4xy, 'g'
    #     case 'A':
    #         return A1, A2, A3, A4, A1xy, A2xy, A3xy, A4xy, 'b'
    #     case 'B':
    #         return B1, B2, B3, B4, B1xy, B2xy, B3xy, B4xy, 'm'
    #     case 'C':
    #         return C1, C2, C3, C4, C1xy, C2xy, C3xy, C4xy, 'c'
