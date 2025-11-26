import matplotlib, cmocean, os, glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig
import waveref, waveref3
import plotmaster

def wave_transmission_emp(epsilon,Hmo,relF,B):
    Kt_rubble = -0.4*(relF) + 0.64*(B/Hmo)**(-0.31) * (1-np.exp(-0.5*epsilon)) # rubble mound structures
    Kt_smooth = -0.3*relF + 0.75*(1-np.exp(-0.5*epsilon))
    return Kt_rubble, Kt_smooth

def computeKt(totalEList):
    return totalEList[-1] / totalEList[0]

def gen_dataframe(dct):
    T = dct['T']
    h = dct['h']
    relH = dct['relH']
    relF = dct['relF']
    relB = dct['relB']
    m = dct['m']
    Cd = dct['Cd']

    Lam = [plotmaster.ldis(T[i],h[i]) for i in range(len(T))]
    dx = [round(l/70,2) for l in Lam]

    Tdata = []
    hdata = []
    adata = []
    dxdata = []
    relHdata = []
    relFdata = []
    relBdata = []
    mdata = []
    Cddata = []
    surfdata = []
    steepdata = []
    ktrubbledata = []
    ktsmoothdata = []
    BHdata = []

    for id in range(len(T)):
        for Hh in range(len(relH)):
            for FH in range(len(relF)):
                for BL in range(len(relB)):
                    for mi in range(len(m)):
                        for cdi in range(len(Cd)):
                            L = plotmaster.ldis(T[id],h[id])
                            H = relH[Hh] * h[id]
                            steepness = H/L
                            surf_sim = (1/m[mi])*np.sqrt(steepness)
                            surfdata.append(surf_sim)
                            steepdata.append(steepness)

                            Rc = relF[FH]
                            B = relB[BL] * L
                            kt_rubble,kt_smooth = wave_transmission_emp(surf_sim,H,Rc,B)
                            ktrubbledata.append(kt_rubble)
                            ktsmoothdata.append(kt_smooth)

                            a = H / 2
                            adata.append(a)

                            relHdata.append(relH[Hh])
                            relFdata.append(relF[FH])
                            relBdata.append(relB[BL])
                            mdata.append(m[mi])
                            Cddata.append(Cd[cdi])

                            dxdata.append(dx[id])
                            Tdata.append(T[id])
                            hdata.append(h[id])
                            BHdata.append(B/H)

    test_bed = pd.DataFrame(data=list(zip(Tdata,hdata,adata,relHdata,relFdata,relBdata,BHdata,mdata,Cddata,surfdata,steepdata,ktsmoothdata,ktrubbledata)),
                            columns=['T','d','a','H/h','F/H','B/L','B/H','m','cd','surfsim','H/L','Kt_smooth','Kt_rubble'])

    return test_bed

def walk_through_files(path):
    dirs = os.listdir(path)
    pdirs = [os.path.join(path,d) for d in dirs]
    # dirs = []
    # for (dirpath, dirnames, filenames) in os.walk(path):
    #     for dirname in dirnames:
    #        dirs.append(os.path.join(dirpath,dirname))
    return pdirs

def parse_ifilenames(dirs):
   Asplist = []
   Tlist = []
   hlist = []
   cwd = os.getcwd()
   for dir in dirs:
      os.chdir(dir)
      fdir = os.path.basename(dir)
      Asp = float(fdir[3:7].replace('d','.'))
      ifile = glob.glob('input*')
      T = int(ifile[0][6:12][1:3])
      h = int(ifile[0][6:12][4:6])

      Asplist.append(Asp)
      Tlist.append(T)
      hlist.append(h)
   df = pd.DataFrame({'A_sponge' : Asplist,'T_period':Tlist,'DEP_WK':hlist})
   os.chdir(cwd)
   return df

def build_df(bpath):
    out_dirs = walk_through_files(bpath)
    df = parse_ifilenames(out_dirs)
    df["path"] = out_dirs
    return df

def run_transmission(bpath,df,station,steadytime,xfactor,labels):
    # df = build_df(bpath)
    KtList = []
    Hmolist = []
    Tplist = []
    Energylist = []
    for index, row in df.iterrows():
        # addPath = 'period%d/depth%d/relH%3.2f/relF%3.2f/relB%3.2f/m%d/cd%4.3f' % (row['T'],row['d'],row["H/h"],row["F/H"],row["B/L"],row['m'],row["cd"])
        # addPath = addPath.replace('.','').replace('-','')
        # simPath = os.path.join(bpath,addPath)
        simPath = row['path']
        simPaths = sorted(glob.glob(simPath+'*'))

        if len(simPaths) == 0: 
            KtList.append(np.nan)
            print("Invalid Sim Path: %s" % simPath)
            continue

        simPath = simPaths[-1]
        opath = os.path.join(simPath,'output')
        ppath = os.path.join(bpath,'postprocessing')
        if not os.path.exists(ppath): os.makedirs(ppath)

        d = plotmaster.readLog(simPath)
        if d["ErrChk"] == 1:
            print("Simulation failed: %s" % simPath)
            # continue
        stationdata = plotmaster.readStationsData(opath,station)
        nfile = len(stationdata)

        fpath = os.path.basename(simPath)
        fname = [fpath+'_%.4d'%station[0],'%.4d'%station[1]]
        Hmo, Tp, Energy = plotmaster.compare_PSD(nfile, ppath, stationdata, steadytime, labels, xfactor,fname)

        Kt = computeKt(Energy)
        KtList.append(Kt)
        Hmolist.append(Hmo)
        Tplist.append(Tp)
        Energylist.append(Energy)

    df["Kt"] = KtList
    
    fname = os.path.basename(os.path.dirname(ppath))
    df_result = pd.DataFrame({'station':station,'Hmo':Hmolist,'Tp':Tplist,'Energy':Energylist,'Kt':KtList})
    df_out = pd.concat([df,df_result],axis=1)
    df_out.to_excel(os.path.join(ppath,'transmission_'+fname+'.xlsx'))
    return(df)

def run_reflection(bpath,stations,df):
    Kr3list = []
    Kr2list = []
    for index,row in df.iterrows():
        simPath = row['path']
        simPaths = sorted(glob.glob(simPath+'*'))

        if len(simPaths) == 0: 
            Kr3list.append(np.nan)
            Kr2list.append(np.nan)
            print("Invalid Sim Path: %s" % simPath)
            continue

        simPath = simPaths[-1]
        opath = os.path.join(simPath,'output')
        ppath = os.path.join(bpath,'postprocessing')
        if not os.path.exists(ppath): os.makedirs(ppath)

        d = plotmaster.readLOG(simPath)
        stationdata = plotmaster.readStationsData(opath,stations)   
        dl = plotmaster.get_station_xdist_1d(simPath,'gauges.txt',stations,d)


        for index, stationData in enumerate(stationdata):
            timeS = stationData[:,0] # first row in sta_000# is time stamp
            etaS = stationData[:,1]
            dt = round(np.diff(timeS).mean(),2)
            if index==0: eta1 = etaS
            elif index==1: eta2 = etaS

        data = np.vstack([stationdata[0][:,1],stationdata[1][:,1],stationdata[2][:,1]]).T
        dt = round(np.diff(stationdata[0][:,0]).mean(),2)
        h = d["DEP_WK"]

        # print(f'DEP_WK: {h}')

        if "FreqMin" in d.keys():
            f_min = d["FreqMin"]
        else:
            f_min = 0.05
        if "FreqMax" in d.keys():    
            f_max = d["FreqMax"]
        else:
            f_max = 0.45

        (a_i, a_r, i_min, i_max, e_i, e_r, K_r) = waveref.reflection(eta1, eta2, dl[0], dt, h)
        # print(f'waveref2 Kr: {K_r}, for delta_l: {dl[0]}')

        ref_dict = waveref3.reflection_analysis(data,h,dt,dl)
        Kr3list.append(ref_dict['refco'])
        Kr2list.append(K_r)

    
    df["Kr_3"] = Kr3list
    df["Kr_2"] = Kr2list
    # print(f'waveref3 Kr: {ref_dict["refco"]}')

    fname = os.path.basename(os.path.dirname(ppath))
    df_result = pd.DataFrame({'stations':stations,'delta_l':dl,'DEP_WK':h,'Kr_3':ref_dict["refco"],'Kr_2':K_r,'dl_2':dl[0]})
    df_out = pd.concat([df,df_result],axis=1)
    df_out.to_excel(os.path.join(ppath,'reflection_'+fname+'.xlsx'))

    return(df)

def run_runup():
    """
    Requires Mike's funwavetvdtools src in FUNWAVE-TVD-Python-Tools
    """

    pass

def run_overtopping():
    pass

