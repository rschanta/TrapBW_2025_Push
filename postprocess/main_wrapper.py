import os
import glob
import xarray as xr
import numpy as np
import pandas as pd


def trap_bw_post(dss):
    #%% REFLECTION STATIONS [1,2,3]
    '''
    Here, we need to reformat things a bit to get it ready for the analysis 
    function from waveref3
    
    '''
    # Select only the first three gages
    dss_eta = dss.eta_sta.sel(GAGE_NUM=[1, 2, 3])
    # Get into array format
    eta_array = dss_eta.transpose("t_station", "GAGE_NUM").values
    
    # Get dt
    dt = dss.PLOT_INTV_STATION 
    # Get water height- its's easiest just to use the first value of Z
    h = float(dss.Z.values[0,0])
    
    # Get distances between gages. A little clunky but gets the job done
    x_gages = dss.X.isel(X=dss.Mglob_gage.sel(GAGE_NUM=[1, 2, 3]) - 1).values.squeeze()
    dl = np.diff(x_gages)
    
    # Call function
    from waveref3 import reflection_analysis
    waveref3_123 = reflection_analysis(eta_array, h, dt, dl, g=9.81)
    
    #%% REFLECTION STATIONS [4,5,6]
    '''
    Realistically this is the same as [1,2,3], just wanted to explicitly show it.
    
    '''
    # Select only the first three gages
    dss_eta = dss.eta_sta.sel(GAGE_NUM=[4, 5, 6])
    # Get into array format
    eta_array = dss_eta.transpose("t_station", "GAGE_NUM").values
    
    # Get dt
    dt = dss.PLOT_INTV_STATION 
    # Get water height- its's easiest just to use the first value of Z
    h = float(dss.Z.values[0,0])
    
    # Get distances between gages. A little clunky but gets the job done
    x_gages = dss.X.isel(X=dss.Mglob_gage.sel(GAGE_NUM=[4, 5, 6]) - 1).values.squeeze()
    dl = np.diff(x_gages)
    
    # Call function
    from waveref3 import reflection_analysis
    waveref3_456 = reflection_analysis(eta_array, h, dt, dl, g=9.81)
    
    #%% REFLECTION WITH 2 STATIONS
    '''
    I forget which two you wanted, so just chose 3 and 4 (?)
    '''
    # Select only the gages you want
    dss_eta = dss.eta_sta.sel(GAGE_NUM=[3,4])
    # Get into array format
    eta_array = dss_eta.transpose("t_station", "GAGE_NUM").values
    # Pull out each array
    eta1 = eta_array[:, 0]
    eta2 = eta_array[:, 1]
    
    # Get dt
    dt = dss.PLOT_INTV_STATION 
    # Get water height- its's easiest just to use the first value of Z
    h = float(dss.Z.values[0,0])
    
    # Get distances between gages. A little clunky but gets the job done
    x_gages = dss.X.isel(X=dss.Mglob_gage.sel(GAGE_NUM=[3,4]) - 1).values.squeeze()
    dl = np.diff(x_gages)
    
    # Call function
    from waveref import reflection
    a_i, a_r, i_min, i_max, e_i, e_r, K_r = reflection(eta1, eta2, dl, dt, h)
    
    # Package into dictionary
    waveref2_45 = {
        "a_i_2_45": a_i,
        "a_r_2_45": a_r,
        "i_min_2_45": i_min,
        "i_max_2_45": i_max,
        "e_i_2_45": e_i,
        "e_r_2_45": e_r,
        "K_r_2_45": K_r
    }
    
    #%% Transmission
    from plotmaster import compute_PSD
    from plotmaster import calculateHsig
    from wavestructmaster import computeKt
    
    # Select by GAGE_NUM directly
    eta_pair = dss.eta_sta.sel(GAGE_NUM=[7, 8])
    
    # Convert to NumPy arrays for processing
    eta1 = eta_pair.sel(GAGE_NUM=7).values
    eta2 = eta_pair.sel(GAGE_NUM=8).values
    
    # Extract the shared time array and compute dt
    time = dss.t_station.values
    dt = dss.PLOT_INTV_STATION 
    
    # PSDs
    f1, S1 = compute_PSD(time, eta1)
    f2, S2 = compute_PSD(time, eta2)
    
    # Spectral stats
    Hmo1, Hrms1, E1 = calculateHsig(f1, S1)
    Hmo2, Hrms2, E2 = calculateHsig(f2, S2)
    
    
    # Transmission coefficient
    Kt = computeKt([E1, E2])
    
    # Package into nice dictionary
    trans_results = {
        "Hmo1_trans": Hmo1,
        "Hrms1_trans": Hrms1,
        "E1_trans": E1,
        "Hmo2_trans": Hmo2,
        "Hrms2_trans": Hrms2,
        "E2_trans": E2,
        "Kt": Kt
    }
    
    #%% Package into one dictionary
    # Rename keys for 3-station analysis
    waveref3_123 = {f"{k}_3_123": v for k, v in waveref3_123.items()}
    waveref3_456 = {f"{k}_3_456": v for k, v in waveref3_456.items()}
    # Combine into one master
    combined_results = {**waveref3_123, **waveref3_456, **waveref2_45, **trans_results}
    
    return combined_results


#%% MAIN CORE LOOP
# [EDIT] Specify path to nc_sta_files
path_to_sta = r'C:\Users\rschanta\DATABASE\USACE\ref\nc_sta_files/'
# [EDIT] Specify path to where the spreadsheets should output
dir_out = r"C:\Users\rschanta\DATABASE\USACE\ref\outputs"
# [EDIT] Specify what to call the output
output_name = 'BreakWaterStats'
# Make output folder
os.makedirs(dir_out,exist_ok=True)


# This automatically gets all the files/paths 
paths = sorted(glob.glob(os.path.join(path_to_sta,'tri_sta_*')))

results = []
# Loop through all files
for station_nc in paths:
    #try:
    print(f'Working on {station_nc}')
    # Load in data
    dss = xr.open_dataset(station_nc)
    # Run post processing
    combined = trap_bw_post(dss)
    # Merge in original attributes
    merged = {**dss.attrs, **combined}
    # Append on 
    results.append(merged)
    print(f'\tSuccessfully post-processed {station_nc} !')
    #except:
    #    print(f'\tIssue with {station_nc}, skipping!')
    
# Build the DataFrame
df_results = pd.DataFrame(results)
print('\nLoop Complete! Saving outputs...')
df_results.to_csv(os.path.join(dir_out,f'{output_name}.csv'), index=False)
print(f"\tCSV saved out to {os.path.join(dir_out,f'{output_name}.csv')}")
df_results.to_excel(os.path.join(dir_out,f'{output_name}.xlsx'), index=False)
print(f"\tXLSX saved out to {os.path.join(dir_out,f'{output_name}.xlsx')}")
df_results.to_parquet(os.path.join(dir_out,f'{output_name}.parquet'), index=False)
print(f"\tParquet saved out to {os.path.join(dir_out,f'{output_name}.parquet')}")

print("Post-processing successful!")

