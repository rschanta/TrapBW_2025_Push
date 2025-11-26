import numpy as np
import os
from scipy.fft import fft
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def wavelen(h, T, max_iter, g):
    # Function to calculate wavelength from the dispersion relation
    # This is a placeholder and needs to be implemented as per the original function
    pass

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


def reflection_analysis(array, h, dt, dl, g=9.81):
    # Parameters
    # Hz = 10  # sampling frequency
    # dt = 1 / Hz  # delta t
    # g = 9.81  # gravity

    # files = [f for f in os.listdir(path) if f.endswith('.mat')]

    # Preallocation
    refanalysis = []
    
    # for file_name in files: # for each file in list of files
    #     data = sio.loadmat(os.path.join(path, file_name)) # load one file

    #     arraylong = np.vstack([data['data'][10] / 100, data['data'][9] / 100, data['data'][8] / 100]).T # take three columns (10, 9, 8) from file, all divided by 100 -- likely three time series of water level recorded in cm, converted to m, transposed to columns by station
    #     arraylong = arraylong - np.mean(arraylong, axis=0)  # Detrending
        
    #     ind = np.where(arraylong[:, 2] > 0.02)[0] # find where water level first rises above 2cm at last station
    #     m = ind[0]

    #     # Set the range of data based on its length
    #     if len(arraylong) > 20000: # limiting total time series length -- not sure if necessary for our case
    #         n = int(19 * 60 * Hz)
    #     elif len(arraylong) > 15000:
    #         n = int(10 * 60 * Hz)
    #     else:
    #         n = int(6.33 * 60 * Hz)

    #     array = arraylong[m:m + n, :]

    nfft = len(array)
    df = 1 / (nfft * dt)  # Frequency resolution
    half = round(nfft / 2)

    An = np.zeros((half - 1, 3)) # preallocation
    Bn = np.zeros((half - 1, 3))
    Sn = np.zeros((half - 1, 3))
    k = []
    Ainc = np.zeros((half - 1, 3))
    Binc = np.zeros((half - 1, 3))
    Aref = np.zeros((half - 1, 3))
    Bref = np.zeros((half - 1, 3))
    nmin = np.zeros(3)
    nmax = np.zeros(3)

    # Solve Fourier coefficients (A1-B2) for all three gauges
    for j in range(3):
        fn = fft(array[:, j], nfft)
        a0 = fn[0] / nfft
        An[:, j] = 2 * np.real(fn[1:half]) / nfft  # Real component
        Bn[:, j] = -2 * np.imag(fn[1:half]) / nfft  # Imaginary component

        fn_squared = np.abs(fn) ** 2
        fn_fold = fn_squared[1:half] * 2
        Sn[:, j] = dt * fn_fold / nfft  # Spectral/energy density

    f = df * np.arange(len(Sn))  # Frequencies

    # Solve for the wavenumber at each frequency
    for i in range(len(f)):
        L = ldis(1 / f[i], h)
        k.append((2 * np.pi) / L)
    k = np.array(k)

    # Solve for amplitude of incident (ai) and reflected waves (ar)
    g1 = [0, 0, 1]
    g2 = [1, 2, 2]
    gpos = np.append(0,dl)
    # gpos = [0, 0.30, 0.90]  # Distance from first gauge in the array

    for j in range(3):
        A1 = An[:, g1[j]]
        A2 = An[:, g2[j]]
        B1 = Bn[:, g1[j]]
        B2 = Bn[:, g2[j]]
        pos1 = gpos[g1[j]]
        pos2 = gpos[g2[j]]

        term1 = -A2 * np.sin(k * pos1) + A1 * np.sin(k * pos2) + B2 * np.cos(k * pos1) - B1 * np.cos(k * pos2)
        term2 = A2 * np.cos(k * pos1) - A1 * np.cos(k * pos2) + B2 * np.sin(k * pos1) - B1 * np.sin(k * pos2)
        term3 = -A2 * np.sin(k * pos1) + A1 * np.sin(k * pos2) - B2 * np.cos(k * pos1) + B1 * np.cos(k * pos2)
        term4 = A2 * np.cos(k * pos1) - A1 * np.cos(k * pos2) - B2 * np.sin(k * pos1) + B1 * np.sin(k * pos2)

        Ainc[:, j] = term1 / (2 * np.sin(k * (pos2 - pos1)))
        Binc[:, j] = term2 / (2 * np.sin(k * (pos2 - pos1)))

        Aref[:, j] = term3 / (2 * np.sin(k * (pos2 - pos1)))
        Bref[:, j] = term4 / (2 * np.sin(k * (pos2 - pos1)))

        # Upper and lower limits of significant spectra
        Lmin = abs(pos2 - pos1) / 0.45  # Ranges suggested by Goda and Suzuki (1976)
        Lmax = abs(pos2 - pos1) / 0.05

        kmin = 2 * np.pi / Lmin
        kmax = 2 * np.pi / Lmax

        wmin = np.sqrt(g * kmin * np.tanh(kmin * h))
        wmax = np.sqrt(g * kmax * np.tanh(kmax * h))

        fmin = wmin / (2 * np.pi)
        fmax = wmax / (2 * np.pi)

        nmin[j] = round(fmax / df)
        nmax[j] = round(fmin / df)

    range_idx = np.arange(int(np.min(nmin)), int(np.max(nmax)) + 1)

    # Averaging overlapped amplitudes across gauges
    for j in range(3):
        for i in range(len(Ainc)):
            if i < nmin[j] or i > nmax[j]:
                Ainc[i, j] = np.nan
                Binc[i, j] = np.nan
                Aref[i, j] = np.nan
                Bref[i, j] = np.nan

    Aincav = np.nanmean(Ainc[range_idx, :], axis=1)
    Bincav = np.nanmean(Binc[range_idx, :], axis=1)
    Arefav = np.nanmean(Aref[range_idx, :], axis=1)
    Brefav = np.nanmean(Bref[range_idx, :], axis=1)

    # Backing out spectra
    Si = (Aincav ** 2 + Bincav ** 2) / (2 * df)
    Sr = (Arefav ** 2 + Brefav ** 2) / (2 * df)
    Sfcheck = (An ** 2 + Bn ** 2) / (2 * df)

    # Evaluate energies of resolved incident and reflected waves
    Ei = np.sum(Si) * df
    Er = np.sum(Sr) * df

    # Reflection coefficient
    refco = np.sqrt(Er / Ei)

    # Calculate incident, reflected, and total Hmo wave height
    mo = np.sum(Sn, axis=0) * df
    Htot = 4.004 * np.sqrt(mo)
    Hi = 4.004 * np.sqrt(Ei)
    Hr = 4.004 * np.sqrt(Er)
    Hicheck = np.mean(Htot) / np.sqrt(1 + refco ** 2)
    Hrcheck = refco * np.mean(Htot) / np.sqrt(1 + refco ** 2)

    refanalysis = {
        'refco': refco,
        'Hi': Hi,
        'Hr': Hr,
        'Hicheck': Hicheck,
        'Hrcheck': Hrcheck
    }

    # debug
    fig, ax = plt.subplots(facecolor='w')
    ax.plot(f[range_idx],Si, f[range_idx],Sr, f[range_idx],Sfcheck[range_idx,0])
    plt.savefig("waveref3_energy.png")
#     %     FigHandle = figure('Position', [100, 100, 700, 597]);
# %     subplot(3,1,1)
# %     plot(dt:dt:dt*length(array),array(:,1)+0.10,'k',dt:dt:dt*length(array),array(:,2),'r',dt:dt:dt*length(array),array(:,3)-0.10,'b')
# %     ylabel('eta [m]')
# %     xlabel('Time (s)')
# %     xlim([m*dt dt*length(array)])
# %     grid on
# %     hold on
# %     pause
    #     %     %For plotting
    # %     subplot(3,1,2)
    # %     plot(f(range),Si,':b',f(range),Sr,'r-.',f(range),Sn(range,1),'k','LineWidth',1.5)
    # %     xlim([f(range(1)) f(range(end))])
    # %     legend('Incident','Reflected','Composite')
    # %     xlabel('Frequency [Hz]')
    # %     ylabel('S_f [m^2*s]')
    # %     ylim([0 max(Sn(range,1))+max(Sn(range,1))*0.05])
    # %     grid on
    # %     hold on
    
    # %     %Band averaging spectra for plotting
    # %     no_bands=5;
    # %     bands=floor(length(Si)/no_bands);
    # %     flim_band=zeros(bands,1);
    # %     Si_band=zeros(bands,1);
    # %     Sr_band=zeros(bands,1);
    # %     Sf_band=zeros(bands,1);
    # %
    # %     flim=f(range);
    # %     Sf=Sn(range,1);
    # %
    # %     for j=1:bands
    # %         flim_band(j,1)= mean(flim((j-1)*no_bands+1:j*no_bands));
    # %         Si_band(j,1)= mean(Si((j-1)*no_bands+1:j*no_bands));
    # %         Sr_band(j,1)= mean(Sr((j-1)*no_bands+1:j*no_bands));
    # %         Sf_band(j,1)= mean(Sf((j-1)*no_bands+1:j*no_bands));
    # %     end
    # %
    # %     subplot(3,1,3)
    # %     plot(flim_band,Si_band,':b',flim_band,Sr_band,'r-.',flim_band,Sf_band,'k','LineWidth',1.5)
    # %     xlim([flim(1) flim(end)])
    # %     legend('Incident','Reflected','Composite')
    # %     xlabel('Frequency [Hz]')
    # %     ylabel('S_f [m^2*s]')
    # %     ylim([0 max(Sf_band)+max(Sf_band)*0.05])
    # %     grid on

    return refanalysis
        # Write results to text file
        # with open('reflection_analysis.txt', 'a') as f:
        #     f.write(f'{file_name.split(".")[0]:<10} {refco:6.3f} {Hi*100:6.2f} {Hr*100:6.2f} {Hicheck*100:6.2f} {Hrcheck*100:6.2f}\n')


def main():
    print("waveref3.py main() invoked")

if __name__ == '__main__':
    main()
