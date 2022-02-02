
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
###-----
import pywt
from scipy import interpolate
import utils as data_utils
from scipy.signal import welch
from scipy.fftpack import fftfreq, fft,rfftfreq,rfft,fftshift

####WaveLet transformation
# https://github.com/regeirk/pycwt/blob/master/pycwt/sample/sample.py
##https://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
#https://pycwt.readthedocs.io/en/latest/tutorial.html#time-series-spectral-analysis-using-wavelets


class  FeatureExteraction:
    def __init__(self,T, YP, YD,f_s=2):
        """
        T = Time vector [N*1]
        YP = Hole Data [# Freq, 0-CrackLength, # Plate, #Emitter, #Rececier, #Time_vector]=Dim=6
        YD = Hole+Growing Crack Data [# Freq, #CrackLength ,# Plate, #Emitter, #Rececier, #Time_vector]=Dim=6
        """
        self.T=T
        self.YP=YP
        self.YD=YD
        self.f_s=f_s
    
    def correlation_coefficient(self):
        a=self.YP
        b=self.YD
        a_mean=np.mean(a, axis=-1, keepdims=True)
        b_mean=np.mean(b, axis=-1, keepdims=True)
        a_std=np.std(a,axis=-1, keepdims=True)
        b_std=np.std(b,axis=-1, keepdims=True)
        U=np.sum((a-a_mean)*(b-b_mean), axis=-1, keepdims=True)#(a_std*b_std)#np.dot(a,np.swapaxes(b,-2,-1))
        D= np.sqrt( np.sum((a-a_mean)**2, axis=-1, keepdims=True)*np.sum((b-b_mean)**2, axis=-1, keepdims=True))
        self.Corr=1-(U/D)
        print("DimCorr={}".format(self.MA.shape))


    
    def Maximum_Amplitude(self):
        MA_1=np.linalg.norm(self.YP-self.YD,axis=-1,keepdims=True)
        MA_2=np.linalg.norm(self.YP,axis=-1,keepdims=True)
        self.MA=MA_1/MA_2
        print("DimMA={}".format(self.MA.shape))
    
    def fft_Damage_Index(self):
        F_Freq, yfft_P=self.get_fft_values(self.YP)
        F_Freq, yfft_D=self.get_fft_values(self.YD)
        self.max_FFT=np.amax(abs(yfft_P-yfft_D),axis=-1,keepdims=True)
        self.F_Freq=F_Freq
        print("DimFFT={}".format(self.max_FFT.shape))
    
    def psd_Damage_Index(self):
        psd_Freq, ypsd_P=self.get_psd_values(self.YP)
        psd_Freq, ypsd_D=self.get_psd_values(self.YD)
        self.max_PSD=np.amax(abs(ypsd_P-ypsd_D),axis=-1,keepdims=True)
        self.psd_Freq=psd_Freq
        print("DimPSD={}".format(self.max_PSD.shape))

    
    def get_fft_values(self,y_values):
        N= len(self.T)
        fft_values_ = fft(y_values, axis=-1)
        f_values = np.linspace(0.0, self.f_s/(2.0), N//2)
        fft_values = 2.0/N * np.abs(fft_values_[...,0:N//2])
        return f_values, fft_values
    
    def get_psd_values(self,y_values):
        N= len(self.T)
        f_values, psd_values = welch(y_values, fs=self.f_s,nperseg=N,axis=- 1)
        return f_values, psd_values
    
    
    def scalling_wavlet(self,dj=0.125):
        t = self.T.flatten()
        N = t.size
        ###----
        self.dt =(t[1]-t[0])
        s0= 40*self.dt
        J = int((1/dj)*np.log2((N*self.dt)/(s0)))+1
        j=np.arange(0,J,1)
        self.scale=s0* 2**(j*dj)
     

    def wavelet_trans_Damage_Index(self, index=4):
        # Index is for low the computational 
        # Index relates to the Frequency
        
        Norm_=np.amax(abs(self.YP), axis=-1, keepdims=True)
        self.YR=(self.YP-self.YD)/Norm_ # Residual signal
        self.CWT, self.cwt_Freq=self.get_wavelet_transformation(self.YR[index,])
        print("Dim_CWT={}".format(self.CWT.shape))
        print("**we are taking the sum axis=0 and axis=-1 for Damage Index**")
        self.sum_CWT=np.sum(np.sum(self.CWT, axis=0), axis=-1,keepdims=True)
        print("Dim_Sum_DamageIndex_CWT={}".format(self.sum_CWT.shape))

    def get_wavelet_transformation(self,signal):
        self.scalling_wavlet()
        # std = dat.std()  # Standard deviation
        # var = std ** 2  # Variance
        # dat_norm = dat #/ std
        waveletname = 'cmor1.5-2.0'
        coefficients,FFF=pywt.cwt(signal, self.scale, waveletname, self.dt, axis=-1)
        power = (abs(coefficients)) ** 2
        return power,FFF
    
    def get_plot_wavelet(self,power, frequencies):
        # power = (abs(coefficients)) ** 2
        period = 1. / frequencies
        levels = np.array([0.0625, 0.125, 0.25, 0.5, 1, 5])
        contourlevels = (levels)
        fig=plt.figure(figsize=(15, 10))
        bx=fig.add_axes([0,0,1,1])
        # plt.contourf(t, numpy.log2(period), numpy.log2(power), numpy.log2(levels),
        #             extend='both', cmap=plt.cm.viridis)
        im=bx.contourf(self.T.flatten(), frequencies, (power), (levels),
                    extend='both', cmap=plt.cm.seismic)
        cbar_ax = fig.add_axes([1, 0.5, 0.03, 0.25])
        fig.colorbar(im, cax=cbar_ax, orientation="vertical")

if __name__ == "__main__":
    Exp_P=np.load('../POD_DATA/Experiment/Exp_Hole_DATA.npy')
    Exp_D=np.load('../POD_DATA/Experiment/Exp_CRACKLENGTH_DATA.npy')
    Exp_T=np.load('../POD_DATA/Experiment/Exp_TimeVector.npy')
    Try=FeatureExteraction(Exp_T)
