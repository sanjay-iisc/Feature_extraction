import numpy as np
from scipy.fftpack import fft
from scipy.signal import welch
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal 
from scipy import interpolate
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn import neighbors
import matplotlib.patches as mpatches

#####-----------------------------------------
def number_points_matrix(brust_p,Va0,Fs):
    Loc_data= pd.read_csv("E:\\Work\\Work\\POD_Analysis\\Pod_Simulation\\without_hole_pristien\\Input_sensor.txt",skiprows=1,delimiter=' ', names=["XR","YR", 'Loca','Type','Ri' ,'Ro'] )
    MM,NN=Loc_data[['XR','YR']].shape
    X=Loc_data['XR'].values
    Y=Loc_data['YR'].values
    dist=np.zeros((MM,MM))
    for em in range(MM):
        dist[em,:]=np.sqrt( ( X[em]-X )**2+( Y[em]-Y )**2)
    Mat_of_number=np.array((brust_p+(dist/Va0))*Fs,dtype=int)
    return  Mat_of_number

def window_in_different_direction(X,Number_mat):
    N=X.shape[-1] # Time span
    A=X.shape[-3] # Number Actuator
    R=X.shape[-2] # Number Sensors
    W_AR=np.zeros((A,R,N))
    for emitter in range(A):
        for recevier in range(R):   
            W_AR[emitter,recevier,:]=window_signal_1D(N,Number_mat[0,0],Number_mat[emitter,recevier]+20)
    return W_AR

def window_Matrix_function(X,Brust,fA0,Fs):
    Window_Matrix=[]
    Nfreq=X.shape[0] # Number of Frequency
    for brust, VA0 in zip(Brust,fA0):   
        Number_mat=number_points_matrix(brust,VA0,Fs)
        window_Matrix=window_in_different_direction(X,Number_mat)
        while window_Matrix.ndim!=X.ndim-1:  # make the dimesnssion are equal
            window_Matrix=window_Matrix[np.newaxis,:]
        Window_Matrix.append(window_Matrix)
    Window_Matrix=np.array(Window_Matrix)    
    Window_Matrix=np.array(Window_Matrix)
    print(Window_Matrix.ndim)
    print(X.ndim)
    return X*Window_Matrix
          
def window_signal_1D(N,Ncoupled,Ndirect):
    # print(Ndirect)
    w_coupl=np.zeros(Ncoupled)#signal.get_window(('tukey', 0.2), Ncoupled)
    w_dir=signal.get_window(('tukey', 0.3), abs(Ncoupled-Ndirect))
    w_zeros=np.zeros(abs(N-Ndirect))
    w_path=np.concatenate((w_coupl,w_dir,w_zeros))
    # print(w_path.shape)
    return w_path



