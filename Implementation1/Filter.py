import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import utilities

import librosa #for converting mp3 files into readable input
import soundfile as sf #for writing mp3 files as output

def examine_effect_of_M():
    M = [10]#,100,1000,10000]
    w_c = np.pi/2#max(L,D) 
    for val in M:
        n = np.arange(-val, val+1)
        h = (1/np.pi * n) * np.sin(w_c*n)
        utilities.plot_dtft(h,val)

#notes on different values of M
#the sharpness of the curve increases drastically as M is increased
#At M = 10000, the curve is extremely sharp but it is still not exactly a rectangular function
#In addition, it applies a gain > 1 at the cutoff frequency

#3.3)----------------------------------------------------------------------------
#Downside 1 - increasing the number of taps in an FIR filter increases the amount of computation necessary
#This applies to both real-time and offline systems as both require the same amount of computation
#Downside 2 - Increasing M creates a delay, the filter must be shifted by M in order to make it a causal system
#This mainly applies to real-time systems as it introduces a delay between the sample and filtered result
#This does not affect offline systems as the time between input and output will likely be less than time
#caused by the delay


#3.4
def Reusable_Filter(w_c, M):
    n = np.arange(-M, M+1)
    h = (w_c/np.pi)*np.sinc(w_c * n/ np.pi) 
    return h

def rate_convert(x,L,D,M=500):
    #Upsampling by L
    xUp = np.zeros(len(x) * L) #array length x times L
    xUp[::L] = x #every Lth element = element of x
    #LPF
    w_c = np.pi / max(L,D)
    h = L * Reusable_Filter(w_c, M)
    #convolve LPF output with input
    xFilt = scipy.signal.lfilter(h, 1, xUp)
    #Downsample by D
    Out = xFilt[::D]
    return Out

def equalizer(x, fLow, fMid, fHigh, GLow, GMid, GHigh, fs, M = 500):
    def BPF(f1, f2): #Band pass filter
        w_1 = f1 /fs * 2 * np.pi
        w_2 = f2/fs * 2 * np.pi
        w_H = (w_1+w_2)/2
        w_c = (w_2 - w_1)/2
        h = Reusable_Filter(w_c,M) * 2 * np.cos(w_H * np.arange(-M, M+1))
        return h
    
    h1 = Reusable_Filter(2 * np.pi * fLow/fs, M)
    h2 = BPF(fLow, fMid)
    h3 = BPF(fMid, fHigh)
    yLow = GLow * scipy.signal.lfilter(h1, 1, x) #linear convolution of h1 and x, 
    yMid = GMid * scipy.signal.lfilter(h2, 1, x)
    yHigh = GHigh * scipy.signal.lfilter(h3, 1, x)
    yOut= yLow+yMid+yHigh
    return yOut


if __name__ == '__main__':
    input_file="Funk.mp3"
    x, fsIn = librosa.load(input_file, sr = None) #loads input_file and does not resample(sr=None), loads sampling rate into fsIn
    GLow = 1.5
    GMid = 1.0
    GHigh = 0.8
    fsOut = 44100#
    xNewRate = rate_convert(x, 147, 160)
    yOut = equalizer(
        xNewRate,
        300, 2000, 20000,
        GLow, GMid, GHigh,
        fsOut
    )   
    sf.write("OutputFile.mp3", yOut, fsOut) #writes yOut at fsOut Hz to OutputFile.mp3