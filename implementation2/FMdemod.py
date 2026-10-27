import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import soundfile as sf

#input
fs = 256_000

def loadInput(pathName):
    xIn = scipy.io.loadmat(pathName)
    x1D = np.asarray(xIn["data"]).squeeze() #1D array
    return x1D

#limiter
def limiter(x1D):
    xLimited = x1D/np.maximum(abs(x1D), 1e-12) #prevents divide by zero
    return xLimited

def differentiator(xLimited, M = 50):
    global groupDelay
    groupDelay = M//2
    #M must be even, larger M increases accuracy but increases computation as well, M = 50 gives group delay 25
    fd = 100_000 #fd is witin nyquist frequency (fs/2 = 128_000) and leaves 28kHz in unconstrained region
    wd = 2 * np.pi * fd/fs #normalized angular freq
    h = scipy.signal.remez(M +1, [0,fd], [2*np.pi], type = "differentiator", fs=fs)
    h = h*fs
    xDiff = scipy.signal.lfilter(h, 1, xLimited)
    return xDiff

def compConjAndDelay(xLimited, Delay):
    #get Conjugate of xLim
    xConj = np.conj(xLimited)
    #Delay conj by Delay
    xDelayed = np.zeros_like(xConj) #array of zeros with size xConj
    xDelayed[Delay:] = xConj[:-Delay] #put Values of xConj into xDelayed after Delay
    return xDelayed

def mult(xDiff, xDelayed):
    return xDiff * xDelayed

def Im(In):
    return np.imag(In)

def deEmphasisFilter(yIn, tauD = 75e-6):
    b = [1,1]
    a = [1 + 2*tauD*fs, 1-2*tauD*fs]
    return scipy.signal.lfilter(b, a, yIn)

def LPF(yIn, M = 300):
    fPass = 15_000
    fStop = 18_000
    #get coefficients for LPF w Parks McClellan
    h = scipy.signal.remez(M + 1,
                           [0,fPass, fStop, fs/2],
                            [1,0], #passgain of 1, stopband gain of 0
                            weight = [1,10], 
                            fs=fs
                            )
    return scipy.signal.lfilter(h, 1, yIn)

def downSample(yIn):
    return yIn[::4] #keep every 4th value

def deMod(pathName):
    xIn = loadInput(pathName)
    xL = limiter(xIn)
    xDiff = differentiator(xL)
    xConjDelay = compConjAndDelay(xL, groupDelay)
    xMultiplied = mult(xDiff, xConjDelay)
    yIn = Im(xMultiplied)
    yDeEmph = deEmphasisFilter(yIn)
    yLPF = LPF(yDeEmph)
    return downSample(yLPF)

if __name__ == '__main__':
    inputFiles = [
    "baseband1.mat", "baseband2.mat", "baseband3.mat",
    "baseband4.mat", "baseband5.mat", "baseband6.mat",
    "baseband7.mat"
    ]
    for file in inputFiles:
        yOut = deMod(file)
        outputName = file.replace(".mat", ".wav")
        #audio was clipping without this this normalizes output
        maxVal = np.max(np.abs(yOut))
        yOut = 0.99 * yOut / maxVal
        sf.write(outputName, yOut, 64_000) #writes yOut at fsOut Hz to OutputFile.mp3
