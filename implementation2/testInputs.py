
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal

inputFiles = [
    "baseband1.mat", "baseband2.mat", "baseband3.mat",
    "baseband4.mat", "baseband5.mat", "baseband6.mat",
    "baseband7.mat"
]

fs = 256_000

def testInputs():
    for file in inputFiles:
        xIn = scipy.io.loadmat(file)

        # Find the signal variable in the MATLAB file
        keys = [k for k in xIn.keys()
                if not k.startswith("__")]

        print(file, keys)

        # Extract signal and convert to 1D
        x1D = np.asarray(xIn[keys[0]]).squeeze()

        # Limiter
        magnitude = np.abs(x1D)
        xLimited = x1D / np.maximum(magnitude, 1e-12)

        # FFT
        N = len(xLimited)
        X = np.fft.fft(xLimited)
        X = np.fft.fftshift(X)

        f = np.fft.fftshift(
            np.fft.fftfreq(N, d=1/fs)
        )

        # Plot spectrum
        plt.figure()

        plt.plot(
            f / 1000,
            20 * np.log10(np.abs(X) / N + 1e-12)
        )

        plt.xlabel("Frequency (kHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title(f"Spectrum of {file}")
        plt.grid()

        plt.savefig(file.replace(".mat", "_spectrum.png"))
        plt.close()

def main():
    testInputs()

main()
