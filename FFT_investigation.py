import numpy as np
import scipy as sp
import time
import pyfftw

n_bins = 1024
n = 1000000
seq = np.random.uniform(-5, 5, size=n_bins)
print(n_bins)

def gf3_fft(seq: np.ndarray, method: str, fft1D_object=None):
    if method == "numpy":
        return np.fft.fft(seq, n_bins)
    if method == "scipy":
        return sp.fft.fft(seq)
    if method == "scipypack":
        return sp.fftpack.fft(seq)
    if method == "fftw_numpy":
        return pyfftw.interfaces.numpy_fft.fft(seq, n_bins)
    if method == "fftw":
        return fft1D_object(seq)
    
def gf3_ifft(seq: np.ndarray, method: str, ifft1D_object=None):
    if method == "numpy":
        return np.fft.ifft(seq, n_bins)
    if method == "scipy":
        return sp.fft.ifft(seq)
    if method == "scipypack":
        return sp.fftpack.ifft(seq)
    if method == "fftw_numpy":
        return pyfftw.interfaces.numpy_fft.ifft(seq, n_bins)
    if method == "fftw":
        return ifft1D_object(seq)

start = time.time()
for _ in range(n):
    numpy_fft = gf3_fft(seq=seq, method="numpy")
end = time.time()
print("numpy:", end - start)

start = time.time()
for _ in range(n):
    scipy_fft = gf3_fft(seq=seq, method="scipy")
end = time.time()
print("scipy:", end - start)

start = time.time()
for _ in range(n):
    scipy_pack_fft = gf3_fft(seq=seq, method="scipypack")
end = time.time()
print("scipypack:", end - start)

start = time.time()
for _ in range(n):
    fftw_np_fft = gf3_fft(seq=seq, method="fftw_numpy")
end = time.time()
print("fftw_numpy:", end - start)

start = time.time()
in1D_array = pyfftw.empty_aligned(n_bins, dtype='complex128')
out1D_array = pyfftw.empty_aligned(n_bins, dtype='complex128')
fft1D_object = pyfftw.FFTW(in1D_array, out1D_array, flags=('FFTW_MEASURE',))
for _ in range(n):
    fftw_fft = gf3_fft(seq=seq, method="fftw", fft1D_object=fft1D_object)
end = time.time()
print("fftw:", end - start)

compare = np.round(fftw_np_fft,10) == np.round(scipy_fft, 10)
for val in compare:
    if val == False:
        print(val)

# -----------INVERSE-----------------------------------------------------

start = time.time()
for _ in range(n):
    numpy_fft = gf3_ifft(seq=seq, method="numpy")
end = time.time()
print("numpy IFFT:", end - start)

start = time.time()
for _ in range(n):
    scipy_fft = gf3_ifft(seq=seq, method="scipy")
end = time.time()
print("scipy IFFT:", end - start)

start = time.time()
for _ in range(n):
    scipy_pack_fft = gf3_ifft(seq=seq, method="scipypack")
end = time.time()
print("scipypack IFFT:", end - start)

start = time.time()
for _ in range(n):
    fftw_np_fft = gf3_ifft(seq=seq, method="fftw_numpy")
end = time.time()
print("fftw_numpy IFFT:", end - start)

start = time.time()
in1D_array = pyfftw.empty_aligned(n_bins, dtype='complex128')
out1D_array = pyfftw.empty_aligned(n_bins, dtype='complex128')
ifft1D_object = pyfftw.FFTW(in1D_array, out1D_array, flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD')
for _ in range(n):
    fftw_fft = gf3_ifft(seq=seq, method="fftw", fft1D_object=ifft1D_object)
end = time.time()
print("fftw IFFT:", end - start)