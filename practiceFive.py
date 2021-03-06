import numpy as np

"""
Practice 5 is about the following:
-> Constructing DFT matrix of size n.
-> Finding out if the matrix is unitary
-> Creating delta impulse signals and perform the fourier transform on each signal.
-> Shuffling elements of data using bit reversal of list index.
-> Performing real-valued discrete Fourier transform of data using fast Fourier transform.
-> Generating tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.
-> Filtering high frequencies above bandlimit.
"""

def dft_matrix(n: int) -> np.ndarray:
    """
    Construct DFT matrix of size n.

    Arguments:
    n: size of DFT matrix

    Return:
    F: DFT matrix of size n

    Forbidden:
    - numpy.fft.*
    """
    # initializing the matrix with proper size
    F = np.zeros((n, n), dtype='complex128')

    # creating the principal term for DFT matrix
    w = np.exp(-2 * np.pi * 1j / n)

    # filling the matrix with values
    indices = np.linspace(0, n, num=n, endpoint=False)
    column_indices, rows_indices = np.meshgrid(indices, indices)
    F = np.power(w, column_indices * rows_indices)

    # normalizing the dft matrix
    F = 1 / np.sqrt(n) * F

    return F


def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if the passed in matrix of size (n times n) is unitary.

    Arguments:
    matrix: the matrix which is checked

    Return:
    unitary: True if the matrix is unitary
    """
    unitary = True
    # checking if F is unitary, if not return false
    hermitian_matrix = matrix.conjugate()
    inverse_matrix = np.linalg.inv(matrix)

    if np.allclose(hermitian_matrix, inverse_matrix):
        pass
    else:
        unitary = False

    return unitary


def create_harmonics(n: int = 128) -> (list, list):
    """
    Create delta impulse signals and perform the fourier transform on each signal.

    Arguments:
    n: the length of each signal

    Return:
    sigs: list of np.ndarrays that store the delta impulse signals
    fsigs: list of np.ndarrays with the fourier transforms of the signals
    """

    # list to store input signals to DFT
    sigs = []
    # Fourier-transformed signals
    fsigs = []

    # creating signals and extract harmonics out of DFT matrix
    Omega = dft_matrix(n)

    for i in range(n):
        signal = np.zeros(n)
        signal[i] = 1
        sigs.append(signal)
        ft_signal = np.dot(Omega, signal)
        fsigs.append(ft_signal)

    return sigs, fsigs


def shuffle_bit_reversed_order(data: np.ndarray) -> np.ndarray:
    """
    Shuffle elements of data using bit reversal of list index.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    data: shuffled data array
    """

    # implementing shuffling by reversing index bits
    data_new = np.zeros(data.size, dtype='complex128')

    for i in range(data.size):
        B = f'{i:b}'
        B = B.zfill(int(np.log2(data.size)))
        r_B = B[::-1]
        D = int(r_B, 2)
        data_new[D] = data[i]

    return data_new


def fft(data: np.ndarray) -> np.ndarray:
    """
    Perform real-valued discrete Fourier transform of data using fast Fourier transform.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    fdata: Fourier transformed data

    Note:
    This is not an optimized implementation but one to demonstrate the essential ideas
    of the fast Fourier transform.

    Forbidden:
    - numpy.fft.*
    """

    fdata = np.asarray(data, dtype='complex128')
    n = fdata.size

    # check if input length is power of two
    if not n > 0 or (n & (n - 1)) != 0:
        raise ValueError

    # first step of FFT: shuffle data
    f = shuffle_bit_reversed_order(fdata)

    # second step, recursively merge transforms

    for m in range(np.log2(n).astype(int)):
        for k in range(np.power(2, m)):
            for i in range(k, n, np.power(2, m + 1)):
                j = np.power(2, m) + i
                w = np.exp(-2 * np.pi * 1j * k / np.power(2, m + 1))
                p = w * f[j]
                f[j] = f[i] - p
                f[i] = f[i] + p

    # normalizing the fft signal
    f = (1 / np.sqrt(n)) * f

    return f


def generate_tone(f: float = 261.626, num_samples: int = 44100) -> np.ndarray:
    """
    Generate tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.

    Arguments:
    f: frequency of the tone

    Return:
    data: the generated signal
    """

    # sampling range
    x_min = 0.0
    x_max = 1.0

    data = np.zeros(num_samples)

    # generating the sine wave with proper frequency

    for i in range(num_samples):
        data[i] = np.sin(2 * np.pi * f * i * (1 / (num_samples - 1)))

    return data


def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100) -> np.ndarray:
    """
    Filter high frequencies above bandlimit.

    Arguments:
    adata: data to be filtered
    bandlimit: bandlimit in Hz above which to cut off frequencies
    sampling_rate: sampling rate in samples/second

    Return:
    adata_filtered: filtered data
    """

    # translate bandlimit from Hz to dataindex according to sampling rate and data size
    bandlimit_index = int(bandlimit * adata.size / sampling_rate)

    # computing the Fourier transform of input data
    f = np.fft.fft(adata)

    # setting high frequencies above bandlimit to zero, make sure the almost symmetry of the transform is respected.
    for i in range(bandlimit_index + 1, f.size - bandlimit_index):
        f[i] = 0

    # computing the inverse transform and extract real component
    adata_filtered = np.fft.ifft(f).real

    return adata_filtered


if __name__ == '__main__':
    print("It's on GitHub!")
