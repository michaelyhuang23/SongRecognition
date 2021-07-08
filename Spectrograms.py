import numpy as np
import matplotlib.mlab as mlab
from numba import njit
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.morphology import iterate_structure


# Adrianna
# needs a sample to test

def spectrogram(samples: np.ndarray, sampling_rate=44100): 
    """
    Parameters
    ----------
    samples : numpy.ndarray, len = N
        A list containing the data from an audio sample
    sampling_rate : int
        The sampling rate of the audio (default is 44100 Hz)

    Returns
    -------
    S : 2D numpy.ndarray. shape-(F, T)
        A 2D array representing the spectrogram for the samples given (logarithmically scaled)
    freqs : numpy.ndarray, shape-(F,)
        A 1D array with the frequencies corresponding to each row of the spectrogram
    times : numpy.ndarray, shape(T,)
        A 1D array with the times corresponding to each column of the spectrogram
    """
    S, freqs, times = mlab.specgram(
        samples, 
        NFFT=4096,
        Fs=sampling_rate,
        window=mlab.window_hanning,
        noverlap=int(4096 / 2),
        mode='magnitude'
        )
    
    S = np.where(S == 0, 1E-20, S)
    S = np.log(S)
    return S, freqs, times


# the following functions are taken from PeakFinding notebook -----------------------------

@njit
def get_peaks(samples: np.ndarray, freqs: np.ndarray, times: np.ndarray, amp_min: float):
    """Peak Finding Algorithm
    Parameters
    ----------
    samples : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected.
    freqs : numpy.ndarray, shape-(N,)
        The 0-centered row indices of the local neighborhood mask 
    times : numpy.ndarray, shape-(N,)
        The 0-centered column indices of the local neighborhood mask
    amp_min : float
        All amplitudes at and below this value are excluded from being local 
        peaks.
    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location. 
    """
    peaks = []  # stores the (row, col) locations of all the local peaks
    # Iterate over the 2-D data in col-major order
    for c, r in np.ndindex(*samples.shape[::-1]):
        if samples[r, c] <= amp_min:
            continue
        # Iterating over the neighborhood centered on (r, c)
        # dr: displacement from r; dc: discplacement from c
        for dr, dc in zip(freqs, times):
            if dr == 0 and dc == 0:
                continue
            if not (0 <= r + dr < samples.shape[0]):
                # neighbor falls outside of boundary
                continue
            # mirror over array boundary
            if not (0 <= c + dc < samples.shape[1]):
                # neighbor falls outside of boundary
                continue
            if samples[r, c] < samples[r + dr, c + dc]:
                # One of the amplitudes within the neighborhood
                # is larger, thus data_2d[r, c] cannot be a peak
                break
        else:
            # if we did not break from the for-loop then (r, c) is a peak
            peaks.append((r, c))
    return peaks


def local_peak_locations(data_2d: np.ndarray, neighborhood: np.ndarray, amp_min: float):
    """
    Defines a local neighborhood and finds the local peaks
    in the spectrogram, which must be larger than the specified `amp_min` (to filter out bg).
    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected
    neighborhood : numpy.ndarray, shape-(h, w)
        A boolean mask indicating the "neighborhood" in which each
        datum will be assessed to determine whether or not it is
        a local peak. h and w must be odd-valued numbers
    amp_min : float
        All amplitudes at and below this value are excluded from being local 
        peaks.
    Returns
    -------
    peaks : List[Tuple[int, int]]
        (row, col) index pair for each local peak location.
    Notes
    -----
    The local peaks are returned in column-major order.
    """
    rows, cols = np.where(neighborhood)
    assert neighborhood.shape[0] % 2 == 1
    assert neighborhood.shape[1] % 2 == 1
    # center neighborhood indices around center of neighborhood
    rows -= neighborhood.shape[0] // 2
    cols -= neighborhood.shape[1] // 2
    return get_peaks(data_2d, rows, cols, amp_min=amp_min)


def local_peaks_mask(data: np.ndarray, cutoff: float) -> np.ndarray:
    """Find local peaks in a 2D array of data, converts them into a binary indicator.
    Parameters
    ----------
    data : numpy.ndarray, shape-(H, W)
    cutoff : float
         A threshold value that distinguishes background from foreground
    Returns
    -------
    Binary indicator, of the same shape as `data`. The value of
    1 indicates a local peak."""

    neighborhood_mask = generate_binary_structure(2, 1) 
    peak_locations = local_peak_locations(data, neighborhood_mask, cutoff) 

    peak_locations = np.array(peak_locations)

    mask = np.zeros(data.shape, dtype=bool)
    mask[peak_locations[:, 0], peak_locations[:, 1]] = 1
    return mask

# -----------------------------------------------------------------------------------------