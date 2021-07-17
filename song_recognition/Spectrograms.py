# Purpose: for creating and processing a spectrogram to extract peaks

from typing import Tuple
from numba.cuda.args import wrap_arg
import numpy as np
import matplotlib.mlab as mlab
from numba import jit, prange


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
        NFFT=1024,
        Fs=sampling_rate,
        window=mlab.window_hanning,
        noverlap=int(1024 / 2),
        mode='magnitude'
        )
    
    S = np.where(S == 0, 1E-20, S)
    S = np.log(S)
    return np.array(S), np.array(freqs), np.array(times)


# the following functions are taken from PeakFinding notebook -----------------------------
@jit(nopython=True, parallel=True)
def get_peaks(samples: np.ndarray, width: int, length: int, amp_min: float, rows: np.ndarray, cols: np.ndarray, height: float):
    """Peak Finding Algorithm
    Parameters
    ----------
    samples : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected.
    rows : numpy.ndarray, shape-(N,)
        The 0-centered row indices of the local neighborhood mask 
    cols : numpy.ndarray, shape-(N,)
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
    indices = list(np.ndindex(*samples.shape[::-1]))
    for j in range(len(indices)):
        c,r = indices[j]
        if samples[r, c] <= amp_min:
            continue
        # Iterating over the neighborhood centered on (r, c)
        # dr: displacement from r; dc: discplacement from c
        # formula: dr*length+dc*width<length*width
        for i in prange(len(rows)):
            dr = rows[i]
            dc = cols[i]
            if not (0 <= r + dr < samples.shape[0]):
                # neighbor falls outside of boundary
                continue
            # mirror over array boundary
            if not (0 <= c + dc < samples.shape[1]):
                # neighbor falls outside of boundary
                continue
            if samples[r, c] <= samples[r + dr, c + dc]:
                break 
        else:
            max_w = samples.shape[0]
            max_l = samples.shape[1]
            subregion = samples[max(0,r-width+1):min(r+width,max_w),max(0,c-length+1):min(c+length,max_l)]
            mean = np.mean(subregion)
            std = np.std(subregion)
            if samples[r,c] >= mean+height*std:
                # if we did not break from the for-loop and it reaches a certain height then (r, c) is a peak
                peaks.append((r, c))
    return peaks


def local_peak_locations(data_2d: np.ndarray, width: int, length: int, w_adj: int, l_adj: int, amp_min: float, height: float):
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
    rows = []
    cols = []
    for dr in range(1-w_adj,w_adj):
        delta = (l_adj*w_adj-dr*l_adj)//w_adj
        if (l_adj*w_adj-dr*l_adj)%w_adj==0:
            delta+=1
        for dc in range(1-delta,delta):
            if dr==0 and dc==0:
                continue
            rows.append(dr)
            cols.append(dc)
    return get_peaks(data_2d, width, length, amp_min, rows, cols, height=height)


def local_peaks(data: np.ndarray, cutoff: float, width: int = 10, length: int = 10, w_adj: int = 3, l_adj: int = 3, height: float = 1) -> np.ndarray:
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

    peak_locations = local_peak_locations(data, width, length, w_adj, l_adj, cutoff, height)

    peak_locations = np.array(peak_locations)

    return peak_locations

# -----------------------------------------------------------------------------------------