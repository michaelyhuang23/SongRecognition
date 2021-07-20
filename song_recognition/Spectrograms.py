# Purpose: for creating and processing a spectrogram to extract peaks

from typing import Tuple, final
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
def get_peaks(samples: np.ndarray, amp_min: float, rows: np.ndarray, cols: np.ndarray, ret_percent: float, thickness: int):
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
    peaks = np.zeros(samples.shape,dtype=np.int8)  # stores the (row, col) locations of all the local peaks
    # Iterate over the 2-D data in col-major order
    for c_s in range(0,samples.shape[1],thickness):
        locations = []
        v_vals = []
        for c in range(c_s,min(samples.shape[1],c_s+thickness)):
            for r in range(samples.shape[0]):
                if samples[r, c] <= amp_min:
                    continue
                # Iterating over the neighborhood centered on (r, c)
                # formula: dr*length+dc*width<length*width
                for i in range(len(rows)):
                    dr = rows[i]
                    dc = cols[i]
                    if not (0 <= r + dr < samples.shape[0]):
                        continue
                    if not (0 <= c + dc < samples.shape[1]):
                        continue
                    if samples[r, c] <= samples[r + dr, c + dc]:
                        break 
                else:
                    locations.append((r,c))
                    v_vals.append(samples[r,c])
        if len(v_vals)==0:
            continue
        v_vals = np.array(v_vals)
        #print(v_vals.shape, ret_percent)
        cutoff = np.percentile(v_vals,ret_percent)
        #print(cutoff)
        for (r,c),v in zip(locations,v_vals):
            if v>=cutoff:
                peaks[r,c]=1
    final_peaks = np.argwhere(np.transpose(peaks))[:,::-1]
    return final_peaks


def local_peak_locations(data_2d: np.ndarray, w_adj: int, l_adj: int, amp_min: float, ret_percent: float, thickness: int):
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
    return get_peaks(data_2d, amp_min, np.array(rows,dtype=np.int32), np.array(cols,dtype=np.int32), ret_percent, thickness)


def local_peaks(data: np.ndarray, cutoff: float, w_adj: int = 3, l_adj: int = 3, ret_percent: float=90, thickness: int=1) -> np.ndarray:
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

    peak_locations = local_peak_locations(data, w_adj, l_adj, cutoff,ret_percent,thickness)
    return np.array(peak_locations)

# -----------------------------------------------------------------------------------------