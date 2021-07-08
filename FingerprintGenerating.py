# For each of the peaks, 
# Loop through until the fanout value.
# Store the fanout for that peak as a tuple, f-<, f_m+1, and t
# Store the fanout for all the peaks in a list.
def fingerprints(peaks, fanout_value):
    """
    Takes in a list of tuples (peaks) with (frequency, times).
    Returns a list of tuples (fingerprint)
    comprising the 
    fanout value for each peak found.
    """
    freqs = []
    times = []
    for i in range(len(peaks)):
        freqs.append(peaks[i][0])
        times.append(peaks[i][1])
    
    fingerprint = []
    for i in range(len(freqs) - fanout_value):
        for x in range(1, fanout_value + 1):
            fanout = (freqs[i], freqs[i+x], times[i+x]-times[i])
            fingerprint.append(fanout)

    """
    Fanout = 15
    So if i is 0-- fanouts are p1-p2, p2-3....15-
    """