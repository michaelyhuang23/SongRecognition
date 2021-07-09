# Purpose: function for storing fingerprints to database and querying a specific fingerprint

from typing import Tuple
import numpy as np
import pickle


# database obj --> dict
# save_fingerprint function --> Michael
# query_fingerprint function -->
# save function
# load function
class FingerPrintDatabase:
    def __init__(self):
        self.database = {}
        # {key: fingerprint value: {key: songname, value: list(time)}}

    def load_data(self, file_path):
        with open(file_path, mode="rb") as f:
            self.database = pickle.load(f)

    def save_data(self, file_path):
        with open(file_path, mode="wb") as f:
            pickle.dump(self.database, f)

    def delete_song(self, songid, fingerprint):
        del self.database[fingerprint][songid]
        if len(self.database[fingerprint])==0:
            del self.database[fingerprint]
        
    def save_fingerprint(self, fingerprint : Tuple, songid : int, time : float):
        if fingerprint in self.database:
            if songid in self.database[fingerprint]:
                self.database[fingerprint][songid].append(time)
            else:
                self.database[fingerprint][songid] = [time]
        else:
            self.database[fingerprint]={songid : [time]}
    def query_fingerprint(self, fingerprint : Tuple):
        if fingerprint not in self.database:
            return None
        prelim = self.database[fingerprint]
        retList = []
        for songid, times in prelim.items():
            ll = [(songid,time) for time in times]
            retList+=ll
        return retList

def get_fingerprints(peaks, fanout_value):
    """
    Takes in a list of tuples (peaks) with (frequency, times).
    Returns a list of tuples (fingerprints)
    comprising the fanout value for each peak found
    and a list of time values for each fingerprint.
    """
    freqs = [peaks[i][0] for i in range(len(peaks))]
    times = [peaks[i][1] for i in range(len(peaks))]

    fingerprints = []
    time_values = []
    for i in range(len(freqs) - fanout_value):
        fingerprint = [(freqs[i],freqs[i+x],times[i+x]-times[i]) for x in range(1,fanout_value+1)]
        fingerprints += fingerprint
        time_values += [times[i] for x in range(1,fanout_value+1)]

    return fingerprints, time_values

# making the comparing database and pickiling it
fingerprint_database = FingerPrintDatabase
        
