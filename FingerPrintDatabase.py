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
        self.database[fingerprint].pop(songid)
        
    def save_fingerprint(self, fingerprint : Tuple, songid : int, time : float):
    # when fingerprint is added assign it an integer based on name
        if fingerprint in self.database:
            if songid in self.database[fingerprint]:
                self.database[fingerprint][songid].append(time)
            else:
                self.database[fingerprint][songid] = [time]
        else:
            self.database[fingerprint]={songid : [time]}
    def query_fingerprint(self, fingerprint : Tuple):
        prelim = list(self.database[fingerprint])
        retList = []
        for songid, times in prelim:
            ll = [(songid,time) for time in times]
            retList+=ll
        return retList
        
