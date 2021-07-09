# Purpose: function for storing fingerprints to database and querying a specific fingerprint

from typing import List, Tuple
import numpy as np
import pickle
from FingerPrintDatabase import * 


class SongDatabase:
    def __init__(self):
       self.database = {}
       self.id2name = []
       self.name2id = {}

    def load_data(self, file_path):
        with open(file_path, mode="rb") as f:
            self.database = pickle.load(f)
        with open(file_path+"_id2name", mode="rb") as f:
            self.id2name = pickle.load(f)
        with open(file_path+"_name2id", mode="rb") as f:
            self.name2id = pickle.load(f)

    def save_data(self, file_path):
        with open(file_path, mode="wb") as f:
            pickle.dump(self.database, f)
        with open(file_path+"_id2name", mode="wb") as f:
            pickle.dump(self.id2name, f)
        with open(file_path+"_name2id", mode="wb") as f:
            pickle.dump(self.name2id, f)
        
    def save_song(self, peaks : List, songname : str, artist : str, fingerprints_database, fanout_value):
        if not songname in self.database:
            self.id2name.append(songname)
            id = len(self.id2name)-1
            self.name2id[songname] = id
            self.database[id] = dict(peaks=peaks, artist=artist)
            fingerprints, times = get_fingerprints(peaks,fanout_value)
            for fingerprint, time in zip(fingerprints,times):
                fingerprints_database.save_fingerprint(fingerprint, id, time)
            
    def delete_song(self, songname, fanout_value, finger_print_database):
        songid = self.name2id[songname]
        peaks = self.database[songid]["peaks"]
        self.database.pop(songid, None)
        fingerprints, times = get_fingerprints(peaks,fanout_value)
        for fingerprint in fingerprints:
            finger_print_database.delete_song(songid, fingerprint)
    
    def list_songs(self):
        for songid in self.database.keys():
            print(f'{self.id2name[songid]} by: {self.database[songid]["artist"]}')
