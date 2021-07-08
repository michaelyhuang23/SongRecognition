# Purpose: function for storing fingerprints to database and querying a specific fingerprint

from typing import List, Tuple
import numpy as np
import pickle


class SongDatabase:
    def __init__(self):
       self.database = {}
       self.id2name = []
       self.name2id = {}

    def load_data(self, file_path):
        with open(file_path, mode="rb") as f:
            self.database = pickle.load(f)

    def save_data(self, file_path):
        with open(file_path, mode="wb") as f:
            pickle.dump(self.database, f)
        
    def save_song(self, peaks : List, songname : str, artist : str):
        if not songname in self.database:
            self.id2name.append(songname)
            id = len(self.id2name)-1
            self.name2id[songname] = id
            self.database[id] = dict(peaks=peaks, artist=artist)
            
    def delete_song(self, songname, finger_print_database):
        songid = self.name2id[songname]
        peaks = self.database[songid]["peaks"]
        self.database.pop(songid, None)
        for p in peaks:
            fingerprint, time = get_fingerprint(p)
            if fingerprint is None:
                continue
            finger_print_database.delete_song(songid, fingerprint)
    
    def list_songs(self):
        for songid in self.database.keys():
            print(self.id2name[songid] + " by: " + self.database[songid]["artist"])
