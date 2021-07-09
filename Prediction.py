from typing import List
from os import listdir
from AudioProcessing import *
from collections import Counter
import numpy as np
from FingerPrintDatabase import FingerPrintDatabase, get_fingerprints
from SongDatabase import *
from Spectrograms import spectrogram, local_peaks


# main prediction functions should be here
# it uses other classes for the prediction
class Predictor:
    def __init__(self) -> None:
        self.fingerprints = FingerPrintDatabase()
        self.songs = SongDatabase()
        self.pollster = Counter()
        self.percent_thres = 75
        self.fanout_value = 15
    
    def tally(self, songs : List):
        self.pollster.update(Counter(songs))

    def get_tally_winner(self):
        return self.songs.id2name[self.pollster.most_common()[0]]
        
    def add_song(self, file_path : str, songname : str, artist : str):
        audio, sampling_rate = read_song(file_path)
        # these should read in discrete digital data
        spectro, freqs, times = spectrogram(audio)
        # returns (Frequency, Time) data
        thres = np.percentile(spectro, self.percent_thres)
        peaks = local_peaks(spectro, thres)
        self.songs.save_song(peaks, songname, artist)
        fingerprints, times = get_fingerprints(peaks,self.fanout_value)
        for fingerprint, time in zip(fingerprints,times):
            self.fingerprints.save_fingerprint(fingerprint, songname, time)
    
    def add_songs(self, *, dir_path : str):
        files = listdir(dir_path)
        for file in files:
            file_parts = file.split('_')
            self.add_song(file, *file_parts[:2])
    
    def delete_song(self, songname : str):
        self.songs.delete_song(songname, self.fingerprints)

    def predict(self, *, file_path : str, record_time : float):
        # this is meant to be a function that indicates the general structure of the program
        # it uses some pseudo functions that should be implemented
        fan_out = 15
        if file_path==None:
            audio = record_song(record_time)
        else:
            audio, sampling_rate = read_song(file_path)
        # these should read in discrete digital data
        spectro, freqs, times = spectrogram(audio)
        # returns (Frequency, Time) data
        thres = np.percentile(spectro, self.percent_thres)
        peaks = local_peaks(spectro,thres)
        # returns a list of peaks (f, t)
        fingerprints = get_fingerprints(peaks, fan_out)
        for fingerprint in fingerprints:
            songs = self.fingerprints.query_fingerprint(fingerprint)
            self.tally(songs)
        return self.get_tally_winner()


predictor = Predictor()
predictor.add_song('Imperial-March_starwars.mp3','Imperial-March','John Williams')
print(len(predictor.fingerprints.database))
predictor.delete_song('Imperial-March')