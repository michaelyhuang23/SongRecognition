from typing import List
from os import listdir
import librosa
from microphone import record_audio
from collections import Counter
import numpy as np
from FingerPrintDatabase import FingerPrintDatabase
from SongDatabase import SongDatabase
from Spectrograms import spectrogram


# main prediction functions should be here
# it uses other classes for the prediction
class Predictor:
    def __init__(self) -> None:
        self.fingerprints = FingerPrintDatabase()
        self.songs = SongDatabase()
        self.pollster = Counter()
    
    def tally(self, songs : List):
        self.pollster.update(Counter(songs))

    def get_tally_winner(self):
        return self.songs.id2name[self.pollster.most_common()[0]]
        
    def add_song(self, file_path : str, songname : str, artist : str):
        audio = read_audio(file_path)
        # these should read in discrete digital data
        spectro = spectrogram(audio)
        # returns (Frequency, Time) data
        peaks = get_peaks(spectro)
        self.songs.save_song(peaks, songname, artist)
        for peak in peaks:
            fingerprint, time = get_fingerprint(peak)
            if fingerprint is None:
                continue
            self.fingerprints.save_fingerprint(fingerprint, songname, time)
    
    def add_songs(self, *, dir_path : str):
        files = listdir(dir_path)
        for file in files:
            file_parts = file.split('_')
            self.add_song(file, *file_parts[:2])

    def predict(self, *, file_path : str, record_time : float):
        # this is meant to be a function that indicates the general structure of the program
        # it uses some pseudo functions that should be implemented
        if file_path==None:
            audio = record_audio(record_time)
        else:
            audio = read_audio(file_path)
        # these should read in discrete digital data
        spectro = spectrogram(audio)
        # returns (Frequency, Time) data
        peaks = get_peaks(spectro)
        # returns a list of peaks (f, t)
        for peak in peaks:
            fingerprint, time = get_fingerprint(peak)
            if fingerprint is None:
                continue
            songs = self.fingerprints.query_fingerprint(fingerprint)
            self.tally(songs)
        return self.get_tally_winner()



    