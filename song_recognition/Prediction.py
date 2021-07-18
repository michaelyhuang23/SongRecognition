from typing import List
from os import listdir
from AudioProcessing import *
from collections import Counter
import numpy as np
from FingerPrintDatabase import FingerPrintDatabase, get_fingerprints
from SongDatabase import *
from Spectrograms import spectrogram, local_peaks
from multiprocessing import Process
import time
'''
potential features:
- real time audio
- ratio for more accurate predictions
- website?

'''

# main prediction functions should be here
# it uses other classes for the prediction
# todo: add background cancelling even in song file
processes = []

class Predictor:
    def __init__(self) -> None:
        global processes
        self.fingerprints = FingerPrintDatabase()
        self.songs = SongDatabase()
        self.pollster = Counter()
        self.percent_thres = 0
        self.store_fanout_value = 2
        self.pred_fanout_value = 30
        self.thres_ratio = 1.5
        self.store_width = 3
        self.store_length = 3
        self.store_perc = 98
        self.thickness = 10
        self.pred_length = 3
        self.pred_width = 3
        self.pred_perc = 80
        self.realtime_accum = []
        processes = []
    
    def tally(self, songs : List, time0):
        if not songs is None: 
            self.pollster.update(Counter([(song, time-time0) for song, time in songs]))

    def get_tally_winner(self):
        # print(self.pollster.most_common()[:4])
        if len(self.pollster)==0:
            return 'None'
        common, ratio = self.confidence_ratio()
        self.pollster = Counter()
        if ratio < self.thres_ratio:
            return 'None'
        return common
        
    def confidence_ratio(self):
        # uses the built in counters to find an approximate ratio for confident guesses
        counter = self.pollster.most_common()
        # takes the "most common" song
        most_common = counter[0][0][0]
        print(counter[0][1])
        common_two = None
        for index in range(1, len(counter)):
            if counter[index][0][0] != most_common:
                common_two = index
                break
        if common_two is None:
            ratio = 1e9
        else:
            ratio = counter[0][1] / counter[common_two][1]
        return self.songs.id2name[most_common], ratio

    def add_song(self, file_path : str, songname : str, artist : str):
        if songname in self.songs.name2id:
            return
        audio, sampling_rate = read_song(file_path)
        # these should read in discrete digital data
        spectro, freqs, times = spectrogram(audio)
        # returns (Frequency, Time) data
        thres = np.percentile(spectro, self.percent_thres)
        peaks = local_peaks(spectro, thres, self.store_width, self.store_length, self.store_perc, self.thickness)
        print(len(peaks))
        self.songs.save_song(peaks, songname, artist, self.fingerprints, self.store_fanout_value)
    
    def add_songs(self, *, dir_path : str):
        files = listdir(dir_path)
        for file in files:
            if 'DS_Store' in file:
                continue
            print(f'reading {file}')
            file_parts = file.split('_')
            self.add_song(dir_path+"/"+file, *file_parts[:2])
    
    def delete_song(self, songname : str):
        self.songs.delete_song(songname, self.store_fanout_value,self.fingerprints)

    def save_data(self, dir_path):
        self.songs.save_data(dir_path+"/songs")
        self.fingerprints.save_data(dir_path+"/fingerprints")
    
    def load_data(self, dir_path):
        self.songs.load_data(dir_path+"/songs")
        self.fingerprints.load_data(dir_path+"/fingerprints")

    def process_prediction(self, audio : np.ndarray):
        # these should read in discrete digital data
        spectro, freqs, times = spectrogram(audio)
        #print(len(freqs),len(times))
        # returns (Frequency, Time) data
        thres = np.percentile(spectro, self.percent_thres)
        #print(spectro[2:10,3:30])
        peaks = local_peaks(spectro, thres, self.pred_width, self.pred_length, self.pred_perc)
        #print(len(peaks))
        # returns a list of peaks (f, t)
        fingerprints, times = get_fingerprints(peaks, self.pred_fanout_value)
        #print(len(fingerprints))
        #print(fingerprints[:])
        for fingerprint, time in zip(fingerprints,times):
            #print(fingerprint)
            songs = self.fingerprints.query_fingerprint(fingerprint)
            self.tally(songs, time)

    def predict(self, *, file_path : str = '', record_time : float = 0, samples : np.ndarray = None):
        # this is meant to be a function that indicates the general structure of the program
        # it uses some pseudo functions that should be implemented
        if file_path!='':
            audio, sampling_rate = read_song(file_path)
        elif record_time > 0:
            audio = record_song(record_time)
        else:
            audio = samples
        self.process_prediction(audio)
        ret = self.get_tally_winner()
        if ret=='None':
            return "Oops, did not find this song!"
        else:
            return ret


    def predict_realtime(self, file_path: str='', samples: np.ndarray = None, step_size: int = 1, state:int = 0):
        global processes
        if state == 0:
            if samples is None:
                audio, sampling_rate = read_song(file_path)
            else:
                audio = samples
            self.realtime_accum.append(audio)
            if len(self.realtime_accum)>=step_size:
                data = np.concatenate(self.realtime_accum)
                self.realtime_accum = []
                #print(data.shape)
                process = Process(target=self.process_prediction,args=(data,))
                process.start()
                processes.append(process)
        else:
            [process.join() for process in processes]
            processes = []
            ret = self.get_tally_winner()
            if ret=='None':
                return "Oops, did not find this song!"
            else:
                return ret


# predictor = Predictor()
# predictor.load_data('song_recognition/database')
# print(predictor.predict(record_time=5))


# peaks = predictor.songs.database[predictor.songs.name2id['Imperial March']]["peaks"]
# fingerprints, times = get_fingerprints(peaks,2)
# print(fingerprints[:])

# predictor.add_songs(dir_path='AGOP-mp3-files')
# predictor.save_data('song_recognition/database')

# first_print = (202, 831, 0)
# print(predictor.fingerprints.database[first_print])
# print(predictor.fingerprints.query_fingerprint(first_print))

# predictor.delete_song('Imperial-March')
# print(len(predictor.fingerprints.database))