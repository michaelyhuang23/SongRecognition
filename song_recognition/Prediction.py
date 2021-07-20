from typing import List
from os import listdir
from AudioProcessing import *
from collections import Counter
import numpy as np
from FingerPrintDatabase import FingerPrintDatabase, get_fingerprints
from SongDatabase import *
from Spectrograms import spectrogram, local_peaks
from multiprocessing import Process, Queue, Value
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

class Predictor:
    def __init__(self) -> None:
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
        self.pred_perc = 70
        self.time_diff_grain = 10
        self.realtime_accum = []
        self.test_accum = []
    
    def tally(self, songs : List, time0):
        if not songs is None: 
            self.pollster.update(Counter([(song, int((time-time0)/self.time_diff_grain)) for song, time in songs]))

    def get_tally_winner(self):
        # print(self.pollster.most_common()[:4])
        if len(self.pollster)==0:
            return -1
        common, ratio = self.confidence_ratio()
        if ratio < self.thres_ratio:
            return -1
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
        return most_common, ratio

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

    def process_prediction(self, audio : np.ndarray, offset : int):
        # these should read in discrete digital data
        spectro, freqs, times = spectrogram(audio)
        print(len(freqs),len(times))
        time_len = len(times)
        # returns (Frequency, Time) data
        thres = np.percentile(spectro, self.percent_thres)
        peaks = local_peaks(spectro, thres, self.pred_width, self.pred_length, self.pred_perc)
        print(peaks.shape[0])
        # returns a list of peaks (f, t)
        fingerprints, times = get_fingerprints(peaks, self.pred_fanout_value)
        print(len(fingerprints))
        for fingerprint, time in zip(fingerprints,times):
            songs = self.fingerprints.query_fingerprint(fingerprint)
            self.tally(songs, time+offset)
        return time_len+offset+1

    def predict(self, *, file_path : str = '', record_time : float = 0, samples : np.ndarray = None):
        self.pollster = Counter()
        # this is meant to be a function that indicates the general structure of the program
        # it uses some pseudo functions that should be implemented
        if file_path!='':
            audio, sampling_rate = read_song(file_path)
        elif record_time > 0:
            audio = record_song(record_time)
        else:
            audio = samples
        self.process_prediction(audio,0)
        ret = self.get_tally_winner()
        if ret==-1:
            return "Oops, did not find this song!"
        else:
            return self.songs.id2name[ret]

    def process_prediction_realtime(self, queue, ret):
        offset = 0
        tmp_ret = -1
        all_data = None
        while True:
            self.pollster = Counter()
            data = queue.get()
            if data is None:
                break
            if all_data is None:
                all_data = data
            else:
                all_data = np.concatenate([all_data,data])
            print(all_data.shape)
            self.process_prediction(all_data, 0)
            tmp_ret = self.get_tally_winner()
            print(tmp_ret)
            if tmp_ret != -1:
                ret.value = tmp_ret
                break
        del all_data
        ret.value = self.get_tally_winner()

    def predict_realtime(self, file_path: str='', samples: np.ndarray = None, step_size: int = 1, state:int = 1):
        if state == 0:
            self.queue = Queue()
            self.realtime_ret = Value('i',-1)
            self.process = Process(target=self.process_prediction_realtime, args=(self.queue,self.realtime_ret,))
            self.process.daemon = True
            self.process.start()
        elif state == 1:
            if self.realtime_ret.value != -1:
                self.process.join()
                return self.songs.id2name[self.realtime_ret.value]
            if samples is None:
                audio, sampling_rate = read_song(file_path)
            else:
                audio = samples
            self.realtime_accum.append(audio)
            if len(self.realtime_accum)>=step_size:
                data = np.concatenate(self.realtime_accum)
                self.realtime_accum = []
                self.queue.put(data)
                self.test_accum.append(audio)
        else:
            self.queue.put(None)
            self.process.join()
            if self.realtime_ret.value==-1:
                return "Oops, did not find this song!"
            else:
                return self.songs.id2name[self.realtime_ret.value]


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