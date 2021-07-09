import Spectrograms
import librosa
import matplotlib.pyplot as plt
import numpy as np
from microphone import record_audio


# Takes in audio and returns samples

def record_song(record_time):
    frames, sample_rate = record_audio(record_time)
    samples = np.hstack([np.frombuffer(i, np.int16) for i in frames])
    return samples

def read_song(file_path):
    samples, sampling_rate = librosa.load(file_path, sr=44100, mono=True)
    return samples, sampling_rate
    
#Splitter: Takes an array of a long (1 minute) song and splits it according to the desired length.
def song_split(samples, split_length):
    splitted_song = np.array_split(samples, split_length)
    return splitted_song


# test --------------------------------------------------------------------------------------------
# #Putting in the song
# samples, sampling_rate = read_song("Imperial-March_starwars.mp3")
# spectro, freqs, times = Spectrograms.spectrogram(samples)
# spectro = np.array(spectro)
# print(spectro.shape)

# fig, ax = plt.subplots()
# # ax.specgram()
# plt.imshow(
#     spectro[...,np.newaxis],
#     origin="lower",
#     # aspect=aspect_ratio,
#     # extent=extent,
#     interpolation="bilinear",
# )
# plt.xlabel("time (sec)")
# plt.ylabel("frequency (Hz)")

# # Gets the spectogram peaks.
# thres = np.percentile(spectro, 75)
# print(thres)
# peaks = Spectrograms.local_peaks(spectro, thres)
# print(peaks[:])

# plt.scatter(peaks[:, 1], peaks[:, 0], c="b", s=2)
# plt.ylabel("rows [freqs]")
# plt.xlabel("columns [times]")

# plt.show()