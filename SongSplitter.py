from microphone import record_audio
import numpy as np
import librosa


#Splitter: Takes an array of a long (1 minute) song and splits it according to the desired length.
# def song_split(samples: np.ndarray, split_length: int):
#     splitted_song = np.array_split(samples, split_length)
#     return splitted_song
    
def analog_to_digital(path: str):
    samples, sampling_rate = librosa.load(path, sr=44100, mono=True)
    
    return samples