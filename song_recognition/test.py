from numpy.lib.function_base import corrcoef
from Prediction import *
from AudioProcessing import *
from Spectrograms import *

predictor = Predictor()
predictor.load_data('database')
correct_count = 0
total_count = 0

def test_song(file_path : str, songname : str, artist : str):
    global correct_count, total_count
    samples, sampling_rate = read_song(file_path)
    audios = song_split(samples, 6)
    for audio in audios:
        pred = predictor.predict(samples=audio)
        if pred==songname:
            correct_count+=1
        total_count+=1

def test_songs(*, dir_path : str):
    files = listdir(dir_path)
    for file in files:
        if 'DS_Store' in file:
            continue
        print(f'reading {file}')
        file_parts = file.split('_')
        test_song(dir_path+"/"+file, *file_parts[:2])

test_songs(dir_path='AGOP-mp3-files')

print(correct_count)
print(total_count)