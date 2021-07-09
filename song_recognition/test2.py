from Prediction import *
from AudioProcessing import *
from Spectrograms import *

predictor = Predictor()
predictor.load_data('song_recognition/database')
correct_count = 0
total_count = 0

print(predictor.predict(record_time=5))