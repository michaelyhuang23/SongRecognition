import pyaudio
from Prediction import Predictor
import numpy as np
import time
from collections import Counter

# todo: using 1 thread for processing
if __name__ == '__main__':

    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    predictor = Predictor()
    predictor.load_data('song_recognition/database')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)


    # Store data in chunks for 3 seconds
    print('main')
    predictor.predict_realtime(state=0)
    time.sleep(0.2)
    print('Recording')
    for i in range(0, int(fs / chunk * 10)):
        data = stream.read(chunk,exception_on_overflow=False)
        data = np.frombuffer(data, np.int16)
        predictor.predict_realtime(samples=data,step_size=100,state=1)
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()
    print(predictor.predict_realtime(state=2))
    offset = 0
    for data in predictor.test_accum:
        print(offset)
        offset = predictor.process_prediction(data,offset)
        print(predictor.pollster.most_common()[:10])
    # time mismatch is created because of unsmooth transition at the edge
    print('pred: '+predictor.get_tally_winner())
    predictor.pollster = Counter()
    data = np.concatenate(predictor.test_accum)
    print(data.shape)
    print('second predict: '+predictor.predict(samples=data))
    print(predictor.pollster.most_common()[:10])
    # samples = np.hstack([np.frombuffer(i, np.int16) for i in all_data])
    # print(predictor.predict(samples=samples))

    # Stop and close the stream 