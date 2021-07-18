import pyaudio
from Prediction import Predictor
import numpy as np

# todo: using 1 thread for processing
if __name__ == '__main__':

    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    predictor = Predictor()
    predictor.load_data('song_recognition/database')

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)


    # Store data in chunks for 3 seconds
    print('main')

    for i in range(0, int(fs / chunk * 5)):
        data = stream.read(chunk,exception_on_overflow=False)
        data = np.frombuffer(data, np.int16)
        predictor.predict_realtime(samples=data,step_size=20,state=0)
    print(predictor.predict_realtime(state=1))
    # samples = np.hstack([np.frombuffer(i, np.int16) for i in all_data])
    # print(predictor.predict(samples=samples))

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()