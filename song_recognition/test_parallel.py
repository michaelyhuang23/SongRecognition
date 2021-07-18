from multiprocessing import Process
import pyaudio
from Prediction import Predictor

def printHello():
    print('hello')

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

    p = Process(target=printHello)
    p.start()
    print('nihao')
    p.join()

    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()