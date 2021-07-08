import Spectrograms
import librosa

def load_song(file_path: str):
    samples, sampling_rate = librosa.load(file_path, sr=44100, mono=True)
    
    return samples, sampling_rate

def record(record_time):
    frames, sample_rate = record_audio(record_time)
    samples = np.hstack([np.frombuffer(i, np.int16) for i in frames])

    return samples

samples, sampling_rate = load_song("Imperial-March_starwars.mp3")
spectro, freqs, times = Spectrograms.spectrogram(samples)
print(spectro)

