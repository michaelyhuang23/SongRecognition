# SongRecognition using fingerprints

This is a fork of the SongRecognition project created by A-Group-of-Pandas. It adds the feature of realtime detection using parallel processing. I also enhanced the detection algorithm to reduce the size of the database to 1/15 of its original size while maintaining roughly equal accuracy. 

The fingerprint detection method has proven to work well with low noise condition. For conditions where the noise is louder than the song, it may fail.

To run the program, follow the examples given in song_recognition/run_example.py or song_recognition/run_realtime_example.py