import numpy as np
import librosa
import os
import csv


def create_features(song):
    y, sr = librosa.load(song, mono=True)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rmse(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    features = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        features += f' {np.mean(e)}'
    return features


def build_csv():
    header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    headers = header.split(' ')
    file = open('../data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(headers)

    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for genre in genres:
        print(genre)
        for filename in os.listdir(f'D:/dataset/{genre}'):
            to_append = create_features(f'D:/dataset/{genre}/{filename}')
            to_append += ' ' + genre
            file = open('../data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split(' '))
