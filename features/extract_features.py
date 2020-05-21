import numpy as np
import librosa
import os
import csv

csv_path = 'save/data.csv'


def create_features(song):
    features_set = []
    for i in range(60, 120, 15):
        y, sr = librosa.load(song, mono=True, offset=i, duration=15)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        features = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for m in mfcc:
            features += f' {np.mean(m)}'
        features_set.append(features)
    return features_set


def build_features_csv(dataset_path):
    headers = 'title chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        headers += f' mfcc{i}'
    headers += ' genre'
    headers = headers.split(' ')

    file = open(csv_path, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(headers)

    for genre in os.listdir(dataset_path):
        print(genre)
        for filename in os.listdir(f'{dataset_path}/{genre}'):
            features_set = create_features(f'{dataset_path}/{genre}/{filename}')
            title = filename.replace(' ', '').replace('\'', '')
            for fs in features_set:
                to_append = title + ' ' + fs + ' ' + genre
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(to_append.split(' '))
