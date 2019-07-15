import os
import pickle
import subprocess
import numpy as np
from random import shuffle
from PIL import Image

genres = 'blues country jazz metal rock'.split()


def generate_spectrograms():
    for genre in genres:
        print(genre)
        genre_folder = f'D:/dataset2/{genre}/'
        for filename in os.listdir(genre_folder):
            subprocess.run(
                ['sox', genre_folder + filename, '-n', 'spectrogram', '-Y', '200', '-X', '50', '-m', '-r', '-o',
                 f'../dataset/specs/{genre}/{filename[:-4]}.png'])


def create_slices():
    for genre in genres:
        print(genre)
        genre_folder = f'../dataset/specs/{genre}/'
        for filename in os.listdir(genre_folder):
            img = Image.open(genre_folder + filename)
            width, height = img.size
            n_samples = int(width / 128)
            for i in range(n_samples):
                start_pixel = i * 128
                img_tmp = img.crop((start_pixel, 1, start_pixel + 128, 128 + 1))
                img_tmp.save(f'../dataset/slices/{genre}/{filename[:-4]}-{i:02d}.png')


def create_dataset():
    training_data = []
    for genre_num, genre in enumerate(genres):
        print(genre)
        genre_folder = f'../dataset/slices/{genre}/'
        for filename in os.listdir(genre_folder):
            img = Image.open(genre_folder + filename)
            img_data = np.asarray(img, dtype=np.uint8).reshape(128, 128, 1)
            img_data = img_data / 255.
            training_data.append([img_data, genre_num])
    shuffle(training_data)
    print('shuffle done')

    X = []
    y = []
    for features, label in training_data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, 128, 128, 1)

    print('writing pickles...')
    pickle_out = open("../dataset/X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()
    pickle_out = open("../dataset/y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


create_dataset()
